import os
from datetime import date
import time
import requests
import pandas as pd
import json
from geopy.geocoders import Nominatim
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
import openmeteo_requests
import requests_cache
from retry_requests import retry
import hopsworks
import hsfs
from pathlib import Path

def plot_player_score_forecast(
    graph_name: str, df: pd.DataFrame, file_path: str, hindcast=False
):
    fig, ax = plt.subplots(figsize=(10, 8))

    # day = pd.to_datetime(df["date"]).dt.date
    # day = date.today()

    # Plot each column separately in matplotlib
    ax.plot(
        df["predicted_score"],
        label="Predicted Score",
        color="red",
        linewidth=2,
        linestyle='None',
        marker="o",
        markersize=3,
    )

    # Set the y-axis to a logarithmic scale
    ax.set_yscale("log")
    ax.set_yticks([0, 1, 3, 5, 7, 9, 11, 13, 15])
    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_ylim(bottom=-3)

    # Set the labels and title
    ax.set_xlabel("Player")
    ax.set_title(f"Graph of {graph_name}")
    ax.set_ylabel("Score")

    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # colors = ["green", "yellow", "orange", "red", "purple", "darkred"]
    # labels = [
    #     "Good",
    #     "Moderate",
    #     "Unhealthy for Some",
    #     "Unhealthy",
    #     "Very Unhealthy",
    #     "Hazardous",
    # ]
    # ranges = [(0, 49), (50, 99), (100, 149), (150, 199), (200, 299), (300, 500)]
    # for color, (start, end) in zip(colors, ranges):
    #     ax.axhspan(start, end, color=color, alpha=0.3)

    # Add a legend for the different Air Quality Categories
    # patches = [
    #     Patch(color=colors[i], label=f"{labels[i]}: {ranges[i][0]}-{ranges[i][1]}")
    #     for i in range(len(colors))
    # ]
    # legend1 = ax.legend(
    #     handles=patches,
    #     loc="upper right",
    #     title="Air Quality Categories",
    #     fontsize="x-small",
    # )

    # Aim for ~10 annotated values on x-axis, will work for both forecasts ans hindcasts
    # if len(df.index) > 11:
    #     every_x_tick = len(df.index) / 10
    #     ax.xaxis.set_major_locator(MultipleLocator(every_x_tick))

    plt.xticks(rotation=45)

    if hindcast == True:
        ax.plot(
            day,
            df["total_points"],
            label="Actual Score",
            color="black",
            linewidth=2,
            marker="^",
            markersize=5,
            markerfacecolor="grey",
        )
        legend2 = ax.legend(loc="upper left", fontsize="x-small")
        ax.add_artist(legend1)

    # Ensure everything is laid out neatly
    plt.tight_layout()

    # # Save the figure, overwriting any existing file with the same name
    plt.savefig(file_path)
    return plt


def delete_feature_groups(fs, name):
    try:
        for fg in fs.get_feature_groups(name):
            fg.delete()
            print(f"Deleted {fg.name}/{fg.version}")
    except hsfs.client.exceptions.RestAPIError:
        print(f"No {name} feature group found")


def delete_feature_views(fs, name):
    try:
        for fv in fs.get_feature_views(name):
            fv.delete()
            print(f"Deleted {fv.name}/{fv.version}")
    except hsfs.client.exceptions.RestAPIError:
        print(f"No {name} feature view found")


def delete_models(mr, name):
    models = mr.get_models(name)
    if not models:
        print(f"No {name} model found")
    for model in models:
        model.delete()
        print(f"Deleted model {model.name}/{model.version}")


def delete_secrets(proj, name):
    secrets = secrets_api(proj.name)
    try:
        secret = secrets.get_secret(name)
        secret.delete()
        print(f"Deleted secret {name}")
    except hopsworks.client.exceptions.RestAPIError:
        print(f"No {name} secret found")


# WARNING - this will wipe out all your feature data and models
def purge_project(proj):
    fs = proj.get_feature_store()
    mr = proj.get_model_registry()

    # Delete Feature Views before deleting the feature groups
    delete_feature_views(fs, "player_score_fv")

    # Delete ALL Feature Groups
    delete_feature_groups(fs, "player_score")

    # Delete all Models
    delete_models(mr, "player_score_xgboost_model")

    # Delete all Secrets
    # delete_secrets(proj, "??")


def secrets_api(proj):
    host = "c.app.hopsworks.ai"
    api_key = os.environ.get("HOPSWORKS_API_KEY")
    conn = hopsworks.connection(host=host, project=proj, api_key_value=api_key)
    return conn.get_secrets_api()


def check_file_path(file_path):
    my_file = Path(file_path)
    if my_file.is_file() == False:
        print(f"Error. File not found at the path: {file_path} ")
    else:
        print(f"File successfully found at the path: {file_path}")
