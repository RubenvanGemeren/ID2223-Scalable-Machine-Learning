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

date_to_gameweek = {
        1: date(2024, 8, 16),
        2: date(2024, 8, 24),
        3: date(2024, 8, 31),
        4: date(2024, 9, 14),
        5: date(2024, 9, 21),
        6: date(2024, 9, 28),
        7: date(2024, 10, 5),
        8: date(2024, 10, 19),
        9: date(2024, 10, 25),
        10: date(2024, 11, 2),
        11: date(2024, 11, 9),
        12: date(2024, 11, 23),
        13: date(2024, 11, 29),
        14: date(2024, 12, 3),
        15: date(2024, 12, 7),
        16: date(2024, 12, 14),
        17: date(2024, 12, 21),
        18: date(2024, 12, 26),
        19: date(2024, 12, 29),
        20: date(2025, 1, 4),
        21: date(2025, 1, 14),
        22: date(2025, 1, 18),
        23: date(2025, 1, 25),
        24: date(2025, 2, 1),
        25: date(2025, 2, 14),
        26: date(2025, 2, 21),
        27: date(2025, 2, 25),
        28: date(2025, 3, 8),
        29: date(2025, 3, 15),
        30: date(2025, 4, 1),
        31: date(2025, 4, 5),
        32: date(2025, 4, 12),
        33: date(2025, 4, 19),
        34: date(2025, 4, 26),
        35: date(2025, 5, 3),
        36: date(2025, 5, 10),
        37: date(2025, 5, 18),
        38: date(2025, 5, 25),
}

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

    plt.xticks(rotation=45)

    # Ensure everything is laid out neatly
    plt.tight_layout()

    # # Save the figure, overwriting any existing file with the same name
    plt.savefig(file_path)
    return plt

def get_player_info(player_id):
    # Get static bootstrap data from fpl api
    bootstrap_url = "https://fantasy.premierleague.com/api/bootstrap-static/"

    player_data = {}

    try:
        response = requests.get(bootstrap_url)
        response.raise_for_status()
        data = response.json()

        for player in data["elements"]:
            if player["id"] == player_id:
                player_data["first_name"] = player["first_name"]
                player_data["second_name"] = player["second_name"]
                player_data["team"] = get_team_name(player["team"], data["teams"])
                player_data["position"] = get_position(player["element_type"], data["element_types"])

        return player_data

    except requests.exceptions.HTTPError as err:
        print(err)
        return None

def get_team_name(team_id, teams):
    for team in teams:
        if team["code"] == team_id:
            return team["name"]

def get_position(position_id, positions):
    for position in positions:
        if position["id"] == position_id:
            return position["singular_name"]


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

def get_date_from_gameweek(gameweek):
    try:
        return date_to_gameweek[gameweek]
    except KeyError:
        return "Invalid gameweek number"


def get_gameweek_from_date(date):
    try:
        date = pd.to_datetime(date)
    except ValueError:
        return "Invalid date format"

    first_date = pd.to_datetime(next(iter(date_to_gameweek.items())))
    last_date = pd.to_datetime(next(iter(reversed(date_to_gameweek.items()))))

    # return first_date[1], last_date[1]

    if date < first_date[1] or date > last_date[1]:
        return "Date is not within the season range"

    for i in range(1, len(date_to_gameweek)):
        if pd.to_datetime(date_to_gameweek[i]) > date:
            return i-1
