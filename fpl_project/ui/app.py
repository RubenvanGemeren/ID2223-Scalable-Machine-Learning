from flask import Flask, render_template, request
import pandas as pd
import hopsworks
import os

app = Flask(__name__)

try:
    with open("../hopsworks/hopsworks-api-key.txt", "r") as file:
        os.environ["HOPSWORKS_API_KEY"] = file.read().rstrip()
except:
    print("In production mode")

project = hopsworks.login()
fs = project.get_feature_store()

players = fs.get_feature_group("fpl_predictions")

# Mock data (replace with actual database query or file read)
sample_players = [
    {
        "firstname": "Harry",
        "lastname": "Kane",
        "club": "Tottenham",
        "position": "ST",
        "points": 100,
        "5latestGws": [
            {"gameweek": 21, "total_points": 10, "predicted_points": 8},
            {"gameweek": 20, "total_points": 7, "predicted_points": 6},
            {"gameweek": 19, "total_points": 13, "predicted_points": 10},
            {"gameweek": 18, "total_points": 5, "predicted_points": 4},
            {"gameweek": 17, "total_points": 11, "predicted_points": 9},
        ],
        "nextGwPrediction": 7,
    },
    {
        "firstname": "Erling",
        "lastname": "Haaland",
        "club": "Man City",
        "position": "ST",
        "points": 120,
        "5latestGws": [
            {"gameweek": 21, "total_points": 15, "predicted_points": 12},
            {"gameweek": 20, "total_points": 9, "predicted_points": 11},
            {"gameweek": 19, "total_points": 8, "predicted_points": 10},
            {"gameweek": 18, "total_points": 12, "predicted_points": 13},
            {"gameweek": 17, "total_points": 14, "predicted_points": 13},
        ],
        "nextGwPrediction": 8,
    },
    {
        "firstname": "Bukayo",
        "lastname": "Saka",
        "club": "Arsenal",
        "position": "MID",
        "points": 80,
        "5latestGws": [
            {"gameweek": 21, "total_points": 7, "predicted_points": 6},
            {"gameweek": 20, "total_points": 6, "predicted_points": 7},
            {"gameweek": 19, "total_points": 9, "predicted_points": 8},
            {"gameweek": 18, "total_points": 10, "predicted_points": 9},
            {"gameweek": 17, "total_points": 8, "predicted_points": 8},
        ],
        "nextGwPrediction": 6,
    },
    {
        "firstname": "Marcus",
        "lastname": "Rashford",
        "club": "Man United",
        "position": "MID",
        "points": 90,
        "5latestGws": [
            {"gameweek": 21, "total_points": 12, "predicted_points": 10},
            {"gameweek": 20, "total_points": 10, "predicted_points": 9},
            {"gameweek": 19, "total_points": 11, "predicted_points": 10},
            {"gameweek": 18, "total_points": 14, "predicted_points": 12},
            {"gameweek": 17, "total_points": 13, "predicted_points": 11},
        ],
        "nextGwPrediction": 9,
    },
    {
        "firstname": "Mohamed",
        "lastname": "Salah",
        "club": "Liverpool",
        "position": "MID",
        "points": 70,
        "5latestGws": [
            {"gameweek": 21, "total_points": 8, "predicted_points": 7},
            {"gameweek": 20, "total_points": 6, "predicted_points": 8},
            {"gameweek": 19, "total_points": 7, "predicted_points": 9},
            {"gameweek": 18, "total_points": 11, "predicted_points": 10},
            {"gameweek": 17, "total_points": 9, "predicted_points": 8},
        ],
        "nextGwPrediction": 7,
    },
]


@app.route("/")
def index():
    """Render the main page."""
    return render_template("players.html", players=sample_players)


@app.route("/api/players", methods=["POST", "GET"])
def get_players():
    """API endpoint to fetch players with all 5 gameweeks, filtering by player name and position if provided."""
    player_name = request.form.get(
        "player", ""
    ).lower()  # Get search query for player name
    position = request.form.get("position", "").upper()  # Get search query for position

    # Filter players based on search term in firstname, lastname or position
    filtered_players = [
        player
        for player in sample_players
        if (
            player_name in player["firstname"].lower()
            or player_name in player["lastname"].lower()
        )
        and (not position or position == player["position"])
    ]

    # If no search filter is provided, show all players by default
    if not player_name and not position:
        filtered_players = sample_players

    # Generate the HTML table rows for filtered players
    player_rows = ""
    for player in filtered_players:
        player_rows += f"""
        <tr>
            <td>{player['firstname']} {player['lastname']}</td>
            <td>{player['position']}</td>
            <td>{player['club']}</td>
            <td>{player['points']}</td>
            <td>
                <table class="inner-table">
                    <tr>
                        <th>Gameweek</th>
                        <th>Points</th>
                        <th>Predicted Points</th>
                    </tr>
        """
        for gw in player["5latestGws"]:
            player_rows += f"""
            <tr>
                <td>{gw['gameweek']}</td>
                <td>{gw['total_points']}</td>
                <td>{gw['predicted_points']}</td>
            </tr>
            """
        player_rows += f"""</table></td>
        <td>{player["nextGwPrediction"]}</td</tr>"""

    return player_rows


if __name__ == "__main__":
    app.run(debug=True)
