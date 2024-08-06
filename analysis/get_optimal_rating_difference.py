"""
Based on the data and rating algorithm, visualize the optimal team rating difference that maximize the game rating gain.
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from classes.games_dataset import GamesDataset
from definitions import USAU_ALGO, WINDMILL_ALGO

ROLLING_WINDOW = 20
DATE = "2024-08-05"
DATA_COLS = ["Tournament", "Date", "Team_1", "Team_2", "Score_1", "Score_2"]
path_info = Path("C:/Users/micha/Desktop/Python/euf_ranking/2024-open-euf-games-20240805.csv")
df_info = pd.read_csv(path_info)[DATA_COLS]

dataset = GamesDataset(df_info, name=f"2024-open-euf", date=DATE)

for algo in [USAU_ALGO, WINDMILL_ALGO]:
    dataset.add_ratings(algo, block_algo=True)
    df_viz = dataset.games.copy().set_index(DATA_COLS)
    df_viz["team_rank_diff"] = df_viz[f"Team_Rank_Diff_{algo.name}"]
    df_viz["game_impact"] = (
        (df_viz[f"Game_Rank_Diff_{algo.name}"] - df_viz[f"Team_Rank_Diff_{algo.name}"])
        * (1 - df_viz[f"Is_Ignored_{algo.name}"])
    )
    df_viz = df_viz[["team_rank_diff", "game_impact"]]
    df_viz.loc[df_viz["team_rank_diff"] < 0] *= -1
    df_viz = df_viz.sort_values(by="team_rank_diff")
    df_rolling = df_viz.rolling(ROLLING_WINDOW).mean()
    df_viz = df_viz.reset_index()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_viz["team_rank_diff"],
            y=df_viz["game_impact"],
            mode="markers",
            marker_opacity=0.5,
            marker_size=10,
            customdata=df_viz[DATA_COLS],
            hovertemplate=(
                "Tournament: %{customdata[0]}<br>"
                "Date: %{customdata[1]}<br>"
                "Teams: %{customdata[2]} vs. %{customdata[3]}<br>"
                "Score: %{customdata[4]}- %{customdata[5]}"
                "<extra></extra>"
            )
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_rolling["team_rank_diff"],
            y=df_rolling["game_impact"],
            mode="lines",
            line_width=5,
            name=f"Rolling average with window {ROLLING_WINDOW}",
        )
    )
    fig.update_layout(
        showlegend=False,
        title_text=f"Division: Open, Season: 2024, Date: {DATE}, Algorithm: {algo.name}",
        xaxis_title_text="Team rating difference",
        yaxis_title_text="Game rating impact",
    )
    fig.write_html(f"C:/Users/micha/Desktop/Python/euf_ranking/optimal_diff_{algo.name}.html")



