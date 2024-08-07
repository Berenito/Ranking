"""
Based on the data and rating algorithm, visualize the optimal team rating difference that maximize the game rating gain.
"""

import pandas as pd
import plotly.graph_objects as go

from classes.games_dataset import GamesDataset
from classes.block_ranking_algorithm import BlockRankingAlgorithm
from definitions import USAU_ALGO, WINDMILL_ALGO

WIN_LOSE_ALGO = BlockRankingAlgorithm(
    algo_name="WinLose",
    rank_diff_func="win_lose",
    game_weight_func="uniform",
    rank_fit_func="regression",
    rank_fit_params={"n_round": 2}
)

ROLLING_WINDOW = 50
DATE = "2024-08-05"
DATA_COLS = ["Division", "Tournament", "Date", "Team_1", "Team_2", "Score_1", "Score_2"]

for algo in [USAU_ALGO, WINDMILL_ALGO, WIN_LOSE_ALGO]:
    viz_list = []
    for division in ["open", "women", "mixed"]:
        df_info = pd.read_csv(f"C:/Users/micha/Desktop/Python/euf_ranking/2024-{division}-euf-games-20240805.csv")
        df_info["Division"] = division.capitalize()
        dataset = GamesDataset(df_info[DATA_COLS], name=f"2024-euf", date=DATE)
        dataset.add_ratings(algo, block_algo=True)
        df_viz_division = dataset.games.copy().set_index(DATA_COLS)
        df_viz_division["team_rank_diff"] = df_viz_division[f"Team_Rank_Diff_{algo.name}"]
        df_viz_division["game_impact"] = (
            (df_viz_division[f"Game_Rank_Diff_{algo.name}"] - df_viz_division[f"Team_Rank_Diff_{algo.name}"])
            * (1 - df_viz_division[f"Is_Ignored_{algo.name}"])
        )
        df_viz_division = df_viz_division[["team_rank_diff", "game_impact"]]
        viz_list.append(df_viz_division)
    df_viz = pd.concat(viz_list)
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
                "Division: %{customdata[0]}<br>"
                "Tournament: %{customdata[1]}<br>"
                "Date: %{customdata[2]}<br>"
                "Teams: %{customdata[3]} vs. %{customdata[4]}<br>"
                "Score: %{customdata[5]}-%{customdata[6]}"
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



