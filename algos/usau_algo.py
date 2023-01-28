"""
USAU Ranking Algo, source https://play.usaultimate.org/teams/events/rankings/.
With (hopefully) better iteration procedure, more robust to smaller components.
This is a version, which can be called from standard RankingAlgorithm class, there is also
block-based implementation, which can be called from BlockRankingAlgorithm class.
"""
import datetime
import logging

import numpy as np
import pandas as pd

from algos.game_ignore_functions import blowout_game_ignore_function
from algos.game_weight_functions import usau_game_weight_function
from algos.rank_diff_functions import usau_rank_diff_function
from utils import dataset

_logger = logging.getLogger("ranking.usau_algo")


def run_ranking_iteration(ratings_start: pd.Series, df_games: pd.DataFrame, n_round: int = 2):
    """
    Set new rating of each team as the weighted average of the ratings of their un-ignored games calculated using the
    current ratings (original version). In this version, you go to ratings_start + 0.5*(ratings_new - ratings_start),
    which has better convergence for smaller number of teams.
    """
    df_games = df_games.copy()
    df_games["Rank_1"] = df_games["Game_Rank_Diff"] + ratings_start.reindex(df_games["Team_2"]).values
    df_games["Rank_2"] = ratings_start.reindex(df_games["Team_1"]).values - df_games["Game_Rank_Diff"]
    df_games["Team_Rank_Diff"] = ratings_start.reindex(df_games["Team_1"]).values - ratings_start.reindex(df_games["Team_2"]).values
    df_games["Is_Ignored"] = blowout_game_ignore_function(df_games, ratings_start)
    df_games.loc[df_games["Is_Ignored"] == 1, "Game_Wght"] = 0
    df_games_extended = pd.concat([df_games.rename(columns={"Team_1": "Team", "Rank_1": "Rank"}),
                                   df_games.rename(columns={"Team_2": "Team", "Rank_2": "Rank"})])
    ratings_new = df_games_extended.groupby("Team").apply(lambda x: dataset.safe_weighted_avg(x["Rank"], x["Game_Wght"]))
    # in the original version, 1 is used instead of 0.5
    ratings_new = (ratings_start + 0.5*(ratings_new - ratings_start)).round(n_round)

    return ratings_new


def get_usau_ratings(df_games: pd.DataFrame, rating_start: float, w0: float, w_first: int, w_last: int) -> pd.Series:
    """
    Wrapper function to be called from RankingAlgorithm class.
    Params: rating_start (1000 in standard version),
            w0, w_first, w_last (date-weight related, optional, default values 0.5, 24, 42 as for 2021)
    """
    df_games["Game_Rank_Diff"] = df_games.apply(
        lambda rw: usau_rank_diff_function(rw["Score_1"], rw["Score_2"]), axis=1
    )
    df_games["Calendar_Week"] = df_games['Date'].apply(
        lambda x: (datetime.datetime.strptime(x, '%Y-%m-%d') - datetime.timedelta(days=1)).isocalendar()[1]
    )
    df_games["Game_Wght"] = df_games.apply(
        lambda rw: usau_game_weight_function(rw["Score_1"], rw["Score_2"], rw["Calendar_Week"], w0, w_first, w_last), axis=1
    )
    df_games["Is_Blowout"] = 1 * (df_games["Score_1"] > 2 * df_games["Score_2"] + 1)

    teams = dataset.get_teams_in_games(df_games)
    n_iter, dist_tol = 1000, 1e-5
    df_ratings_iter = pd.DataFrame(rating_start, index=teams, columns=range(n_iter + 1))
    for i in range(n_iter):
        df_ratings_iter[i + 1] = run_ranking_iteration(df_ratings_iter[i], df_games)
        dist_iter = np.sqrt(((df_ratings_iter[i + 1] - df_ratings_iter[i]) ** 2).mean())
        _logger.info(f"{i + 1}/{n_iter}, Change: {dist_iter:.7f}, Rating 1: {df_ratings_iter[i + 1].max():.4f}.")
        if dist_iter < dist_tol:
            n_iter = i + 1
            break
    df_ratings_iter = df_ratings_iter[range(n_iter + 1)]
    ratings_final = df_ratings_iter[n_iter].sort_values(ascending=False)
    return ratings_final
