"""
Functions used in the stage 3 of the BlockRankingAlgorithm class ranking procedure - obtaining fitted rating for each
team in the Games Dataset (they are not intended to be called directly).
During the fitting procedure, we call additional, game-ignore function (game_ignore_func), which can give 0 weight
to "rank-damaging" blowout wins for the good teams (and vice versa). Additional parameters to game_ignored_func are
provided in game_ignore_kwargs dictionary.
During the ranking procedure, get_rank_fit wrapper function is called and it will call desired rank-fit function.
Other, specific rank-fit functions can potentially use the following parameters: game_ignore_func, teams, df_games,
df_components, game_ignore_kwargs and function-specific **kwargs.
"""
import logging
import typing as t

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from algos.game_ignore_functions import get_ignored_games
from utils.dataset import safe_weighted_avg

_logger = logging.getLogger("ranking.algos.rank_fit")


def get_rank_fit(
    option: t.Union[str, t.Callable], 
    game_ignore_func: t.Optional[t.Union[str, t.Callable]], 
    teams: pd.Series, 
    df_games: pd.DataFrame, 
    components: pd.Series,
    game_ignore_kwargs: t.Dict, 
    **kwargs,
) -> t.Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Apply specific rank-fit function given by "option" on the given Games Dataset.
    
    :param option: Rank-fit function identifier
    :param game_ignore_func: Game-ignore function identifier
    :param teams: Series of teams
    :param df_games: Games Table
    :param components: Graph component of each team
    :param game_ignore_kwargs: Additional arguments to the game-ignore function
    :param kwargs: Additional arguments to the rank-fit function
    :return: Series with the calculated ratings; Series with ignored games; Series with team rate difference per game
    """
    df_games = df_games.copy()
    if option in ["regression", regression_rank_fit_function]:
        ratings = regression_rank_fit_function(teams, df_games, components, **kwargs)
    elif option in ["iteration", iteration_rank_fit_function]:
        ratings = iteration_rank_fit_function(teams, df_games, game_ignore_func, game_ignore_kwargs, **kwargs)
    else:
        raise ValueError("Unknown rank-fit option, make sure it is defined in algos/rank_fit_functions.py.")

    df_games["Team_1_Rank"] = ratings.reindex(df_games["Team_1"]).values
    df_games["Team_2_Rank"] = ratings.reindex(df_games["Team_2"]).values
    df_games["Team_Rank_Diff"] = df_games["Team_1_Rank"] - df_games["Team_2_Rank"]
    is_ignored = get_ignored_games(game_ignore_func, df_games, ratings, **game_ignore_kwargs)
    return ratings, is_ignored, df_games["Team_Rank_Diff"]


def regression_rank_fit_function(
    teams: pd.Series, df_games: pd.DataFrame, components: pd.Series, mean_rating: int = 0, n_round: int = 10
):
    """
    Fit the ratings with the standard (weighted) linear regression. Used in Windmill algorithm.

    Does not support game_ignore_func at the moment.

    :param teams: Series of teams
    :param df_games: Games Table
    :param components: Graph component of each team
    :param mean_rating: Mean rating
    :param n_round: Number of decimals for rounding
    :return: Series with the calculated ratings (LR coefficients)
    """
    coefficients = pd.Series(index=teams)
    for i_comp, comp in components.reset_index().groupby("Component"):
        teams_comp = comp["Team"]
        df_comp = df_games.loc[df_games["Team_1"].isin(teams_comp) | df_games["Team_2"].isin(teams_comp)].reset_index()
        df_comp["const"] = 1
        df_plus = df_comp.pivot(index="index", columns="Team_1", values="const").fillna(0)
        df_minus = df_comp.pivot(index="index", columns="Team_2", values="const").fillna(0)
        x = df_plus.subtract(df_minus, fill_value=0)[teams_comp]
        y = df_comp["Game_Rank_Diff"]
        wghts = df_comp["Game_Wght"]
        lr = LinearRegression(fit_intercept=False).fit(x, y, sample_weight=wghts)
        coefficients[teams_comp] = (lr.coef_ - lr.coef_.mean() + mean_rating).round(n_round)
    return coefficients


def iteration_rank_fit_function(
    teams: pd.Series,
    df_games: pd.DataFrame,
    game_ignore_func: t.Optional[t.Union[str, t.Callable]],
    game_ignore_kwargs: t.Dict,
    n_iter: int = 1000,
    tol: float = 1e-8,
    rating_start: int = 1000,
    n_round: int = 10,
    verbose: bool = True,
):
    """
    Fit the ratings with the iteration procedure.

    In every iteration, do the following:
        * get ignored games based on ratings_start
        * get ratings_new as the weighted average of the ratings for each game
        * set ratings to ratings_start + 0.5 * (ratings_new - ratings_start) - better convergence than using ratings_new
    Repeat until desired convergence is reached.

    :param teams: Series of teams
    :param df_games: Games Table
    :param game_ignore_func: Game-ignore function identifier
    :param game_ignore_kwargs: Additional arguments to the game-ignore function
    :param n_iter: Maximum number of iterations
    :param tol: Tolerance of RMSE between ratings from subsequent iterations
    :param rating_start: Starting rating
    :param verbose: Whether to print the information
    :param n_round: Number of decimals for rounding
    :return: Series with the calculated ratings
    """
    df_ratings_iter = pd.DataFrame(rating_start, index=teams, columns=range(n_iter + 1))
    for i in range(n_iter):
        ratings_iter = df_ratings_iter[i]
        df_games_iter = df_games.copy()
        df_games_iter["Team_1_Rank"] = ratings_iter.reindex(df_games_iter["Team_1"]).values
        df_games_iter["Team_2_Rank"] = ratings_iter.reindex(df_games_iter["Team_2"]).values
        df_games_iter["Team_Rank_Diff"] = df_games_iter["Team_1_Rank"] - df_games_iter["Team_2_Rank"]
        df_games_iter["Is_Ignored"] = get_ignored_games(
            game_ignore_func, df_games_iter, ratings_iter, **game_ignore_kwargs
        )
        df_games_iter.loc[df_games_iter["Is_Ignored"] == 1, "Game_Wght"] = 0
        df_games_iter["Rank_1"] = df_games_iter["Game_Rank_Diff"] + df_games_iter["Team_2_Rank"]
        df_games_iter["Rank_2"] = df_games_iter["Team_1_Rank"] - df_games_iter["Game_Rank_Diff"]
        # Duplicate the games, such that we can make the weighted average of game ranks with one groupby call
        df_games_extended = pd.concat(
            [
                df_games_iter.rename(columns={"Team_1": "Team", "Rank_1": "Rank"}),
                df_games_iter.rename(columns={"Team_2": "Team", "Rank_2": "Rank"}),
            ]
        )
        ratings_new = df_games_extended.groupby("Team").apply(lambda x: safe_weighted_avg(x["Rank"], x["Game_Wght"]))
        df_ratings_iter[i + 1] = (ratings_iter + 0.5 * (ratings_new - ratings_iter)).round(n_round)
        rmse_change = np.sqrt(((df_ratings_iter[i + 1] - df_ratings_iter[i]) ** 2).mean())
        if verbose:
            _logger.info(f"{i + 1}/{n_iter}, Change: {rmse_change:.7f}, Rating 1: {df_ratings_iter[i + 1].max():.4f}.")
        if rmse_change < tol:
            n_iter = i + 1
            break
    df_ratings_iter = df_ratings_iter[range(n_iter + 1)]
    return df_ratings_iter[n_iter]


