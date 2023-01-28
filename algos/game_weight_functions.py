"""
Functions used in the stage 2 of the BlockRankingAlgorithm class ranking procedure - obtaining game weight
for each game in the Games Dataset (they are not intended to be called directly).
During the ranking procedure, get_game_weight wrapper function is called and it will call desired game-weight function.
Other, specific game-weight functions should have format game_wght = game_weight_func(**kwargs).
"""
import datetime
import typing as t

import numpy as np
import pandas as pd


def get_game_weight(option: t.Optional[t.Union[str, t.Callable]], game_row: pd.Series, **kwargs):
    """
    Apply specific game-weight function defined by "option" on the given "game row" from the Games Table.

    :param option: Identifier of the game-weight function
    :param game_row: Row from the Games Table
    :param kwargs: Additional arguments to the game-weight function (besides score_w and score_l)
    :return: Rate difference for the given game
    """
    if option in ["uniform", None, uniform_game_weight_function]:
        game_wght = uniform_game_weight_function()
    elif option in ["usau", usau_game_weight_function]:
        # Use Monday as part of the previous week to resolve the issue of tournaments with some games played on Monday
        w_num = (datetime.datetime.strptime(game_row["Date"], "%Y-%m-%d") - datetime.timedelta(days=1)).isocalendar()[1]
        game_wght = usau_game_weight_function(game_row["Score_1"], game_row["Score_2"], w_num, **kwargs)
    elif option in ["usau_no_date", usau_no_date_game_weight_function]:
        game_wght = usau_no_date_game_weight_function(game_row["Score_1"], game_row["Score_2"])
    else:
        raise ValueError("Unknown game-weight option, make sure it is defined in algos/game_weight_functions.py.")

    return game_wght


def uniform_game_weight_function() -> float:
    """
    Give weight 1 to every game.
    """
    return 1


def usau_game_weight_function(
    score_w: int, score_l: int, w_num: int, w0: float = 0.5, w_first: int = 24, w_last: int = 42
) -> float:
    """
    Return USAU game-weight function, source https://play.usaultimate.org/teams/events/rankings/.

    :param score_w: Winning score
    :param score_l: Losing score
    :param w_num: Calendar week
    :param w0: Weight of the first week
    :param w_first: First calendar week of the season
    :param w_last: Last calendar week of the season
    :return: Game weight
    """
    score_wght = np.min([1, np.sqrt((score_w + np.max([score_l, np.floor(0.5 * (score_w - 1))])) / 19)])
    if w_num >= w_last:
        date_wght = 1
    else:
        date_wght = w0 * ((1 / w0) ** (1 / (w_last - w_first))) ** (w_num - w_first)
    return score_wght * date_wght


def usau_no_date_game_weight_function(score_w: int, score_l: int) -> float:
    """
    Return USAU game-weight function, source https://play.usaultimate.org/teams/events/rankings/ without
    the date component.

    :param score_w: Winning score
    :param score_l: Losing score
    :return: Game weight
    """
    return np.min([1, np.sqrt((score_w + np.max([score_l, np.floor(0.5 * (score_w - 1))])) / 19)])
