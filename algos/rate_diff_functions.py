"""
Functions used in the stage 1 of the BlockRankingAlgorithm class ranking procedure - obtaining rate difference
(rate-diff) for each game in the Games Dataset (they are not intended to be called directly).
During the ranking procedure, get_rate_diff wrapper function is called and it will call desired rate-diff function.
Other, specific rate-diff functions should have format r_diff = rate_diff_func(score_w, score_l, **kwargs).
"""
import typing as t

import numpy as np
import pandas as pd


def get_rate_diff(option: t.Union[str, t.Callable], game_row: pd.Series, **kwargs) -> float:
    """
    Wrapper function to apply specific rate-diff function defined by "option" on the given "game row" from the
    Games Table.
    
    :param option: Identifier of the rate-diff function
    :param game_row: Row from the Games Table
    :param kwargs: Additional arguments to the rate-diff function (besides score_w and score_l)
    :return: Rate difference for the given game
    """
    score_w, score_l = game_row["Score_1"], game_row["Score_2"]
    if option in ["win_lose", win_lose_rate_diff_function]:
        r_diff = win_lose_rate_diff_function(score_w, score_l)
    elif option in ["score_diff", "windmill", score_diff_rate_diff_function]:
        r_diff = score_diff_rate_diff_function(score_w, score_l)
    elif option in ["usau", usau_rate_diff_function]:
        r_diff = usau_rate_diff_function(score_w, score_l)
    else:
        raise ValueError("Unknown rate-diff option, make sure it is defined in algos/rate_diff_functions.py.")
    return r_diff


def win_lose_rate_diff_function(score_w: int, score_l: int) -> float:
    """
    Always 1 for the winner (unless it is a draw), regardless of the score.

    :param score_w: Winning score
    :param score_l: Losing score
    :return: Game rate difference
    """
    return np.sign(score_w - score_l)


def score_diff_rate_diff_function(score_w: int, score_l: int) -> float:
    """
    Basic score difference, used in Windmill algo.

    :param score_w: Winning score
    :param score_l: Losing score
    :return: Game rate difference
    """
    return score_w - score_l


def usau_rate_diff_function(score_w: int, score_l: int) -> float:
    """
    USAU rate-diff function, source https://play.usaultimate.org/teams/events/rankings/.

    :param score_w: Winning score
    :param score_l: Losing score
    :return: Game rate difference
    """
    if score_w == score_l:
        r_diff = 0
    else:
        r = score_l / (score_w - 1) if score_w != 1 else 1
        r_diff = 125 + 475 * (np.sin(np.min([1, 2 * (1 - r)]) * 0.4 * np.pi) / np.sin(0.4 * np.pi))
    return r_diff
