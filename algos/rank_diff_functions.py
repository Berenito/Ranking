"""
Functions used in the stage 1 of the BlockRankingAlgorithm class ranking procedure - obtaining rank difference
(rank-diff) for each game in the Games Dataset (they are not intended to be called directly).
During the ranking procedure, get_rank_diff wrapper function is called and it will call desired rank-diff function.
Other, specific rank-diff functions should have format r_diff = rank_diff_func(score_w, score_l, **kwargs).
"""
import typing as t

import numpy as np
import pandas as pd


def get_rank_diff(option: t.Union[str, t.Callable], game_row: pd.Series, **kwargs) -> float:
    """
    Apply specific rank-diff function defined by "option" on the given "game row" from the Games Table.
    
    :param option: Identifier of the rank-diff function
    :param game_row: Row from the Games Table
    :param kwargs: Additional arguments to the rank-diff function (besides score_w and score_l)
    :return: rank difference for the given game
    """
    score_w, score_l = game_row["Score_1"], game_row["Score_2"]
    if option in ["win_lose", win_lose_rank_diff_function]:
        r_diff = win_lose_rank_diff_function(score_w, score_l)
    elif option in ["score_diff", "windmill", score_diff_rank_diff_function]:
        r_diff = score_diff_rank_diff_function(score_w, score_l)
    elif option in ["usau", usau_rank_diff_function]:
        r_diff = usau_rank_diff_function(score_w, score_l)
    else:
        raise ValueError("Unknown rank-diff option, make sure it is defined in algos/rank_diff_functions.py.")
    return r_diff


def win_lose_rank_diff_function(score_w: int, score_l: int) -> float:
    """
    Always return 1 for the winner (unless it is a draw), regardless of the score.

    :param score_w: Winning score
    :param score_l: Losing score
    :return: Game rank difference
    """
    return np.sign(score_w - score_l)


def score_diff_rank_diff_function(score_w: int, score_l: int) -> float:
    """
    Return basic score difference, used in Windmill algo.

    :param score_w: Winning score
    :param score_l: Losing score
    :return: Game rank difference
    """
    return score_w - score_l


def usau_rank_diff_function(score_w: int, score_l: int) -> float:
    """
    Return USAU rank-diff function, source https://play.usaultimate.org/teams/events/rankings/.

    :param score_w: Winning score
    :param score_l: Losing score
    :return: Game rank difference
    """
    if score_w == score_l:
        r_diff = 0
    else:
        r = score_l / (score_w - 1) if score_w != 1 else 1
        r_diff = 125 + 475 * (np.sin(np.min([1, 2 * (1 - r)]) * 0.4 * np.pi) / np.sin(0.4 * np.pi))
    return r_diff
