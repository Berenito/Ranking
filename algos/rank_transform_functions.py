"""
Functions used to convert the team rank difference to the game rank difference.
"""
import typing as t

import numpy as np
import pandas as pd


def get_transformed_rank_diff(
    option: t.Union[str, t.Callable], ratings_1: pd.Series, ratings_2: pd.Series, **kwargs
) -> pd.Series:
    """

    """
    if option in [None, "identity"]:
        return ratings_1 - ratings_2
    elif option in ["sigmoid", sigmoid_rank_transform_function]:
        return sigmoid_rank_transform_function(ratings_1 - ratings_2, **kwargs)
    else:
        raise ValueError("Unknown rank-transform option, make sure it is defined in algos/rank_transform_functions.py.")


def sigmoid_rank_transform_function(diff_ratings: pd.Series, diff_max: int = 15) -> pd.Series:
    """

    """
    return 2 * diff_max / (1 + np.exp(-diff_ratings / 100)) - diff_max
