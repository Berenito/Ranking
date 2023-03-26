"""
Functions used during the rank-fitting stage of the BlockRankingAlgorithm class ranking procedure
(they are not intended to be called directly).
They are used to convert the team rank difference to the domain of game rank difference.
During the ranking procedure, get_transformed_rank_diff wrapper function is called and it will call desired
rank-transform function.
"""
import typing as t

import numpy as np
import pandas as pd


def get_transformed_rank_diff(
    option: t.Union[str, t.Callable], ratings_1: pd.Series, ratings_2: pd.Series, **kwargs
) -> pd.Series:
    """
    Apply specific rank-transform function given by "option" on the current ratings.

    :param option: Rank-transform function identifier
    :param ratings_1: Ratings of the first team in the Games Table
    :param ratings_2: Ratings of the second team in the Games Table
    :param kwargs: Additional parameters to the rank-transform function
    :return: Series with transformed team rating differences
    """
    if option in [None, "identity"]:
        return ratings_1 - ratings_2
    elif option in ["sigmoid", sigmoid_rank_transform_function]:
        return sigmoid_rank_transform_function(ratings_1 - ratings_2, **kwargs)
    else:
        raise ValueError("Unknown rank-transform option, make sure it is defined in algos/rank_transform_functions.py.")


def sigmoid_rank_transform_function(diff_ratings: pd.Series, diff_max: int = 15) -> pd.Series:
    """
    Return the difference in ratings transformed by the sigmoid function.

    :param diff_ratings: Difference in the ratings
    :param diff_max: Asymptote of the sigmoid function
    :return: Transformed difference in ratings
    """
    return 2 * diff_max / (1 + np.exp(-diff_ratings / 100)) - diff_max
