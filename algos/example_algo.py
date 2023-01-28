"""
Easy algo for example RankingAlgorithm usage.
"""
import pandas as pd

from utils import dataset


def get_example_ratings(df_games: pd.DataFrame, lb: float, ub: float) -> pd.Series:
    """
    An easy example of rating function; calculate W_Ratio * Opponent_W_Ratio and transform (0, 1) -> (lb, ub).
    Lb and ub should be defined during RankingAlgorithm initialization.

    :param df_games: Games Table
    :param lb: Lower bound of the rating interval
    :param ub: Upper bound of the rating interval
    :return: Series with the calculated ratings
    """
    df_summary = dataset.get_summary_of_games(df_games)
    ratings = df_summary["W_Ratio"] * df_summary["Opponent_W_Ratio"]
    ratings = lb + ratings*(ub - lb)
    
    return ratings
