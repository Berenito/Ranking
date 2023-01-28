"""
Define the RankingAlgorithm class.
"""
import typing as t

import pandas as pd


class RankingAlgorithm:
    """
    Class to work with ranking algorithm.
    """

    def __init__(self, rating_func: t.Callable, algo_name: str = "Unknown_Algo", **kwargs):
        """
        Initialize the algorithm.

        Examples:
            example_algo = RankingAlgorithm(example_func, algo_name="Example_Algo", p1=100, p2=200)
            usau_algo = RankingAlgorithm(get_usau_ratings, algo_name="USAU_Algo")

        :param rating_func: Function (df_games, **kwargs) -> ratings
        :param algo_name: Used for naming exported stuff
        :param kwargs: Additional inputs to rating_func
        :return: Initialized RankingAlgorithm object
        """
        self.name = algo_name
        self.rating_func = rating_func
        self.params = kwargs

    def get_ratings(self, df_games: pd.DataFrame, date: t.Optional[str] = None) -> pd.Series:
        """
        Calculate the ratings of the provided dataset.

        :param df_games: Games Table
        :param date: Date in YYYY-MM-DD format (only games up to it will be included)
        :return: Series with the calculated ratings
        """
        df_games = df_games.copy()
        if date is not None:
            df_games = df_games.loc[df_games["Date"] <= date]
        ratings = self.rating_func(df_games, **self.params)
        ratings = ratings.rename("Rating_{}".format(self.name))
        return ratings
