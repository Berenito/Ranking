"""
Define the BlockRankingAlgorithm class. It provides a very general framework to define the ranking algorithm. 
Many already existing ranking algorithms can be defined with its help; i.e., both Windmill and USAU algorithms.
In addition to ratings provides also additional per-game statistics like expected and true game rate differences.
"""
import typing as t

import pandas as pd

from algos.rank_diff_functions import get_rank_diff
from algos.game_weight_functions import get_game_weight
from algos.rank_fit_functions import get_rank_fit
from utils import dataset


class BlockRankingAlgorithm:
    """
    Class to work with ranking algorithm defined with the help of blocks.
    """

    def __init__(
        self, 
        rank_diff_func: t.Union[str, t.Callable] = "win_lose",
        game_weight_func: t.Union[str, t.Callable] = "uniform", 
        rank_fit_func: t.Union[str, t.Callable] = "regression",
        game_ignore_func: t.Optional[t.Union[str, t.Callable]] = None, 
        algo_name: str = "Unknown_Algo", 
        rank_diff_params: t.Dict = {},
        game_weight_params: t.Dict = {}, 
        rank_fit_params: t.Dict = {}, 
        game_ignore_params: t.Dict = {}
    ):
        """
        Initialize the algorithm.
        
        Examples:
            windmill_algo = BlockRankingAlgorithm(
                algo_name="Windmill", rank_diff_func="score_diff", game_weight_func="uniform", 
                rank_fit_func="regression", rank_fit_params={"n_round": 2}
            )
            usau_algo = BlockRankingAlgorithm(
                algo_name="USAU", rank_diff_func="usau", game_weight_func="usau", rank_fit_func="iteration",
                game_ignore_func="blowout", game_weight_params={"w0": 0.5, "w_first": 29, "w_last": 42},
                rank_fit_params={"rating_start": 1000, "n_round": 2, "n_iter": 1000}
            )

        :param rank_diff_func: Rank-diff function identifier
        :param game_weight_func: Game-weight function identifier
        :param rank_fit_func: Rank-fit function identifier
        :param game_ignore_func: Game-ignore function identifier
        :param algo_name: Used for naming exported stuff
        :param rank_diff_params: Additional inputs to rank_diff_func
        :param game_weight_params: Additional inputs to game_weight_func
        :param rank_fit_params: Additional inputs to rank_fit_func
        :param game_ignore_params: Additional inputs to game_ignore_func
        :return: Initialized BlockRankingAlgorithm object
        """
        self.name = algo_name
        self.rank_diff_func = rank_diff_func
        self.game_weight_func = game_weight_func
        self.rank_fit_func = rank_fit_func
        self.game_ignore_func = game_ignore_func
        self.rank_diff_params = rank_diff_params
        self.game_weight_params = game_weight_params
        self.rank_fit_params = rank_fit_params
        self.game_ignore_params = game_ignore_params

    def get_ratings(
        self, df_games: pd.DataFrame, return_games: bool = False, date: t.Optional[str] = None
    ) -> t.Union[pd.Series, t.Tuple[pd.Series, pd.DataFrame]]:
        """
        Calculate the ratings of the provided dataset.

        :param df_games: Games Table
        :param date: Date in YYYY-MM-DD format (only games up to it will be included)
        :param return_games: Whether to return additional per-game information
        :return: Series with the calculated ratings; [Games Table with additional information if return_games]
        """
        df_games = df_games.copy()
        teams = dataset.get_teams_in_games(df_games)
        components = dataset.get_graph_components(dataset.get_games_graph(df_games), teams)
        if date is not None:
            df_games = df_games.loc[df_games["Date"] <= date]
            teams = dataset.get_teams_in_games(df_games)
            components = dataset.get_graph_components(dataset.get_games_graph(df_games), teams)
        df_games["Game_Rank_Diff"] = df_games.apply(
            lambda rw: get_rank_diff(self.rank_diff_func, rw, **self.rank_diff_params), axis=1
        )
        df_games["Game_Wght"] = df_games.apply(
            lambda rw: get_game_weight(self.game_weight_func, rw, **self.game_weight_params), axis=1
        )
        ratings, df_games["Is_Ignored"], df_games["Team_Rank_Diff"] = get_rank_fit(
            self.rank_fit_func, self.game_ignore_func, teams, df_games, components, self.game_ignore_params, **self.rank_fit_params
        )
        ratings = ratings.rename("Rating_{}".format(self.name))
        if return_games:
            df_games = df_games.rename(
                columns={
                    "Game_Rank_Diff": "Game_Rank_Diff_{}".format(self.name),
                    "Team_Rank_Diff": "Team_Rank_Diff_{}".format(self.name),
                    "Game_Wght": "Game_Wght_{}".format(self.name),
                    "Is_Ignored": "Is_Ignored_{}".format(self.name),
                }
            )
            return ratings, df_games
        else:
            return ratings
