"""
Compare Standard and Block version of USAU Algorithm and compare it to ratings from frisbee-rankings.com.
--> The two versions are the same and converged perfectly, there is slight difference to Cody"s ratings
    (possible different score-weight function and bad convergence for small components, mainly in Mixed division).
"""
import datetime

import numpy as np
import pandas as pd

from algos.game_ignore_functions import blowout_game_ignore_function
from algos.game_weight_functions import usau_game_weight_function
from algos.rank_diff_functions import usau_rank_diff_function
from algos.usau_algo import run_ranking_iteration, get_usau_ratings
from classes.block_ranking_algorithm import BlockRankingAlgorithm
from classes.games_dataset import GamesDataset
from classes.ranking_algorithm import RankingAlgorithm
from utils.dataset import get_ranking_metrics
from utils.logging import setup_logger


def main():
    setup_logger()
    for division in ["Men", "Women", "Mixed"]:
        # Prepare dataset
        dataset_name = "USAU_2021_{}".format(division)
        dataset_path = "data/games_usau_cody_2021_{}.csv".format(division.lower())
        usau_dataset = GamesDataset(dataset_path, dataset_name)

        # Prepare game info (such that we can run iteration from the ranking process manually)
        rating_start, w0, w_first, w_last = 1000, 0.5, 29, 42
        df_games = usau_dataset.games.copy()
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

        # Apply USAU Block Algo
        usau_block_algo = BlockRankingAlgorithm(
            algo_name="USAU_Block_Algo",
            rank_diff_func="usau",
            game_weight_func="usau",
            rank_fit_func="iteration",
            game_ignore_func="blowout",
            game_weight_params={"w0": 0.5, "w_first": 29, "w_last": 42},
            rank_fit_params={"rating_start": 1000, "n_round": 2, "n_iter": 1000},
        )
        ratings_usau_block, df_games_block = usau_block_algo.get_ratings(usau_dataset.games, return_games=True)

        # Check if max avg resid is ~0
        max_sum_resid = get_ranking_metrics(df_games_block, algo_name="USAU_Block_Algo")[1]
        assert max_sum_resid < 0.5

        # Apply USAU Algo
        usau_algo = RankingAlgorithm(get_usau_ratings, "USAU_Algo", rating_start=1000, w0=0.5, w_first=29, w_last=42)
        ratings_usau = usau_algo.get_ratings(usau_dataset.games)
        df_games["Team_Rank_Diff"] = ratings_usau.reindex(df_games["Team_1"]).values - ratings_usau.reindex(df_games["Team_2"]).values
        df_games["Is_Ignored"] = blowout_game_ignore_function(df_games, ratings_usau)

        # Read Cody's Ratings
        ratings_cody = pd.read_csv(
            "data/rankings_usau_cody_2021_{}.csv".format(division.lower())
        )[["Team", "Rating"]].set_index("Team").squeeze()

        # Get New (Next-Iteration) Ratings (based on USAU Algo)
        ratings_usau_new = run_ranking_iteration(ratings_usau, df_games).sort_values(ascending=False)
        ratings_usau_block_new = run_ranking_iteration(ratings_usau_block, df_games).sort_values(ascending=False)
        ratings_cody_new = run_ranking_iteration(ratings_cody, df_games).sort_values(ascending=False)

        # Check Convergence Quality (RMSE)
        rmse_usau = np.sqrt(((ratings_usau - ratings_usau_new)**2).mean())
        rmse_usau_block = np.sqrt(((ratings_usau_block - ratings_usau_block_new)**2).mean())
        rmse_cody = np.sqrt(((ratings_cody - ratings_cody_new)**2).mean())
        rmse_diff_usau_cody = np.sqrt(((ratings_usau - ratings_cody)**2).mean())
        rmse_diff_usau_block = np.sqrt(((ratings_usau - ratings_usau_block)**2).mean())

        # Compare Results
        df_comp = pd.concat(
            [
                ratings_usau.rename("Rating_USAU"),
                ratings_usau_block.rename("Rating_USAU_Block"),
                ratings_cody.rename("Rating_Cody"),
                ratings_usau_new.rename("Rating_USAU_New"),
                ratings_usau_block_new.rename("Rating_USAU_Block_New"),
                ratings_cody_new.rename("Rating_Cody_New")
            ],
            axis=1,
        )
        df_comp["Diff_USAU_Cody"] = ratings_usau - ratings_cody
        df_comp["Diff_USAU_Block"] = ratings_usau - ratings_usau_block

        assert rmse_usau == 0
        assert rmse_usau_block == 0
        assert rmse_diff_usau_block == 0


if __name__ == "__main__":
    main()

