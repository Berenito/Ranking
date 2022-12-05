"""
Test the functionality of the Windmill Algo by comparing to 2018 results from Leaguevine (scraped from the website).
Windmill Algo is constructed in the new, block fashion using BlockRankingAlgorithm class.
--> All the rounds except Round 2 gives the same results, for Round 2 there are some 0.01 differences in the ratings,
    but their results have slightly worse RMSE, so it"s ok.
--> For sufficient rounding, the regression and iteration results are almost the same, again with some 0.01 differences
    for Round 2 (no idea why just for Round 2 though).
"""
import logging

import numpy as np
import pandas as pd

from classes.block_ranking_algorithm import BlockRankingAlgorithm
from classes.games_dataset import GamesDataset
from utils.dataset import get_ranking_metrics
from utils.logging import setup_logger

_logger = logging.getLogger("ranking.tests.windmill")


def main():
    setup_logger()
    # Download Windmill games & results
    tournament_id_dict = {"windmill-2018-mixed": "20661", "windmill-2018-women": "20662", "windmill-2018-open": "20659"}
    check_list = []
    for tournament_name, tournament_id in tournament_id_dict.items():
        _logger.info(f"Downloading data for tournament: {tournament_name}.")
        df_games_list_raw = pd.read_html(
            f"https://www.leaguevine.com/tournaments/{tournament_id}/{tournament_name}/swiss/games/",
            header=0,
            encoding="utf-8",
        )
        df_games_list = []
        for i, df in enumerate(df_games_list_raw):
            m, d = df.columns[0].split(" ")[1].split("/")
            df = df.iloc[:, 1:-1].rename(columns={"Team": "Team_1", "Team.1": "Team_2"})
            df[["Score_1", "Score_2"]] = df[["Result"]].apply(
                lambda x: [int(k) for k in x["Result"].strip().split("-")], axis=1, result_type="expand"
            )
            df["Round"] = 8 - i
            df["Tournament"] = "Windmill Windup 2018"
            df["Date"] = f"2018-{m.rjust(2, '0')}-{d.rjust(2, '0')}"
            df_games_list.append(df[["Round", "Tournament", "Date", "Team_1", "Team_2", "Score_1", "Score_2"]])
        df_games = pd.concat(df_games_list).reset_index(drop=True)

        df_summary_list_raw = pd.read_html(
            f"https://www.leaguevine.com/tournaments/{tournament_id}/{tournament_name}/swiss/standings/",
            header=0,
            encoding="utf-8",
        )
        df_summary_list = []
        for i, df in enumerate(df_summary_list_raw):
            df = df.rename(
                columns={
                    "Swiss Points": "Rating",
                    "Avg of Opponents' Swiss Points": "Avg_Opponent_Rating",
                    "Avg Goal Diff": "Avg_Goal_Diff",
                }
            )
            df[["W", "L"]] = df[["Record"]].apply(
                lambda x: [int(k) for k in x["Record"].strip().split("-")], axis=1, result_type="expand"
            )
            df["Round"] = 8 - i
            df_summary_list.append(
                df[["Round", "Team", "Rank", "Rating", "W", "L", "Avg_Goal_Diff", "Avg_Opponent_Rating"]]
            )
        df_summary = pd.concat(df_summary_list).reset_index(drop=True)

        # Check Regression
        for n_rounds in range(1, 6):
            _logger.info(f"Rounds: {n_rounds}.")
            df_games_round = df_games.loc[df_games["Round"] <= n_rounds]
            df_summary_round = df_summary.loc[df_summary["Round"] == n_rounds].set_index("Team")
            windmill_dataset = GamesDataset(df_games_round, name="Windmill_Dataset", calculate_weekly=False)
            windmill_algo = BlockRankingAlgorithm(
                algo_name="Windmill_Algo",
                rank_diff_func="score_diff",
                game_weight_func="uniform",
                rank_fit_func="regression",
                rank_fit_params={"n_round": 2}
            )
            windmill_algo_iter = BlockRankingAlgorithm(
                algo_name="Windmill_Algo_Iter",
                rank_diff_func="score_diff",
                game_weight_func="uniform",
                rank_fit_func="iteration",
                rank_fit_params={"rating_start": 0, "n_round": 6, "tol": 1e-10, "verbose": False}
            )

            windmill_dataset.add_ratings(windmill_algo, block_algo=True)
            windmill_dataset.add_ratings(windmill_algo_iter, block_algo=True)
            g = windmill_dataset.games
            s = windmill_dataset.summary
            s["Rating_Windmill_Algo_Iter"] = s["Rating_Windmill_Algo_Iter"].round(2)

            g["Resid_Orig"] = g["Game_Rank_Diff_Windmill_Algo"] - g[["Team_1", "Team_2"]].apply(
                lambda x: df_summary_round.loc[x["Team_1"], "Rating"] - df_summary_round.loc[x["Team_2"], "Rating"],
                axis=1,
            )
            g["Resid_Windmill_Algo"] = g["Game_Rank_Diff_Windmill_Algo"] - g[["Team_1", "Team_2"]].apply(
                lambda x: s.loc[x["Team_1"], "Rating_Windmill_Algo"] - s.loc[x["Team_2"], "Rating_Windmill_Algo"],
                axis=1,
            )
            g["Resid_Windmill_Algo_Iter"] = g["Game_Rank_Diff_Windmill_Algo_Iter"] - g[["Team_1", "Team_2"]].apply(
                lambda x: s.loc[x["Team_1"], "Rating_Windmill_Algo_Iter"] - s.loc[x["Team_2"], "Rating_Windmill_Algo_Iter"],
                axis=1,
            )

            rmse_orig = np.sqrt((g["Resid_Orig"]**2).mean())
            rmse_windmill_algo = np.sqrt((g["Resid_Windmill_Algo"]**2).mean())
            rmse_windmill_algo_iter = np.sqrt((g["Resid_Windmill_Algo_Iter"] ** 2).mean())
            diff_max_orig = (s["Rating_Windmill_Algo"] - df_summary_round["Rating"]).abs().max()
            diff_max_iter = (s["Rating_Windmill_Algo"] - s["Rating_Windmill_Algo_Iter"]).abs().max()
            check_list.append(pd.Series({
                "Tournament": tournament_name,
                "Round": n_rounds,
                "RMSE_Orig": rmse_orig,
                "RMSE_Windmill_Algo": rmse_windmill_algo,
                "RMSE_Windmill_Algo_Iter": rmse_windmill_algo_iter,
                "Diff_Max_Orig": diff_max_orig,
                "Diff_Max_Iter": diff_max_iter
            }))

            # Check if sum of residuals for each team is ~0
            max_avg_resid_windmill_algo = get_ranking_metrics(g, algo_name="Windmill_Algo")[1]
            max_avg_resid_windmill_algo_iter = get_ranking_metrics(g, algo_name="Windmill_Algo_Iter")[1]
            assert max_avg_resid_windmill_algo < 0.011
            assert max_avg_resid_windmill_algo_iter < 0.011
    df_check = pd.concat(check_list, axis=1).T
    assert df_check.loc[df_check["Round"] != 2, ["Diff_Max_Orig", "Diff_Max_Iter"]].max().max() == 0
    assert df_check.loc[df_check["Round"] == 2, ["Diff_Max_Orig", "Diff_Max_Iter"]].max().max() < 0.011


if __name__ == "__main__":
    main()
