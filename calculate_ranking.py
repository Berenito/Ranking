import argparse
import logging
import pickle
from pathlib import Path

from classes.games_dataset import GamesDataset
from definitions import USAU_ALGO
from utils.dataset import get_ranking_metrics
from utils.logging import setup_logger

ALGORITHMS = {"usau": USAU_ALGO}


def main():
    """
    Calculate the rankings for the given division and algorithm.

    Prerequisites:
    * Run prepare_data script

    Arguments:
    * --input - path to the folder with all necessary files
    * --division - women/mixed/open
    * --season - current year
    * --date - date of calculation
    * --algorithm - algorithm name
    * --output - path to save the output files

    Outputs:
    * CSV with Games, Summary; pickle with GamesDataset object

    """
    parser = argparse.ArgumentParser(description="Parser for ranking calculation.")
    parser.add_argument(
        "--input", required=True, type=Path, help="Folder containing the CSV with Games Table (from prepare_data)"
    )
    parser.add_argument("--season", required=True, type=int, help="Current year (for naming purposes)")
    parser.add_argument(
        "--division", required=True, choices=["women", "mixed", "open"], help="Division (women/mixed/open)"
    )
    parser.add_argument("--date", required=True, help="Date of calculation")
    parser.add_argument("--algorithm", required=True, help="Algorithm name")
    parser.add_argument("--output", required=True, type=Path, help="Path to the folder to save the output CSVs")
    args = parser.parse_args()

    setup_logger()
    logger = logging.getLogger("ranking.data_preparation")

    algo = ALGORITHMS[args.algorithm.lower()]
    dataset = GamesDataset(
        args.input / f"EUF-{args.season}-{args.division}-games.csv",
        name=f"EUF-{args.season}-{args.division}",
    )
    logger.info(f"Applying {args.algorithm} algorithm on the {dataset.name} dataset.")
    dataset.add_ratings(algo, block_algo=True)

    rmse, max_sum_resid = get_ranking_metrics(dataset.games, algo.name)
    logger.info(f"RMSE: {rmse:.2f}, Max Sum Resid: {max_sum_resid:.2f}")

    date_str = args.date.replace("-", "")
    dataset.games.to_csv(args.output / f"{dataset.name}-games-{args.algorithm}-{date_str}.csv", index=False)
    dataset.summary.to_csv(args.output / f"{dataset.name}-summary-{args.algorithm}-{date_str}.csv")
    with open(args.output / f"{dataset.name}-{args.algorithm}-{date_str}.pkl", "wb") as f:
        pickle.dump(dataset, f)
    logger.info(f"Output files saved to {args.output}.")


if __name__ == "__main__":
    main()
