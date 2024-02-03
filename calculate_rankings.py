import argparse
import logging
import pickle
import os
from pathlib import Path

from classes.games_dataset import GamesDataset
from definitions import DIVISIONS, ALGORITHMS
from utils.dataset import get_ranking_metrics
from utils.logging import setup_logger


def main():

    parser = argparse.ArgumentParser(description="Parser for EUF ranking calculation.")
    parser.add_argument(
        "--input", required=True, type=Path, help="Folder containing the CSV with Games Table (from prepare_data)"
    )
    parser.add_argument("--season", required=True, type=int, help="Current year (for naming purposes)")
    parser.add_argument(
        "--division", default="all", choices=["women", "mixed", "open", "all"], help="Division (women/mixed/open/all)"
    )
    parser.add_argument("--date", required=True, help="Date of calculation")
    parser.add_argument("--output", required=True, type=Path, help="Path to the folder to save the output CSVs")
    args = parser.parse_args()

    date_str = args.date.replace("-", "")

    if args.division == "all":
        divisions = DIVISIONS  # Calculate rankings for all divisions at once
    else:
        divisions = [args.division]

    calculate_rankings(args.input, args.season, divisions, date_str, args.output)


def calculate_rankings(input_path, season, divisions, date_str, output_path):
    """
    Calculate the rankings for the given division and algorithm.

    Prerequisites:
    * Run prepare_data.py script (use its output path as input to this script)

    Arguments:
    * --input - path to the folder with all necessary files
    * --division - women/mixed/open/all
    * --season - current year
    * --date - date of calculation
    * --output - path to save the output files

    Outputs:
    * CSV with Games, Summary (including ratings)
    """



    os.makedirs(output_path, exist_ok=True)
    setup_logger(output_path / f"calculate_rankings-{season}-{'_'.join(divisions)}-{date_str}.log")
    logger = logging.getLogger("ranking.data_preparation")

    for division in divisions:
        dataset = GamesDataset(
            input_path / f"{season}-{division}-euf-games.csv",
            name=f"{season}-{division}-euf",
            date=date_str,
        )

        for algo in ALGORITHMS:
            logger.info(f"Applying {algo.name} algorithm on the {dataset.name} dataset.")
            dataset.add_ratings(algo, block_algo=True)

            rmse, max_sum_resid = get_ranking_metrics(dataset.games, algo.name)
            logger.info(f"RMSE: {rmse:.2f}, Max Sum Resid: {max_sum_resid:.2f}")


        dataset.games.to_csv(output_path / f"{dataset.name}-games-{date_str}.csv", index=False, float_format="%.3f")
        dataset.summary.to_csv(output_path / f"{dataset.name}-summary-{date_str}.csv", float_format="%.3f")
        logger.info(f"Output files saved to {output_path}.")


if __name__ == "__main__":
    main()
