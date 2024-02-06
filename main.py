import argparse
from pathlib import Path

from definitions import DIVISIONS
from prepare_data import prepare_data
from calculate_rankings import calculate_rankings


def main():
    """
    Run prepare_data, calculate_rankings, or both with given arguments.
    Descriptions of pre-requisites and/or procedures can be found in the corresponding functions

    Arguments:
    * --input - path to the folder with all necessary files
    * --season - current year
    * --division - women/mixed/open/all
    * --date - date of calculation
    * --output - path to the folder in which to save the output files
    * --methods - method(s) to be called: only data preparation, only ranking calculation, or both

    Outputs:
    * CSVs with Games, Tournaments, Calendar, and Summary (without any rankings) if prepare_data is called
    * CSV with Games, Summary (including ratings) if calculate_rankings is called
    """
    parser = argparse.ArgumentParser(
        description="Parser for EUF data prep and ranking calculation."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=Path,
        dest="input",
        help="Path to the folder with all necessary files",
    )
    parser.add_argument(
        "--season",
        "-s",
        required=True,
        type=int,
        dest="season",
        help="Current year (for naming purposes)",
    )
    parser.add_argument(
        "--division",
        "-D",
        default="all",
        choices=["women", "mixed", "open", "all"],
        dest="division",
        help="Division (women/mixed/open/all)",
    )
    parser.add_argument(
        "--date", "-d", required=True, dest="date", help="Date of calculation"
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        type=Path,
        dest="output",
        help="Path to the folder to save the output CSVs",
    )
    parser.add_argument(
        "--methods",
        "-m",
        default="both",
        choices=["prepare", "calculate", "both"],
        dest="methods",
        help="Method(s) to be run (prepare_data/calculate_rankings/both)",
    )
    args = parser.parse_args()

    if args.division == "all":
        divisions = DIVISIONS  # Calculate rankings for all divisions at once
    else:
        divisions = [args.division]

    if args.methods == "both":
        # Create additional folder for data preparation. Output of preparation, input for calculation
        calc_input_path = prep_output_path = args.output / "data_preparation"

        prepare_data(args.input, args.season, divisions, prep_output_path)
        calculate_rankings(
            calc_input_path, args.season, divisions, args.date, args.output
        )

    elif args.methods == "prepare":
        prepare_data(args.input, args.season, divisions, args.output)

    elif args.methods == "calculate":
        calculate_rankings(args.input, args.season, divisions, args.date, args.output)


if __name__ == "__main__":
    main()
