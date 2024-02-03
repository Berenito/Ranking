import argparse
from pathlib import Path

from definitions import DIVISIONS
from prepare_data import prepare_data
from calculate_rankings import calculate_rankings


def main():

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
        "-d",
        default="all",
        choices=["women", "mixed", "open", "all"],
        dest="division",
        help="Division (women/mixed/open/all)",
    )
    parser.add_argument(
        "--date", "-D", required=True, dest="date", help="Date of calculation"
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        type=Path,
        dest="output",
        help="Path to the folder to save the output CSVs",
    )
    args = parser.parse_args()

    date_str = args.date.replace("-", "")

    if args.division == "all":
        divisions = DIVISIONS  # Calculate rankings for all divisions at once
    else:
        divisions = [args.division]

    calc_input = prep_output = args.output / "data_preparation"
    prep_and_calculate(
        args.input,
        calc_input,
        args.season,
        divisions,
        date_str,
        prep_output,
        args.output,
    )


def prep_and_calculate(
    prep_input, calc_input, season, divisions, date_str, prep_output, calc_output
):

    prepare_data(prep_input, season, divisions, prep_output)
    calculate_rankings(calc_input, season, divisions, date_str, calc_output)


if __name__ == "__main__":
    main()
