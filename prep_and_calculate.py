import argparse
from pathlib import Path

from definitions import DIVISIONS
from prepare_data import prepare_data
from calculate_rankings import calculate_rankings


def main():
    """
    Take data from all the CSV files in the input folder and join them to create a big Game Table with the clean data;
    export some preliminary summary statistics. Then use this data to calculate the rankings.

    Prerequisites:
    * Prepare a folder with the tournament result data - CSV files with columns Tournament, Date, Team_1, Team_2,
      Score_1, Score_2, and Division
    * Add to the same folder a TXT file per division specifying the teams in the EUF system; multiple aliases can be
      defined for each team in the same row, separated with commas (filename should be teams-<division>.txt)
    * Add to the same folder a TXT file with the pairs <team>, <tournament>; specifying that the given team has met the
      EUF roster requirements for the particular tournament (filename should be teams_at_tournaments-<division>.txt)

    Arguments:
    * --input - path to the folder with all necessary files
    * --season - current year
    * --division - women/mixed/open/all
    * --date - date of calculation
    * --output - path to the folder in which to save the output files

    Outputs:
    * CSVs with Games, Tournaments, Calendar, and Summary (without any rankings)
    * CSV with Games, Summary (including ratings)
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
    args = parser.parse_args()

    date_str = args.date.replace("-", "")

    if args.division == "all":
        divisions = DIVISIONS  # Calculate rankings for all divisions at once
    else:
        divisions = [args.division]

    prep_and_calculate(
        args.input,
        args.season,
        divisions,
        date_str,
        args.output,
    )


def prep_and_calculate(input_path, season, divisions, date_str, output_path):
    """
    Take data from all the CSV files in the input folder and join them to create a big Game Table with the clean data;
    export some preliminary summary statistics. Then use this data to calculate the rankings.

    Prerequisites:
    * Prepare a folder with the tournament result data - CSV files with columns Tournament, Date, Team_1, Team_2,
      Score_1, Score_2, and Division
    * Add to the same folder a TXT file per division specifying the teams in the EUF system; multiple aliases can be
      defined for each team in the same row, separated with commas (filename should be teams-<division>.txt)
    * Add to the same folder a TXT file with the pairs <team>, <tournament>; specifying that the given team has met the
      EUF roster requirements for the particular tournament (filename should be teams_at_tournaments-<division>.txt)

    Outputs:
    * CSVs with Games, Tournaments, Calendar, and Summary (without any rankings)
    * CSV with Games, Summary (including ratings)

    :input_path: path to the folder with all necessary files
    :season: current year
    :divisions: list of divisions for which the ranking should be calculated
    :date_str: date of calculation
    :output_path: path to the folder in which to save the output files
    """

    calc_input_path = prep_output_path = output_path / "data_preparation"

    prepare_data(input_path, season, divisions, prep_output_path)
    calculate_rankings(calc_input_path, season, divisions, date_str, output_path)


if __name__ == "__main__":
    main()
