import logging
import argparse
import pandas as pd
from pathlib import Path

from classes.games_dataset import GamesDataset
from classes.block_ranking_algorithm import BlockRankingAlgorithm
from definitions import DIVISIONS, ALGORITHMS, DIVISION_ALIASES
from utils.dataset import get_ranking_metrics, process_games
from utils.logging import setup_logger


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


def prepare_data(input_path: Path, season: int, divisions: [str], output_path: Path):
    """
    Take data from all the CSV files in the input folder and join them to create a big Game Table with the clean data;
    export some preliminary summary statistics (no rankings are calculated here).

    Prerequisites:
    * Prepare a folder with the tournament result data - CSV files with columns Tournament, Date, Team_1, Team_2,
      Score_1, Score_2, and Division
    * Add to the same folder a TXT file per division specifying the teams in the EUF system; multiple aliases can be
      defined for each team in the same row, separated with commas (filename should be teams-<division>.txt)
    * Add to the same folder a TXT file with the pairs <team>, <tournament>; specifying that the given team has met the
      EUF roster requirements for the particular tournament (filename should be teams_at_tournaments-<division>.txt)

    Procedure:
    * Read the tournament results CSVs and take only the games for the given division
    * Read the list of EUF teams; replace aliases where applicable
    * Read teams at tournaments list; add suffix to all teams without EUF-sanctioned roster for the given tournament
    * Process the Game Table (check for invalid rows)
    * Calculate basic statistics for the season (without any rankings)
    * Save the output CSVs

    :input_path: path to the folder with all necessary files
    :season: current year
    :divisions: list of divisions for which the ranking should be calculated
    :output_path: path to the folder in which to save the output CSVs
    """
    output_path.mkdir(exist_ok=True, parents=True)
    setup_logger(output_path / f"prepare_data-{season}-{'_'.join(divisions)}.log")
    logger = logging.getLogger("ranking.data_preparation")

    for division in divisions:
        logger.info(f"Preparing data for season {season}, {division} division.")
        df_teams = pd.read_csv(input_path / f"teams-{division}.csv")
        teams_euf = df_teams["Team"].tolist()
        logger.info(f"{len(teams_euf)} teams defined for the {division} division.")

        df_games = build_games_dataframe(input_path, division)
        logger.info(f"{df_games.shape[0]} raw games found for the {division} division.")

        df_teams_at_tournaments = pd.read_csv(input_path / f"teams_at_tournaments-{division}.csv")

        replace_aliases(df_teams, df_games, df_teams_at_tournaments)

        # Check if all tournament names in teams_at_tournaments file are correct
        is_valid_tournament = df_teams_at_tournaments["Tournament"].isin(df_games["Tournament"])
        invalid_tournaments = df_teams_at_tournaments.loc[~is_valid_tournament, "Tournament"].unique()
        if len(invalid_tournaments) > 0:
            logger.warning(f"Incorrect tournament names detected in teams_at_tournaments: {invalid_tournaments}")

        # Check if all team names in teams_at_tournaments file are correct
        is_valid_team = df_teams_at_tournaments["Team"].isin(teams_euf)
        invalid_teams = df_teams_at_tournaments.loc[~is_valid_team, "Team"].unique()
        if len(invalid_teams) > 0:
            logger.warning(f"Incorrect team names detected in teams_at_tournaments: {invalid_teams}")

        # Add suffix `@ <tournament>` to all teams without valid EUF roster for the given tournament
        df_teams_at_tournaments = df_teams_at_tournaments.pivot_table(
            index="Team", columns="Tournament", aggfunc="size", fill_value=0
        )
        for team_lbl in ["Team_1", "Team_2"]:
            df_games[team_lbl] = df_games.apply(
                lambda x: add_suffix_if_not_euf_team_with_roster(df_teams_at_tournaments, x[team_lbl], x["Tournament"]),
                axis=1,
            )

        df_games = process_games(df_games)
        logger.info(f"{df_games.shape[0]} valid games found for the {division} division.")

        df_games_euf = df_games.loc[~df_games["Team_1"].str.contains(" @ ") & ~df_games["Team_2"].str.contains(" @ ")]
        logger.info(f"{df_games_euf.shape[0]} valid games found for the {division} division between EUF teams.")

        dataset_all = GamesDataset(df_games, f"{season}-{division}-all")
        dataset_euf = GamesDataset(df_games_euf, f"{season}-{division}-euf")

        dataset_euf.games.to_csv(output_path / f"{dataset_euf.name}-games.csv", index=False)
        dataset_euf.tournaments.to_csv(output_path / f"{dataset_euf.name}-tournaments.csv")
        dataset_euf.calendar.to_csv(output_path / f"{dataset_euf.name}-calendar.csv")
        dataset_euf.summary.to_csv(output_path / f"{dataset_euf.name}-summary.csv", float_format="%.3f")
        logger.info(f"CSV files saved to {output_path}.")

        # Print the list of non-EUF teams to check
        logger.info(
            "Non-EUF teams found in the data:\n" + "\n".join([t for t in sorted(dataset_all.teams) if "@" in t])
        )


def build_games_dataframe(input_path: Path, division: str) -> pd.DataFrame:
    """
    Create and return games dataframe from the given input path for the given division

    :param input_path: Path to the folder with all games files
    :param division: string name of division for which the games dataframe should be created
    :return: DataFrame with all games and their tournament, date, teams, and scores
    """
    df_list = []
    for filename in [f for f in input_path.iterdir() if f.name.startswith("games")]:
        df_tournament = pd.read_csv(filename)
        df_tournament = df_tournament.loc[df_tournament["Division"].str.lower().isin(DIVISION_ALIASES[division])]
        df_list.append(df_tournament.drop(columns="Division"))
    return pd.concat(df_list)

    
def replace_aliases(df_teams: pd.DataFrame, df_games: pd.DataFrame, df_teams_at_tournaments: pd.DataFrame):
    """
    Replace aliases of all teams in df_games and df_teams_at_tournaments data frames with the primary team name

    :param df_teams: DataFrame with all teams, their primary names, and their aliases
    :param df_games: DataFrame with all games and their tournament, date, teams, and scores
    :param df_teams_at_tournaments: DataFrame of 0/1 with teams as indices and tournaments as columns
    """
    for team, aliases in zip(df_teams["Team"], df_teams["Aliases"]):
        if not pd.isna(aliases):
            df_games["Team_1"] = df_games["Team_1"].apply(lambda x: team if x in aliases else x)
            df_games["Team_2"] = df_games["Team_2"].apply(lambda x: team if x in aliases else x)
            df_teams_at_tournaments["Team"] = df_teams_at_tournaments["Team"].apply(
                lambda x: team if x in aliases else x
            )


def add_suffix_if_not_euf_team_with_roster(df_teams_at_tournaments: pd.DataFrame, team: str, tournament: str) -> str:
    """
    Add suffix @ <tournament> the the non-EUF teams or EUF teams which did not fulfilled roster requirements for the
    given tournament.

    :param df_teams_at_tournaments: DataFrame of 0/1 with teams as indices and tournaments as columns
    :param team: Team name
    :param tournament: Tournament name
    :return: Final team name (either with suffix or not)
    """
    if (
        (team not in df_teams_at_tournaments.index)
        or (tournament not in df_teams_at_tournaments.columns)
        or (not df_teams_at_tournaments.loc[team, tournament])
    ):
        return f"{team} @ {tournament}"
    else:
        return team


def calculate_rankings(input_path: Path, season: int, divisions: [str], date: str, output_path: Path):
    """
    Perform the calculation of the rankings for the given division(s) and input.

    Prerequisites:
    * Run prepare_data.py script (use its output path as input to this script)

    Outputs:
    * CSV with Games, Summary (including ratings)

    :input_path: path to the folder with all necessary files
    :season: current year
    :divisions: list of divisions for which the ranking should be calculated
    :date_str: date of calculation
    :output_path: path to the folder in which to save the output files
    """
    date_str = date.replace("-", "")
    output_path.mkdir(exist_ok=True, parents=True)

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
            dataset.add_ratings(algo, block_algo=isinstance(algo, BlockRankingAlgorithm))

            # rmse, max_sum_resid = get_ranking_metrics(dataset.games, algo.name)
            # logger.info(f"RMSE: {rmse:.2f}, Max Sum Resid: {max_sum_resid:.2f}")


        dataset.games.to_csv(output_path / f"{dataset.name}-games-{date_str}.csv", index=False, float_format="%.3f")
        dataset.summary.to_csv(output_path / f"{dataset.name}-summary-{date_str}.csv", float_format="%.3f")
        logger.info(f"Output files saved to {output_path}.")


if __name__ == "__main__":
    main()
