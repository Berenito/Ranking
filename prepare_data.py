"""
Take data from all the CSV files in the input folder and join them to create a big Game Table.
"""
import logging
from pathlib import Path

import pandas as pd

from classes.games_dataset import GamesDataset
from utils.dataset import process_games, get_teams_in_games
from utils.logging import setup_logger


def main():
    # TODO: use argparse
    input_path = Path("C:/Users/micha/Desktop/Python/euf_ranking/data_2019")
    division = "Open"
    season = 2019
    teams_file = Path("C:/Users/micha/Desktop/Python/euf_ranking/teams_2019/open.txt")
    output_path = Path("C:/Users/micha/Desktop/Python/euf_ranking/output_2019")

    setup_logger()
    logger = logging.getLogger("ranking.data_preparation")

    with open(teams_file, "r") as f:
        teams = f.read().split("\n")
    logging.info(f"{len(teams)} teams found in the file.")

    df_list = []
    for filename in input_path.iterdir():
        df_tournament = pd.read_csv(filename)
        df_tournament = df_tournament.loc[df_tournament["Division"] == division].drop(columns="Division")
        df_list.append(df_tournament)
    df_games_raw = pd.concat(df_list)
    logger.info(f"{df_games_raw.shape[0]} raw games found for the {division} division.")

    teams_missing = set(get_teams_in_games(df_games_raw)).difference(teams)
    logger.info(f"{len(teams_missing)} teams from the CSVs are missing in the file: {teams_missing}.")

    df_games = process_games(df_games_raw, teams=teams)

    teams_no_games = set(teams).difference(set(get_teams_in_games(df_games)))
    logger.info(f"{len(teams_no_games)} teams with no games found in the file: {teams_no_games}.")

    logger.info(f"{df_games.shape[0]} valid games found for the {division} division.")

    dataset = GamesDataset(df_games, f"EUF-{season}-{division}")
    dataset.games.to_csv(output_path / f"{dataset.name}-games.csv", index=False)
    dataset.tournaments.to_csv(output_path / f"{dataset.name}-tournaments.csv")
    dataset.calendar.to_csv(output_path / f"{dataset.name}-calendar.csv")
    dataset.summary.to_csv(output_path / f"{dataset.name}-summary.csv")
    logger.info(f"CSV files saved to {output_path}.")


if __name__ == "__main__":
    main()



