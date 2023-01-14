import argparse
from pathlib import Path

import datapane as dp
import pandas as pd


def main():
    """
    """
    parser = argparse.ArgumentParser(description="Parser for exporting to Datapane.")
    parser.add_argument("--input", required=True, type=Path, help="Input folder for the export")
    parser.add_argument("--token", required=True, help="Datapane token for logging in")
    parser.add_argument("--season", required=True, type=int, help="Current year (for naming purposes)")
    parser.add_argument("--division", required=True, choices=["women", "mixed", "open"], help="Division (women/mixed/open)")
    args = parser.parse_args()

    dp.login(token=args.token)
    df_summary = pd.read_csv(args.input / f"EUF-{args.season}-{args.division.capitalize()}-summary.csv")
    df_tournaments = pd.read_csv(args.input / f"EUF-{args.season}-{args.division.capitalize()}-tournaments.csv")
    df_calendar = pd.read_csv(args.input / f"EUF-{args.season}-{args.division.capitalize()}-calendar.csv")
    app = dp.App(
        "# EUF Ranking Test",
        dp.Select(
            blocks=[
                dp.Group(get_summary_page(df_summary), label="Summary"),
                dp.Group(get_games_per_team_page(), label="Games per Team"),
                dp.Group(get_tournaments_page(df_tournaments), label="Tournaments"),
                dp.Group(get_games_per_tournament_page(), label="Games per Tournament"),
                dp.Group(get_calendar_page(df_calendar), label="Calendar"),
            ],
            type=dp.SelectType.TABS,
        )
    )
    app.upload(name="EUF New Test", description="Testing EUF Ranking", open=True)


def get_summary_page(df_summary: pd.DataFrame):
    page = dp.Group(dp.Table(df_summary))
    return page


def get_games_per_team_page():
    page = dp.Group()
    return page


def get_tournaments_page(df_tournaments: pd.DataFrame):
    page = dp.Group(dp.Table(df_tournaments))
    return page


def get_games_per_tournament_page():
    page = dp.Group()
    return page


def get_calendar_page(df_calendar: pd.DataFrame):
    page = dp.Group(dp.Table(df_calendar))
    return page


if __name__ == "__main__":
    main()
