import argparse
from pathlib import Path

import datapane as dp
import pandas as pd

from classes.games_dataset import GamesDataset


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
    dataset = GamesDataset(args.input / f"EUF-{args.season}-{args.division.capitalize()}-games.csv")
    app = dp.App(
        "# EUF Ranking Test",
        dp.Select(
            blocks=[
                dp.Group(get_summary_page(dataset), label="Summary"),
                dp.Group(get_games_per_team_page(dataset), label="Games per Team"),
                dp.Group(get_tournaments_page(dataset), label="Tournaments"),
                dp.Group(get_games_per_tournament_page(dataset), label="Games per Tournament"),
                dp.Group(get_calendar_page(dataset), label="Calendar"),
            ],
            type=dp.SelectType.TABS,
        )
    )
    app.upload(name="EUF New Test", description="Testing EUF Ranking", open=True)


def get_summary_page(dataset: GamesDataset):
    """
    """
    summary_show = dataset.summary.copy()
    summary_show["Record"] = summary_show["Wins"].astype(str) + "-" + summary_show["Losses"].astype(str)
    summary_show["Score"] = summary_show["Goals_For"].astype(str) + "-" + summary_show["Goals_Against"].astype(str)
    summary_show = summary_show[["Tournaments", "Record", "W_Ratio", "Opponent_W_Ratio", "Score"]]
    summary_show.index.name = "Team"
    summary_styled = summary_show.style.applymap_index(
        lambda v: "color:silver;" if "@" in v else "color:black;", axis=0
    ).apply(
        lambda v: ["color:silver;"] * summary_show.shape[1] if "@" in v.name else ["color:black;"] * summary_show.shape[1], axis=1
    ).format(
        {"W_Ratio": "{:.3f}", "Opponent_W_Ratio": "{:.3f}"}
    )
    page = dp.Group(
        dp.Group(
            dp.BigNumber(heading="EUF Teams", value=(~dataset.teams.str.contains("@")).sum()),
            dp.BigNumber(heading="All Teams", value=dataset.n_teams),
            dp.BigNumber(heading="Tournaments", value=dataset.n_tournaments),
            dp.BigNumber(heading="Games", value=dataset.n_games),
            columns=4,
        ),
        dp.Table(summary_styled)
    )
    return page


def get_games_per_team_page(dataset: GamesDataset):
    page = dp.Group()
    return page


def get_tournaments_page(dataset: GamesDataset):
    page = dp.Group(dp.Table(dataset.tournaments))
    return page


def get_games_per_tournament_page(dataset: GamesDataset):
    page = dp.Group()
    return page


def get_calendar_page(dataset: GamesDataset):
    page = dp.Group(dp.Table(dataset.calendar))
    return page


if __name__ == "__main__":
    main()
