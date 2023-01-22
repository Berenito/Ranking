import argparse
from pathlib import Path

import datapane as dp
import pandas as pd

from classes.games_dataset import GamesDataset
from utils.dataset import duplicate_games


def main():
    """
    Export the Ranking data to the datapane report.

    Prerequisites:
    * TBA

    Arguments:
    * --input - path to the folder with all necessary files
    * --division - women/mixed/open
    * --season - current year
    * --token - datapane token for logging in
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
        dp.Page(title="Summary", blocks=[get_summary_page(dataset)]),
        dp.Page(title="Games per Team", blocks=[get_games_per_team_page(dataset)]),
        dp.Page(title="Tournaments", blocks=[get_tournaments_page(dataset)]),
        dp.Page(title="Games per Tournament", blocks=[get_games_per_tournament_page(dataset)]),
        dp.Page(title="Calendar", blocks=[dp.Group(get_calendar_page(dataset), label="Calendar")]),
    )
    app.upload(name=f"EUF {args.season} {args.division}", description="Testing EUF Ranking", open=True)


def get_summary_page(dataset: GamesDataset):
    """
    Create a page that shows the summary for all the teams.

    :param dataset: Games Dataset
    :return: Datapane page
    """
    summary_show = dataset.summary.copy()
    summary_show["Record"] = summary_show["Wins"].astype(str) + "-" + summary_show["Losses"].astype(str)
    summary_show["Score"] = summary_show["Goals_For"].astype(str) + "-" + summary_show["Goals_Against"].astype(str)
    summary_show = summary_show[["Tournaments", "Record", "W_Ratio", "Opponent_W_Ratio", "Score"]].rename(
        columns={"W_Ratio": "Win Ratio", "Opponent_W_Ratio": "Opponent Win Ratio"}
    )
    summary_show.index.name = "Team"
    summary_styled = summary_show.style.applymap_index(
        lambda v: "color:silver;" if "@" in v else "color:black;", axis=0
    ).apply(
        lambda v: (["color:silver;"] if "@" in v.name else ["color:black;"]) * summary_show.shape[1], axis=1
    ).format(
        {"Win Ratio": "{:.3f}", "Opponent Win Ratio": "{:.3f}"}
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
    """
    Create a page that shows the games for the selected team.

    :param dataset: Games Dataset
    :return: Datapane page
    """
    page = dp.Group(
        dp.Select(
            blocks=[
                dp.Group(*get_games_per_team_info(dataset, team), label=team) for team in dataset.teams if "@" not in team
            ],
            type=dp.SelectType.DROPDOWN,
        ),
    )
    return page


def get_games_per_team_info(dataset: GamesDataset, team: str):
    """
    Helpfunction to get info for the selected team.

    :param dataset: Games Dataset
    :param team: Team name
    :return: Datapane page
    """
    games_show = dataset.filter_games(team=team)
    n_tournaments = games_show["Tournament"].nunique()
    n_games = games_show.shape[0]
    record = f"{dataset.summary.loc[team, 'Wins']}-{dataset.summary.loc[team, 'Losses']}"
    score = f"{dataset.summary.loc[team, 'Goals_For']}-{dataset.summary.loc[team, 'Goals_Against']}"
    big_numbers = dp.Group(
        dp.BigNumber(heading="Tournaments", value=n_tournaments),
        dp.BigNumber(heading="Games", value=n_games),
        dp.BigNumber(heading="Record", value=record),
        dp.BigNumber(heading="Score", value=score),
        columns=4,
    )
    games_show = duplicate_games(games_show)
    games_show = games_show.loc[games_show["Team_1"] == team]
    games_show = games_show.rename(columns={"Team_2": "Opponent"})

    def get_result_abbr(x, y):
        return "W" if x > y else ("L " if x < y else "D ")

    games_show["Result"] = games_show[["Score_1", "Score_2"]].apply(
        lambda x: f"{get_result_abbr(x['Score_1'], x['Score_2'])} {x['Score_1']}-{x['Score_2']}", axis=1
    )
    games_show = games_show[["Opponent", "Date", "Tournament", "Result"]].sort_values(
        by=["Date", "Opponent"], ascending=[False, True]
    ).reset_index(drop=True)
    return f"### Games for {team}", big_numbers, dp.Table(style_games_for_team(games_show))


def style_games_for_team(games_show: pd.DataFrame):
    """
    Style the games per team table.

    :param games_show: Dataframe of the games for the selected team
    :return: Styled dataframe
    """
    games_show.index += 1
    games_styled = games_show.style.apply(
        lambda v: (
                      ["background-color:honeydew;"] if "W" in v["Result"]
                      else (["background-color:seashell;"] if "L" in v["Result"] else ["background-color:white;"])
                  ) * games_show.shape[1],
        axis=1,
    )
    return games_styled


def get_tournaments_page(dataset: GamesDataset):
    """
    Create a page that shows the summary of the played tournaments.

    :param dataset: Games Dataset
    :return: Datapane page
    """
    tournaments_show = dataset.tournaments.copy()
    tournaments_show = tournaments_show[["Date_First", "Date_Last", "N_Teams_EUF", "N_Teams_All", "N_Games"]].rename(
        columns={"Date_First": "Date First", "Date_Last": "Date Last", "N_Teams_EUF": "Teams EUF", "N_Teams_All": "Teams All", "N_Games": "Games"}
    )
    page = dp.Group(dp.Table(tournaments_show))
    return page


def get_games_per_tournament_page(dataset: GamesDataset):
    """
    Create a page that shows the games for the selected tournament.

    :param dataset: Games Dataset
    :return: Datapane page
    """
    page = dp.Group(
        dp.Select(
            blocks=[
                dp.Group(*get_games_per_tournament_info(dataset, tournament), label=tournament)
                for tournament in dataset.tournaments.index
            ],
            type=dp.SelectType.DROPDOWN,
        ),
    )
    return page


def get_games_per_tournament_info(dataset, tournament):
    """
    Helpfunction to get info for the selected tournament.

    :param dataset: Games Dataset
    :param tournament: Tournament name
    :return: Datapane page
    """
    games_show = dataset.filter_games(tournament=tournament)
    tournament_info = dataset.tournaments.loc[tournament]
    games_show["Result"] = games_show["Score_1"].astype(str) + "-" + games_show["Score_2"].astype(str)
    games_show = games_show[["Date", "Team_1", "Team_2", "Result"]].rename(
        columns={"Team_1": "Team 1", "Team_2": "Team 2"}
    )
    games_show.index += 1
    big_numbers = dp.Group(
        dp.BigNumber(heading="Date First", value=tournament_info["Date_First"]),
        dp.BigNumber(heading="Date Last", value=tournament_info["Date_Last"]),
        dp.BigNumber(heading="EUF Teams", value=tournament_info["N_Teams_EUF"]),
        dp.BigNumber(heading="All Teams", value=tournament_info["N_Teams_All"]),
        dp.BigNumber(heading="Games", value=tournament_info["N_Games"]),
        columns=5,
    )
    return f"### Games from {tournament}", big_numbers, dp.Table(games_show)


def get_calendar_page(dataset: GamesDataset):
    """
    Create a page that shows the summary per calendar weeks.

    :param dataset: Games Dataset
    :return: Datapane page
    """
    calendar_show = dataset.calendar.copy().reset_index().astype(str)
    calendar_show["Calendar Week"] = calendar_show["Year"] + "/" + calendar_show["Calendar_Week"]
    calendar_show["Tournaments (Cum)"] = calendar_show["N_Tournaments"] + " (" + calendar_show["N_Tournaments_Cum"] + ")"
    calendar_show["Teams EUF (Cum)"] = calendar_show["N_Teams_EUF"] + " (" + calendar_show["N_Teams_EUF_Cum"] + ")"
    calendar_show["Teams All (Cum)"] = calendar_show["N_Teams_All"] + " (" + calendar_show["N_Teams_All_Cum"] + ")"
    calendar_show["Games (Cum)"] = calendar_show["N_Games"] + " (" + calendar_show["N_Games_Cum"] + ")"
    calendar_show = calendar_show[
        ["Calendar Week", "Date_Start", "Date_End", "Tournaments (Cum)", "Teams EUF (Cum)", "Teams All (Cum)", "Games (Cum)"]
    ].rename(columns={"Date_Start": "Date Start", "Date_End": "Date End"})
    calendar_styled = calendar_show.style.hide(axis="index")
    page = dp.Group(dp.Table(calendar_styled))
    return page


if __name__ == "__main__":
    main()
