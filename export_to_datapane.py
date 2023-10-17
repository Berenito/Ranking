import argparse
import pickle
import typing as t
import warnings
from pathlib import Path

import datapane as dp
import pandas as pd

from classes.games_dataset import GamesDataset
from definitions import DIVISIONS, ALGORITHMS
from utils.dataset import duplicate_games


def main():
    """
    Export the Ranking data to the datapane report.

    Prerequisites:
    * Create the datapane account and get the token for app deployment
    * Run calculate_rankings.py script (use its output path as input to this script)

    Arguments:
    * --input - path to the folder with all necessary files
    * --season - current year
    * --token - datapane token for logging in
    * --date - date of calculation
    * --division - women/mixed/open/all

    Outputs:
    * Datapane webpage with deployed application

    Note:
    An error can occur during the run of this script, but the report can be deployed correctly anyway.
    """
    warnings.filterwarnings("ignore", category=FutureWarning)

    parser = argparse.ArgumentParser(description="Parser for exporting to Datapane.")
    parser.add_argument("--input", required=True, type=Path, help="Input folder for the export")
    parser.add_argument("--token", required=True, help="Datapane token for logging in")
    parser.add_argument("--season", required=True, type=int, help="Current year (for naming purposes)")
    parser.add_argument("--date", required=True, help="Date of calculation")
    parser.add_argument(
        "--division", default="all", choices=["women", "mixed", "open", "all"], help="Division (women/mixed/open/all)"
    )
    args = parser.parse_args()

    dp.login(token=args.token)
    divisions = DIVISIONS if args.division == "all" else [args.division]
    for division in divisions:
        print(f"Division - {division}")
        try:
            app = dp.App(*get_division_page(args, division))
            app.upload(name=f"EUF {args.season} Rankings {division}", description=f"EUF {args.season} Rankings", open=True)
        except Exception:
            print("Error occurred")  # TODO: better handling


    # if args.division == "all":
    #     app = dp.App(
    #         *[dp.Page(title=division.capitalize(), blocks=get_division_page(args, division)) for division in DIVISIONS]
    #     )
    # else:
    #     app = dp.App(*get_division_page(args, args.division))
    #
    # app.upload(name=f"EUF {args.season} Rankings", description=f"EUF {args.season} Rankings", open=True)


def get_division_page(args: argparse.Namespace, division: str) -> t.Tuple[dp.Group, dp.Select]:
    """
    Read the pickle output from `calculate_rankings` and get Datapane page per one division (mixed/open/women).

    :param args: Parsed args
    :param division: Division name
    :return: Section with the basic big-number info; main select of the page (with summary, tournaments, etc.)
    """
    date_str = args.date.replace("-", "")
    with open(args.input / f"EUF-{args.season}-{division}-{date_str}.pkl", "rb") as f:
        dataset = pickle.load(f)

    algo_names = [algo.name for algo in ALGORITHMS]
    for algo_name in algo_names:
        dataset.summary[f"Rank_{algo_name}"] = "-"
        dataset.summary.loc[dataset.summary["Eligible"] == 1, f"Rank_{algo_name}"] = dataset.summary.loc[
            dataset.summary["Eligible"] == 1, f"Rating_{algo_name}"
        ].rank(ascending=False).astype(int).astype(str)

    page_main_select = dp.Select(
        blocks=[
            dp.Group(get_summary_page(dataset, algo_names), label="Summary"),
            dp.Group(get_games_per_team_page(dataset, algo_names), label="Games per Team"),
            dp.Group(get_tournaments_page(dataset), label="Tournaments"),
            dp.Group(get_games_per_tournament_page(dataset), label="Games per Tournament"),
            dp.Group(get_calendar_page(dataset), label="Calendar"),
        ],
        type=dp.SelectType.TABS,
    )
    return add_basic_info(args, division), page_main_select


def add_basic_info(args: argparse.Namespace, division: str) -> dp.Group:
    """
    Add basic info about the page - season, division and the calculation date.

    :param args: Parsed arguments from argparse
    :param division: Division (mixed/open/women)
    :return: Basic info group for Datapane page
    """
    basic_info = dp.Group(
        dp.BigNumber(heading="EUF Season", value=args.season),
        dp.BigNumber(heading="Division", value=division.capitalize()),
        dp.BigNumber(heading="Date", value=args.date),
        columns=3,
    )
    return basic_info


def get_summary_page(dataset: GamesDataset, algo_names: t.List[str]) -> dp.Select:
    """
    Create a page that shows the summary for all the teams.

    :param dataset: Games Dataset
    :param algo_names: Algorithm names
    :return: Datapane page
    """
    summary_show = dataset.summary.copy()
    summary_show["Record"] = summary_show["Wins"].astype(str) + "-" + summary_show["Losses"].astype(str)
    summary_show["Score"] = summary_show["Goals_For"].astype(str) + "-" + summary_show["Goals_Against"].astype(str)
    summary_show.index.name = "Team"
    summary_show = summary_show.reset_index()
    page = dp.Group(
        dp.Group(
            dp.BigNumber(heading="EUF Teams", value=(~dataset.teams.str.contains("@")).sum()),
            dp.BigNumber(heading="All Teams", value=dataset.n_teams),
            dp.BigNumber(heading="Tournaments", value=dataset.n_tournaments),
            dp.BigNumber(heading="Games", value=dataset.n_games),
            columns=4,
        ),
        dp.Select(
            blocks=[
                dp.Group(get_summary_for_one_algo(summary_show, algo_name), label=algo_name) for algo_name in algo_names
            ],
            type=dp.SelectType.TABS,
        )
    )
    return page


def get_summary_for_one_algo(df_summary: pd.DataFrame, algo_name: str) -> dp.Table:
    """
    Get summary table for one algorithm.

    :param df_summary: GamesDataset.summary table
    :param algo_name: Algorithm name (to select proper columns)
    :return: Datapane table
    """
    df_summary = df_summary.rename(
        columns={
            f"Rating_{algo_name}": "Rating",
            f"Rank_{algo_name}": "Rank",
            "W_Ratio": "Win Ratio",
            "Opponent_W_Ratio": "Opponent Win Ratio",
        }
    )
    df_summary = df_summary[
        ["Rank", "Team", "Rating", "Tournaments", "Record", "Win Ratio", "Opponent Win Ratio", "Score"]
    ].sort_values(by="Rating", ascending=False)
    summary_styled = df_summary.style.apply(
        lambda v: (["color:silver;"] if v["Rank"] == "-" else ["color:black;"]) * df_summary.shape[1], axis=1
    ).format(
        {"Win Ratio": "{:.3f}", "Opponent Win Ratio": "{:.3f}", "Rating": "{:.2f}"}
    ).hide(axis="index")
    return dp.Table(summary_styled)


def get_games_per_team_page(dataset: GamesDataset, algo_names: t.List[str]) -> dp.Select:
    """
    Create a page that shows the games for the selected team.

    :param dataset: Games Dataset
    :param algo_names: Algorithm names
    :return: Datapane page
    """
    page = dp.Select(
        blocks=[
            dp.Group(get_games_per_team_info(dataset, team, algo_names), label=team)
            for team in dataset.teams if "@" not in team
        ],
        type=dp.SelectType.DROPDOWN,
    )
    return page


def get_games_per_team_info(dataset: GamesDataset, team: str, algo_names: t.List[str]) -> dp.Group:
    """
    Helpfunction to get info for the selected team.

    :param dataset: Games Dataset
    :param team: Team name
    :param algo_names: Algorithm names
    :return: Datapane page per one team
    """
    games_show = dataset.filter_games(team=team)
    n_tournaments = games_show["Tournament"].nunique()
    record = f"{dataset.summary.loc[team, 'Wins']}-{dataset.summary.loc[team, 'Losses']}"
    score = f"{dataset.summary.loc[team, 'Goals_For']}-{dataset.summary.loc[team, 'Goals_Against']}"

    def get_result_abbr(x, y):
        return "W" if x > y else ("L " if x < y else "D ")

    games_show = duplicate_games(games_show)
    games_show = games_show.loc[games_show["Team_1"] == team]
    games_show = games_show.rename(columns={"Team_2": "Opponent"})
    games_show["Result"] = games_show[["Score_1", "Score_2"]].apply(
        lambda x: f"{get_result_abbr(x['Score_1'], x['Score_2'])} {x['Score_1']}-{x['Score_2']}", axis=1
    )
    for algo_name in algo_names:
        games_show[f"Game_Wght_{algo_name}"] = (
            games_show[f"Game_Wght_{algo_name}"] * (1 - games_show[f"Is_Ignored_{algo_name}"])
        )
        games_show[f"Game_Wght_{algo_name}"] = (
            100 * games_show[f"Game_Wght_{algo_name}"] / games_show[f"Game_Wght_{algo_name}"].sum()
        )
        games_show[f"Game_Diff_{algo_name}"] = (
            games_show[f"Game_Rank_Diff_{algo_name}"] - games_show[f"Team_Rank_Diff_{algo_name}"]
        )
    group = dp.Group(
        dp.Group(
            dp.BigNumber(heading="Team", value=team),
            dp.BigNumber(heading="Tournaments", value=n_tournaments),
            dp.BigNumber(heading="Record", value=record),
            dp.BigNumber(heading="Score", value=score),
            *[
                dp.BigNumber(
                    heading=f"Rating {algo_name}",
                    value=f"{dataset.summary.loc[team, f'Rating_{algo_name}']} ({dataset.summary.loc[team, f'Rank_{algo_name}']})",
                )
                for algo_name in algo_names
            ],
            columns=4,
        ),
        dp.Select(
            blocks=[
                dp.Group(get_games_per_team_for_one_algo(games_show, algo_name), label=algo_name)
                for algo_name in algo_names
            ],
            type=dp.SelectType.TABS,
        ),
    )
    return group


def get_games_per_team_for_one_algo(df_games: pd.DataFrame, algo_name: str) -> dp.Table:
    """
    Get main table in the games-per-team select for the chosen algorithm.

    :param df_games: GamesDataset.games table
    :param algo_name: Algorithm name (to select proper columns)
    :return: Datapane table
    """
    df_games = df_games.rename(
        columns={f"Game_Wght_{algo_name}": "Game Weight", f"Game_Diff_{algo_name}": "Game Difference"}
    )
    df_games = df_games[["Opponent", "Date", "Tournament", "Result", "Game Weight", "Game Difference"]].sort_values(
        by=["Date", "Opponent"], ascending=[False, True]
    ).reset_index(drop=True)
    df_games.index += 1
    games_styled = df_games.style.apply(
        lambda v: (
                      ["background-color:honeydew;"] if "W" in v["Result"]
                      else (["background-color:seashell;"] if "L" in v["Result"] else ["background-color:white;"])
                  ) * df_games.shape[1],
        axis=1,
    ).format({"Game Weight": "{:.1f}%", "Game Difference": "{:.2f}"})
    return dp.Table(games_styled)


def get_tournaments_page(dataset: GamesDataset) -> dp.Table:
    """
    Create a page that shows the summary of the played tournaments.

    :param dataset: Games Dataset
    :return: Tournaments table for Datapane page
    """
    tournaments_show = dataset.tournaments.copy()
    tournaments_show = tournaments_show[["Date_First", "Date_Last", "N_Teams_EUF", "N_Teams_All", "N_Games"]].rename(
        columns={
            "Date_First": "Date First",
            "Date_Last": "Date Last",
            "N_Teams_EUF": "Teams EUF",
            "N_Teams_All": "Teams All",
            "N_Games": "Games",
        }
    )
    return dp.Table(tournaments_show)


def get_games_per_tournament_page(dataset: GamesDataset) -> dp.Select:
    """
    Create a page that shows the games for the selected tournament.

    :param dataset: Games Dataset
    :return: Games per tournament select for Datapane page
    """
    page = dp.Select(
        blocks=[
            dp.Group(get_games_per_tournament_info(dataset, tournament), label=tournament)
            for tournament in dataset.tournaments.index
        ],
        type=dp.SelectType.DROPDOWN,
    )
    return page


def get_games_per_tournament_info(dataset: GamesDataset, tournament: str) -> dp.Group:
    """
    Helpfunction to get info for the selected tournament.

    :param dataset: Games Dataset
    :param tournament: Tournament name
    :return: Games per tournament info for Datapane page
    """
    games_show = dataset.filter_games(tournament=tournament)
    tournament_info = dataset.tournaments.loc[tournament]
    games_show["Result"] = games_show["Score_1"].astype(str) + "-" + games_show["Score_2"].astype(str)
    games_show = games_show[["Date", "Team_1", "Team_2", "Result"]].rename(
        columns={"Team_1": "Team 1", "Team_2": "Team 2"}
    )
    games_show.index += 1
    big_numbers = dp.Group(
        dp.BigNumber(heading="Tournament", value=tournament),
        dp.BigNumber(heading="Date First", value=tournament_info["Date_First"]),
        dp.BigNumber(heading="Date Last", value=tournament_info["Date_Last"]),
        dp.BigNumber(heading="EUF Teams", value=tournament_info["N_Teams_EUF"]),
        dp.BigNumber(heading="All Teams", value=tournament_info["N_Teams_All"]),
        dp.BigNumber(heading="Games", value=tournament_info["N_Games"]),
        columns=3,
    )
    return dp.Group(big_numbers, dp.Table(games_show))


def get_calendar_page(dataset: GamesDataset) -> dp.Table:
    """
    Create a page that shows the summary per calendar weeks.

    :param dataset: Games Dataset
    :return: Calendar table for Datapane page
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
    return dp.Table(calendar_styled)


if __name__ == "__main__":
    main()
