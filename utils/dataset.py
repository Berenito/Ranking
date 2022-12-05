"""
Help-functions for dataset processing.
Games Table is a DataFrame with columns Tournament, Date, Team_1, Team_2, Score_1, Score_2.
"""
import functools
import logging
import typing as t
from datetime import datetime

import networkx as nx
import numpy as np
import pandas as pd


def process_games(
    df_games: pd.DataFrame, teams: t.Optional[t.Union[list, pd.Series]] = None, remove_draws: bool = False
) -> pd.DataFrame:
    """
    Check and correct the potential issues in the Games Table.

    Remove the teams not in the provided list.
    Reorder such that Score_1 > Score_2, i.e. Team_1 is the winner of the match.
    There were some draws in USAU datasets, so their removal is now optional.
    Try to convert all dates to YYYY-MM-DD format.

    :param df_games: Raw Games Table
    :param teams: Both teams must be from this list; otherwise, do not consider the game (not applicable if equal to None)
    :param remove_draws: Whether to remove the rows with draws
    :return: Processed Games Table
    """
    logger = logging.getLogger("ranking.process_games")
    df_games = df_games.astype(
        {"Tournament": str, "Date": str, "Team_1": str, "Team_2": str, "Score_1": int, "Score_2": int}
    )
    df_games["Tournament"] = df_games["Tournament"].fillna("Unknown Tournament")
    # Keep only games featuring given teams
    if teams is not None:
        df_games = get_games_for_teams(df_games, teams, how="only")
    # Drop NaNs
    idx_nan = df_games.isna().any(axis=1)
    if idx_nan.any():
        df_games = df_games.loc[~idx_nan]
        logger.info(f"{idx_nan.sum()} invalid rows removed from the dataset.")
    # Remove draws
    if remove_draws:
        idx_draws = df_games["Score_1"] == df_games["Score_2"]
        if idx_draws.any():
            logger.info(f"{idx_draws.sum()} draws removed from the dataset.")
            df_games = df_games.loc[~idx_draws]
    # Reorder such that Team_1 is the winner
    idx_bad_w_l = df_games["Score_1"] < df_games["Score_2"]
    if idx_bad_w_l.any():
        logger.info(f"{idx_bad_w_l.sum()} matches' W-L reordered in the dataset!")
        df_games.loc[idx_bad_w_l, ["Team_1", "Team_2", "Score_1", "Score_2"]] = df_games.loc[
            idx_bad_w_l, ["Team_2", "Team_1", "Score_2", "Score_1"]
        ].values
    # Find dates in DD.MM.YYYY format and convert them to YYYY-MM-DD
    dates_bad_format = df_games["Date"].str[2] == "."
    df_games.loc[dates_bad_format, "Date"] = df_games.loc[dates_bad_format, "Date"].apply(
        lambda x: datetime.strptime(x, "%d.%m.%Y").strftime("%Y-%m-%d")
    )
    return df_games.sort_values(by=["Date", "Tournament", "Team_1", "Team_2"]).reset_index(drop=True)


def duplicate_games(df_games: pd.DataFrame) -> pd.DataFrame:
    """
    Add duplicates of the games with Team_1 <-> Team_2 and Score_1 <-> Score_2; i.e., each game will be twice
    in the returned Games Table (some functions are easier to apply on the Games Table in this format).
    
    :param df_games: Games Table
    :return: Duplicated Games Table
    """
    df_games_reversed = df_games.rename(
        columns={"Team_1": "Team_2", "Team_2": "Team_1", "Score_1": "Score_2", "Score_2": "Score_1"}
    )
    return pd.concat([df_games, df_games_reversed]).reset_index(drop=True)


def get_teams_in_games(df_games: pd.DataFrame, date: t.Optional[str] = None) -> pd.Series:
    """
    Get all teams present in the Games Table (optionally take into account only games up to a given date).

    :param df_games: Games Table
    :param date: Date in YYYY-MM-DD format
    :return: Series of the teams
    """
    if date is not None:
        df_games = df_games.loc[df_games["Date"] <= date]
    df_games_dupl = duplicate_games(df_games)
    return pd.Series(df_games_dupl["Team_1"].unique()).rename("Team").sort_values()


def get_opponents_for_team(df_games: pd.DataFrame, team: str, date: t.Optional[str] = None) -> pd.Series:
    """
    Get all the opponents for the given team present in the Games Table (optionally take into account only games
    up to a given date).

    :param df_games: Games Table
    :param team: Team name
    :param date: Date in YYYY-MM-DD format
    :return: Series of the opponents for given team
    """
    if date is not None:
        df_games = df_games.loc[df_games["Date"] <= date]
    df_games_dupl = duplicate_games(df_games)

    return (df_games_dupl.loc[df_games_dupl["Team_1"] == team, "Team_2"].unique()).rename("Team").sort_values()


def get_games_for_teams(
    df_games: pd.DataFrame, teams: t.Union[str, list, pd.Series], how: str = "any", date: t.Optional[str] = None
) -> pd.DataFrame:
    """
    Filter the Games Table by the teams played (optionally take into account only games up to a given date).

    :param df_games: Games Table
    :param teams: Name/s of the teams
    :param how: Filter option: "any" of the teams (default), "only" given teams, teams' "common" opponents of the teams
    :param date: Date in YYYY-MM-DD format
    :return: Filtered part of the Games Table
    """
    if date is not None:
        df_games = df_games.loc[df_games["Date"] <= date]
    if isinstance(teams, str):
        teams = [teams]
    if how == "only":
        df_for_teams = df_games.loc[df_games["Team_1"].isin(teams) & df_games["Team_2"].isin(teams)]
    elif how == "any":
        df_for_teams = df_games.loc[df_games["Team_1"].isin(teams) | df_games["Team_2"].isin(teams)]
    elif how == "common":
        teams_common = functools.reduce(
            lambda x, y: list(set(get_opponents_for_team(df_games, x)) & set(get_opponents_for_team(df_games, y))),
            teams,
        )
        df_for_teams = df_games.loc[df_games["Team_1"].isin(teams_common) | df_games["Team_2"].isin(teams_common)]
    return df_for_teams


def get_summary_of_games(df_games: pd.DataFrame, date: t.Optional[str] = None) -> pd.DataFrame:
    """
    Calculate summary statistics from the Games Table (optionally take into account only games up to a given date).

    :param df_games: Games Table
    :param date: Date in YYYY-MM-DD format
    :return: Summary DataFrame with columns Tournaments, Games, Wins, Losses, W_Ratio, Opponent_W_Ratio, Goals_For,
             Goals_Against, Avg_Point_Diff
    """
    if date is not None:
        df_games = df_games.loc[df_games["Date"] <= date]
    df_games_dupl = duplicate_games(df_games)
    teams = get_teams_in_games(df_games)
    df_summary = pd.DataFrame(index=teams)
    df_summary["Wins"] = df_games.groupby("Team_1")["Score_1"].count().reindex(teams).fillna(0).astype("int")
    df_summary["Losses"] = df_games.groupby("Team_2")["Score_2"].count().reindex(teams).fillna(0).astype("int")
    df_summary["Games"] = df_summary["Wins"] + df_summary["Losses"]
    df_summary["Tournaments"] = df_games_dupl.groupby("Team_1")["Tournament"].nunique()
    df_summary["Goals_For"] = df_games_dupl.groupby("Team_1")["Score_1"].sum().reindex(teams).fillna(0)
    df_summary["Goals_Against"] = df_games_dupl.groupby("Team_1")["Score_2"].sum().reindex(teams).fillna(0)
    df_summary["W_Ratio"] = df_summary["Wins"] / df_summary["Games"]
    df_games_dupl["Opponent_W_Ratio"] = df_summary["W_Ratio"].reindex(df_games_dupl["Team_2"]).values
    df_summary["Opponent_W_Ratio"] = df_games_dupl.groupby("Team_1")["Opponent_W_Ratio"].mean().reindex(teams).fillna(0)
    df_summary["Avg_Point_Diff"] = (df_summary["Goals_For"] - df_summary["Goals_Against"]) / df_summary["Games"]
    df_summary = df_summary[
        ["Tournaments", "Games", "Wins", "Losses", "W_Ratio", "Opponent_W_Ratio", "Goals_For", "Goals_Against", "Avg_Point_Diff"]
    ].sort_values(by="W_Ratio", ascending=False)
    return df_summary


def get_summary_of_tournaments(df_games: pd.DataFrame, date: t.Optional[str] = None) -> pd.DataFrame:
    """
    Calculate summary statistics for the tournaments occurring in the Games Table (optionally take into account
    only games up to a given date).

    :param df_games: Games Table
    :param date: Date in YYYY-MM-DD format
    :return: Summary DataFrame of the tournaments with columns Date_First, Date_Last, N_Teams, N_Games
    """
    if date is not None:
        df_games = df_games.loc[df_games["Date"] <= date]
    df_games_dupl = duplicate_games(df_games)
    df_tournaments = df_games_dupl.groupby("Tournament").agg({"Date": ["first", "last"], "Team_1": ["nunique", "count"]})
    df_tournaments.columns = ["Date_First", "Date_Last", "N_Teams", "N_Games"]
    df_tournaments["N_Games"] /= 2
    df_tournaments = df_tournaments.reset_index().sort_values(by=["Date_First", "Date_Last", "Tournament"]).set_index("Tournament")
    return df_tournaments


def get_calendar_summary(df_games: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the summary of each calendar week in the Games Table.

    :param df_games: Games Table
    :return: Weekly summary DataFrame with columns Date_Start, Date_End, Year, Calendar_Week, N_Tournaments, N_Teams,
             N_Games (in the given week), N_Tournaments_Cum, N_Teams_Cum, N_Games_Cum (cumulative)
    """
    df_games_dupl = duplicate_games(df_games)
    date_first, date_last = df_games["Date"].min(), df_games["Date"].max()
    date_range = pd.date_range(date_first, date_last, freq="W").strftime("%Y-%m-%d")
    if len(date_range) == 0:
        date_range = [(pd.to_datetime(date_last) + pd.tseries.offsets.Week(weekday=6)).strftime("%Y-%m-%d")]
    df_calendar = pd.DataFrame(
        columns=["Date_Start", "Date_End", "Year", "Calendar_Week", "N_Tournaments", "N_Teams", "N_Games", "N_Tournaments_Cum", "N_Teams_Cum", "N_Games_Cum"]
    )
    df_calendar["Date_End"] = date_range
    df_calendar["Date_Start"] = (pd.to_datetime(df_calendar["Date_End"]) - pd.tseries.offsets.Day(6)).dt.strftime(
        "%Y-%m-%d")
    df_calendar["Year"] = df_calendar["Date_Start"].str[:4]
    df_calendar["Calendar_Week"] = df_calendar["Date_End"].apply(
        lambda x: datetime.strptime(x, "%Y-%m-%d").isocalendar()[1])
    df_calendar["N_Tournaments"] = df_calendar.apply(
        lambda x: df_games.loc[df_games["Date"].between(x["Date_Start"], x["Date_End"])]["Tournament"].nunique(), axis=1
    )
    df_calendar["N_Teams"] = df_calendar.apply(
        lambda x: df_games_dupl.loc[df_games_dupl["Date"].between(x["Date_Start"], x["Date_End"])]["Team_1"].nunique(), axis=1
    )
    df_calendar["N_Games"] = df_calendar.apply(
        lambda x: df_games.loc[df_games["Date"].between(x["Date_Start"], x["Date_End"])].shape[0], axis=1
    )
    df_calendar["N_Tournaments_Cum"] = df_calendar["Date_End"].apply(
        lambda x: df_games.loc[df_games["Date"] <= x]["Tournament"].nunique()
    )
    df_calendar["N_Teams_Cum"] = df_calendar["Date_End"].apply(
        lambda x: df_games_dupl.loc[df_games_dupl["Date"] <= x]["Team_1"].nunique()
    )
    df_calendar["N_Games_Cum"] = df_calendar["Date_End"].apply(lambda x: df_games.loc[df_games["Date"] <= x].shape[0])
    df_calendar = df_calendar.set_index(["Year", "Calendar_Week"])

    return df_calendar


def get_games_graph(df_games: pd.DataFrame, date: t.Optional[str] = None) -> nx.Graph:
    """
    Get graph representation of the connected teams (with played game) from the Games Table using networkx library
    (optionally take into account only games up to a given date).

    :param df_games: Games Table
    :param date: Date in YYYY-MM-DD format
    :return: Graph with connections between the teams that played together
    """
    teams = get_teams_in_games(df_games)
    if date is not None:
        df_games = df_games.loc[df_games["Date"] <= date]
    df_connected_init = pd.DataFrame(0, index=teams, columns=teams)  # To ensure that all the teams are in the DataFrame
    df_connected = duplicate_games(df_games).groupby(["Team_1", "Team_2"])["Tournament"].count().rename("N_Games")
    df_connected = df_connected.reset_index().pivot(index="Team_1", columns="Team_2", values="N_Games").fillna(0)
    df_connected = df_connected_init.add(df_connected, fill_value=0).astype("int").clip(upper=1)
    return nx.from_pandas_adjacency(df_connected)


def get_graph_components(graph_connections: nx.Graph, teams: t.Union[list, pd.Series]) -> pd.Series:
    """
    Return the graph component label for each team in the graph. Numbering is ordered based on the number
    of the teams in the component (bigger components go first).

    :param graph_connections: Graph with connections between the teams that played together
    :param teams: A list of teams we are interested in
    :return: Graph component label for each team
    """
    components_raw = list(nx.algorithms.connected_components(graph_connections))
    components_raw = sorted(components_raw, key=len, reverse=True)  # Sort list of lists by length in descending order
    dict_components = dict(zip(range(1, len(components_raw) + 1), components_raw))
    components = pd.Series(index=teams, name="Component", dtype="int")
    for team in teams:
        components[team] = [k for k, v in dict_components.items() if team in v][0]
    return components


def get_ranking_metrics(df_games: pd.DataFrame, algo_name: str = "") -> t.Tuple[float, float]:
    """
    Get quality metrics for given ratings.

    :param df_games: Table with games (Game_Rank_Diff_{algo_name} and Team_Rank_Diff_{algo_name} must be calculated)
    :param algo_name: Algorithm name, for correct column finding
    :return: RMSE; max avg of teams' residuals
    """
    df_games = df_games.copy()
    if len(algo_name) > 0:
        algo_name = f"_{algo_name}"
    df_games["Resid_1"] = df_games[f"Game_Rank_Diff{algo_name}"] - df_games[f"Team_Rank_Diff{algo_name}"]
    df_games["Resid_2"] = -df_games["Resid_1"]
    rmse = np.sqrt(safe_wma(df_games["Resid_1"]**2, df_games[f"Game_Wght{algo_name}"]))
    df_games_extended = pd.concat(
        [
            df_games.rename(columns={"Team_1": "Team", "Resid_1": "Resid"}),
            df_games.rename(columns={"Team_2": "Team", "Resid_2": "Resid"})
        ]
    )
    avg_resid_teams = df_games_extended.groupby("Team").apply(lambda x: safe_wma(x["Resid"], x[f"Game_Wght{algo_name}"]))
    max_avg_resid_team = avg_resid_teams.abs().max()
    return rmse, max_avg_resid_team


def safe_wma(vals: np.ndarray, wghts: np.ndarray) -> float:
    """
    Weighted average, which does not throw error when applied on the empty list and does not take Nans
    into account.

    :param vals: Array
    :param wghts: Weights of the array elements
    :return: Weighted average
    """
    if vals.shape[0] == 0 or wghts.sum() == 0:
        return np.nan
    else:
        return np.average(vals[~np.isnan(vals)], weights=wghts[~np.isnan(vals)])
