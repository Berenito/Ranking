"""
Help-functions for dataset processing.
"""

import pandas as pd
import numpy as np
import datetime
import networkx as nx
import functools


# ----------------------------------------

def process_dataset(df_games, remove_draws=False):
    """
    Check if there are some issues with the dataset.
    Reorder such that Score_1 > Score_2, i.e. Team_1 is winner of the match.
    There were some draws in USAU datasets, so their removal is now optional.
    """
    df_games = df_games.astype({'Tournament': str, 'Date': str, 'Team_1': str,
                                'Team_2': str, 'Score_1': int, 'Score_2': int})
    df_games['Tournament'] = df_games['Tournament'].fillna('Unknown Tournament')
    # drop nans
    idx_nan = df_games.isna().any(axis=1)
    if idx_nan.any():
        print('{} invalid rows removed from the dataset!'.format(idx_nan.sum()))
        df_games = df_games.loc[~idx_nan]
    # remove draws
    if remove_draws:
        idx_draws = df_games['Score_1'] == df_games['Score_2']
        if idx_draws.any():
            print('{} draws removed from the dataset!'.format(idx_draws.sum()))
            df_games = df_games.loc[~idx_draws]
    # reorder such that Team_1 is winner
    idx_bad_w_l = df_games['Score_1'] < df_games['Score_2']
    if idx_bad_w_l.any():
        print('{} matches W,L reordered in the dataset!'.format(idx_bad_w_l.sum()))
        df_games.loc[idx_bad_w_l, ['Team_1', 'Team_2', 'Score_1', 'Score_2']] = \
            df_games.loc[idx_bad_w_l, ['Team_2', 'Team_1', 'Score_2', 'Score_1']].values
    #
    df_games = df_games.sort_values(by=['Date', 'Tournament', 'Team_1', 'Team_2']).reset_index(drop=True)

    return df_games


# ----------

def duplicate_games(df_games):
    """
    Add duplicates of the games with Team_1 <-> Team_2 and Score_1 <-> Score_2,
    i.e. each game will be twice in df_games_dupl.
    Some functions are easier to apply on the dataset in this format.
    """
    df_games_copy = df_games.rename(columns={'Team_1': 'Team_2', 'Team_2': 'Team_1',
                                             'Score_1': 'Score_2', 'Score_2': 'Score_1'})
    df_games_dupl = pd.concat([df_games, df_games_copy]).reset_index(drop=True)

    return df_games_dupl


# ----------

def get_teams_in_dataset(df_games, date=None):
    """
    Get all teams present in the games dataset.
    Optionally take into account only games up to a given date.
    """
    if date is not None:
        df_games = df_games.loc[df_games['Date'] <= date]
    df_games_dupl = duplicate_games(df_games)
    teams = pd.Series(df_games_dupl['Team_1'].unique()).rename('Team').sort_values()

    return teams


# ----------

def get_opponents_for_team(df_games, team, date=None):
    """
    Get all the opponents for the given team present in the games dataset.
    Optionally take into account only games up to a given date.
    """
    if date is not None:
        df_games = df_games.loc[df_games['Date'] <= date]
    df_games_dupl = duplicate_games(df_games)
    opponents = (df_games_dupl.loc[df_games_dupl['Team_1'] == team, 'Team_2'].unique()).rename('Team').sort_values()

    return opponents


# ----------

def get_games_for_teams(df_games, teams_list, how='any', date=None):
    """
    Get games featuring any of given teams (how='any') - default.
    Get games featuring only given teams (how='only').
    Get games featuring the common opponents of the given teams (how='common').
    Optionally take into account only games up to a given date.
    """
    if date is not None:
        df_games = df_games.loc[df_games['Date'] <= date]
    if isinstance(teams_list, str):
        teams_list = [teams_list]
    if how == 'only':
        df_for_teams = df_games.loc[df_games['Team_1'].isin(teams_list) & df_games['Team_2'].isin(teams_list)]
    elif how == 'any':
        df_for_teams = df_games.loc[df_games['Team_1'].isin(teams_list) | df_games['Team_2'].isin(teams_list)]
    elif how == 'common':
        teams_common = functools.reduce(lambda x, y: list(set(get_opponents_for_team(df_games, x)) &
                                                          set(get_opponents_for_team(df_games, y))), teams_list)
        df_for_teams = df_games.loc[df_games['Team_1'].isin(teams_common) | df_games['Team_2'].isin(teams_common)]

    return df_for_teams


# ----------

def get_summary_of_games(df_games, date=None):
    """
    Calculate summary statistics from the given games.
    Optionally take into account only games up to a given date.
    Columns to return:
        'Tournaments', 'Games', 'Wins', 'Losses', 'W_Ratio', 'Opponent_W_Ratio', 'Goals_For',
        'Goals_Against', 'Avg_Point_Diff'
    """
    if date is not None:
        df_games = df_games.loc[df_games['Date'] <= date]
    df_games_dupl = duplicate_games(df_games)
    teams = get_teams_in_dataset(df_games)
    df_summary = pd.DataFrame(index=teams)
    df_summary['Wins'] = df_games.groupby('Team_1')['Score_1'].count().reindex(teams).fillna(0).astype('int')
    df_summary['Losses'] = df_games.groupby('Team_2')['Score_2'].count().reindex(teams).fillna(0).astype('int')
    df_summary['Games'] = df_summary['Wins'] + df_summary['Losses']
    df_summary['Tournaments'] = df_games_dupl.groupby('Team_1')['Tournament'].nunique()
    df_summary['Goals_For'] = df_games_dupl.groupby('Team_1')['Score_1'].sum().reindex(teams).fillna(0)
    df_summary['Goals_Against'] = df_games_dupl.groupby('Team_1')['Score_2'].sum().reindex(teams).fillna(0)
    df_summary['W_Ratio'] = df_summary['Wins'] / df_summary['Games']
    df_games_dupl['Opponent_W_Ratio'] = df_summary['W_Ratio'].reindex(df_games_dupl['Team_2']).values
    df_summary['Opponent_W_Ratio'] = df_games_dupl.groupby('Team_1')['Opponent_W_Ratio'].mean().reindex(teams).fillna(0)
    df_summary['Avg_Point_Diff'] = (df_summary['Goals_For'] - df_summary['Goals_Against']) / df_summary['Games']
    df_summary = df_summary[['Tournaments', 'Games', 'Wins', 'Losses', 'W_Ratio', 'Opponent_W_Ratio', 'Goals_For',
                             'Goals_Against', 'Avg_Point_Diff']].sort_values(
        by='W_Ratio', ascending=False)

    return df_summary


# ----------

def get_summary_of_tournaments(df_games, date=None):
    """
    Calculate summary statistics for the tournaments occurring in the games dataset.
    Optionally take into account only games up to a given date.
    Columns to return:
        'Date_First', 'Date_Last', 'N_Teams', 'N_Games'
    """
    if date is not None:
        df_games = df_games.loc[df_games['Date'] <= date]
    df_games_dupl = duplicate_games(df_games)
    df_tournaments = df_games_dupl.groupby('Tournament').agg({'Date': ['first', 'last'],
                                                              'Team_1': ['nunique', 'count']})
    df_tournaments.columns = ['Date_First', 'Date_Last', 'N_Teams', 'N_Games']
    df_tournaments['N_Games'] /= 2
    df_tournaments = df_tournaments.reset_index().sort_values(by=['Date_First', 'Date_Last', 'Tournament']).set_index(
        'Tournament')

    return df_tournaments


# ----------

def get_calendar_summary(df_games):
    """
    Calculate the summary of each calendar week in the dataset.
    Columns to return:
        Date_Start, Date_End, Year, Calendar_Week, N_Tournaments, N_Teams, N_Games (in the given week),
        N_Tournaments_Cum, N_Teams_Cum, N_Games_Cum (cumulative)
    """
    df_games_dupl = duplicate_games(df_games)
    date_first, date_last = df_games['Date'].min(), df_games['Date'].max()
    date_range = pd.date_range(date_first, date_last, freq='W').strftime('%Y-%m-%d')
    if len(date_range) == 0:
        date_range = [(pd.to_datetime(date_last) + pd.tseries.offsets.Week(weekday=6)).strftime('%Y-%m-%d')]
    df_calendar = pd.DataFrame(columns=['Date_Start', 'Date_End', 'Year', 'Calendar_Week',
                                        'N_Tournaments', 'N_Teams', 'N_Games',
                                        'N_Tournaments_Cum', 'N_Teams_Cum', 'N_Games_Cum'])
    df_calendar['Date_End'] = date_range
    df_calendar['Date_Start'] = (pd.to_datetime(df_calendar['Date_End']) - pd.tseries.offsets.Day(6)).dt.strftime(
        '%Y-%m-%d')
    df_calendar['Year'] = df_calendar['Date_Start'].str[:4]
    df_calendar['Calendar_Week'] = df_calendar['Date_End'].apply(
        lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').isocalendar()[1])
    df_calendar['N_Tournaments'] = df_calendar.apply(lambda x: df_games.loc[df_games['Date'].between(
        x['Date_Start'], x['Date_End'])]['Tournament'].nunique(), axis=1)
    df_calendar['N_Teams'] = df_calendar.apply(lambda x: df_games_dupl.loc[df_games_dupl['Date'].between(
        x['Date_Start'], x['Date_End'])]['Team_1'].nunique(), axis=1)
    df_calendar['N_Games'] = df_calendar.apply(lambda x: df_games.loc[df_games['Date'].between(
        x['Date_Start'], x['Date_End'])].shape[0], axis=1)
    df_calendar['N_Tournaments_Cum'] = df_calendar['Date_End'].apply(
        lambda x: df_games.loc[df_games['Date'] <= x]['Tournament'].nunique())
    df_calendar['N_Teams_Cum'] = df_calendar['Date_End'].apply(
        lambda x: df_games_dupl.loc[df_games_dupl['Date'] <= x]['Team_1'].nunique())
    df_calendar['N_Games_Cum'] = df_calendar['Date_End'].apply(lambda x: df_games.loc[df_games['Date'] <= x].shape[0])
    df_calendar = df_calendar.set_index(['Year', 'Calendar_Week'])

    return df_calendar


# ----------

def get_games_graph(df_games, date=None):
    """
    Get graph representation of the dataset using networkx library.
    """
    teams = get_teams_in_dataset(df_games)
    if date is not None:
        df_games = df_games.loc[df_games['Date'] <= date]
    df_conn_all = pd.DataFrame(0, index=teams, columns=teams)
    df_conn = duplicate_games(df_games).groupby(['Team_1', 'Team_2'])['Tournament'].count().rename('N_Games')
    df_conn = df_conn.reset_index().pivot(index='Team_1', columns='Team_2', values='N_Games').fillna(0)
    df_conn[df_conn > 1] = 1
    df_conn = df_conn_all.add(df_conn, fill_value=0).astype('int')
    g = nx.from_pandas_adjacency(df_conn)

    return g


# -----------

def get_graph_components(g, teams):
    """
    Return the graph component label for each team in the dataset. Numbering is ordered based on the number
    of the teams in the component.
    """
    components_raw = list(nx.algorithms.connected_components(g))
    components_raw = sorted(components_raw, key=len, reverse=True)  # sort list of lists by length in descending order
    dict_components = dict(zip(range(1, len(components_raw) + 1), components_raw))
    df_components = pd.Series(index=teams, name='Component', dtype='int')
    for team in teams:
        df_components[team] = [k for k, v in dict_components.items() if team in v][0]

    return df_components


# ----------

def get_shortest_paths(g, teams, return_all=False):
    """
    Get the information about the shortest paths between each pair of teams in the dataset (teams that played
    a game together have distance 1, teams that share a common opponent have distance 2, etc.).
    Setting return_all as True will return also more detailed information.
    """
    df_shortest_path_raw = pd.DataFrame(dict(nx.all_pairs_shortest_path(g)))
    df_shortest_path_len = pd.concat([df_shortest_path_raw[c].apply(lambda x: len(x) if isinstance(x, list) else x)
                                      for c in df_shortest_path_raw.columns], axis=1) - 1
    df_shortest_path_len = df_shortest_path_len.reindex(teams).reindex(columns=teams)
    if not return_all:
        return df_shortest_path_len
    else:
        df_shortest_paths_by_team = pd.concat([df_shortest_path_len[c].value_counts(dropna=False)
                                               for c in df_shortest_path_len.columns], axis=1).fillna(0).astype(
            'int').T.drop(columns=0)
        df_shortest_paths_by_team.columns = [str(int(c)) if not np.isnan(c) else 'NaN'
                                             for c in df_shortest_paths_by_team.columns]

    return df_shortest_path_len, df_shortest_path_raw, df_shortest_paths_by_team

