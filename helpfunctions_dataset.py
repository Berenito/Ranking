"""
Helpfunction for dataset processing.
-----
Last Update: 2021-12-03
"""

import pandas as pd

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
        df_games = df_games.loc[~idx_bad_w_l]
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

def get_teams_in_dataset(df_games):
    """
    Get all teams present in the DataFrame with games.
    """
    df_games_dupl = duplicate_games(df_games)
    teams = pd.Series(df_games_dupl['Team_1'].unique()).rename('Team')
    
    return teams

# ----------
    
def get_games_for_teams(df_games, teams_list, how='any'):
    """
    Get games featuring any of given teams (how='any').
    Get games featuring only given teams (how='only'). 
    """
    if isinstance(teams_list, str): 
        teams_list = [teams_list]
    if how == 'only':
        df_for_teams = df_games.loc[df_games['Team_1'].isin(teams_list) & df_games['Team_2'].isin(teams_list)] 
    elif how == 'any':
        df_for_teams = df_games.loc[df_games['Team_1'].isin(teams_list) | df_games['Team_2'].isin(teams_list)] 
        
    return df_for_teams

# ----------

def get_summary_of_games(df_games, date=None):
    """
    Calculate summary statistics from the given games.
    Optionally take into account only games up to a given date.
    Columns to return:
        'Games', 'Wins', 'Losses', 'W_Ratio', 'Opponent_W_Ratio', 'Goals_For',
        'Goals_Against', 'Avg_Point_Diff'
    """
    if date is not None:
        df_games = df_games.loc[df_games['Date'] <= date]
    df_games_dupl = duplicate_games(df_games)
    teams = get_teams_in_dataset(df_games)
    df_summary = pd.DataFrame(index=teams)
    # df_summary.index.name = 'Team'
    df_summary['Wins'] = df_games.groupby('Team_1')['Score_1'].count().reindex(teams).fillna(0)
    df_summary['Losses'] = df_games.groupby('Team_2')['Score_2'].count().reindex(teams).fillna(0)
    df_summary['Games'] = df_summary['Wins'] + df_summary['Losses']
    df_summary['Goals_For'] = df_games_dupl.groupby('Team_1')['Score_1'].sum().reindex(teams).fillna(0)
    df_summary['Goals_Against'] = df_games_dupl.groupby('Team_1')['Score_2'].sum().reindex(teams).fillna(0)
    df_summary['W_Ratio'] = df_summary['Wins'] / df_summary['Games']
    df_games_dupl['Opponent_W_Ratio'] = df_summary['W_Ratio'].reindex(df_games_dupl['Team_2']).values
    df_summary['Opponent_W_Ratio'] = df_games_dupl.groupby('Team_1')['Opponent_W_Ratio'].mean().reindex(teams).fillna(0)
    df_summary['Avg_Point_Diff'] = (df_summary['Goals_For'] - df_summary['Goals_Against']) / df_summary['Games']
    df_summary = df_summary[['Games', 'Wins', 'Losses', 'W_Ratio', 'Opponent_W_Ratio', 'Goals_For', 'Goals_Against', 'Avg_Point_Diff']].sort_values(
        by='W_Ratio', ascending=False)
    
    return df_summary