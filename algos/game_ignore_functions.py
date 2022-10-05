"""
Functions used during the rank-fitting stage of the BlockRankingAlgorithm class ranking procedure
(They are not intended to be called directly).
During the ranking procedure, get_ignored_games wrapper function is called and it will call desired game-ignore function
Other, specific game-ignore functions use input df_games, ratings and potentially function-specific **kwargs.
"""

import pandas as pd


# ---------------------------------
# MAIN FUNCTION
# ---------------------------------

def get_ignored_games(opt, df_games, ratings, **kwargs):
    """
    Wrapper function to apply specific game-ignore function ('opt') on the given Games dataset based on the current
    ratings.
    Option 'opt' can be either string or function reference or basically any identifier that is defined here.
    **kwargs are specified in {GamesDataset}.game_ignore_params.
    If new game-ignore function is specified, it has to be also added here.
    """
    df_games = df_games.copy()
    if opt in [None, 'none', no_game_ignore_function]:
        is_ignored = no_game_ignore_function(df_games)
    elif opt in ['usau', 'blowout', blowout_game_ignore_function]:
        is_ignored = blowout_game_ignore_function(df_games, ratings, **kwargs)
    else:
        raise ValueError("Unknown game-ignore option, see algos/game_ignore_functions.py for more info.")

    return is_ignored


# ---------------------------------
# INDIVIDUAL FUNCTIONS
# ---------------------------------

def no_game_ignore_function(df_games):
    """
    All games are used.
    """
    is_ignored = pd.Series(0, name='Is_Ignored', index=df_games.index)

    return is_ignored


def blowout_game_ignore_function(df_games, ratings, min_rank_diff=600, min_valid=5,
                                 blowout_fcn=lambda x: x['Score_1'] > 2 * x['Score_2'] + 1):
    """
    For a game to be ignored, the following conditions have to be satisfied:
        - it has to be a blowout (score_w > 2*score_l + 1) (blowout_fcn)
        - rating difference have to be > 600 (min_rank_diff)
        - winner must have at least 5 (min_valid) un-ignored games
    In case only some of the games can be ignored (due to 5-games rule), the most damaging ones are ignored.
    Default case is equal to the ignore function used in USAU algorithm, but now can be also generalized easily.
    """
    df_games['Is_Blowout'] = df_games.apply(blowout_fcn, axis=1)
    df_games['Is_Ignored'] = 1 * (df_games['Is_Blowout'] & ((df_games['Team_Rank_Diff'] > min_rank_diff)
                                                            | df_games['Team_Rank_Diff'].isna()))
    for team in ratings.index:
        n_ignored = df_games.loc[df_games['Team_1'] == team, 'Is_Ignored'].sum()
        n_valid = (1 - df_games.loc[(df_games['Team_1'] == team) | (df_games['Team_2'] == team), 'Is_Ignored']).sum()
        if n_valid < min_valid and n_ignored > 0:
            n_unignore = int(min(min_valid - n_valid, n_ignored))
            idx_ignored = df_games.loc[(df_games['Team_1'] == team) & (df_games['Is_Ignored'] == 1)].sort_values(
                by='Team_Rank_Diff').index.tolist()
            df_games.loc[idx_ignored[:n_unignore], 'Is_Ignored'] = 0

    return df_games['Is_Ignored']
