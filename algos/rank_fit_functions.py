"""
Functions used in the stage 3 of the BlockRankingAlgorithm class ranking procedure - obtaining fitted rating for each
team in the dataset. (They are not intended to be called directly)
During the fitting procedure, we call additional, game-ignore function (game_ignore_func), which can give 0 weight
to 'rank-damaging' blowout wins for the good teams (and vice versa). Additional parameters to game_ignored_func are
provided in game_ignore_kwargs dictionary.
During the ranking procedure, get_rank_fit wrapper function is called and it will call desired rank-fit function.
Other, specific rank-fit functions can potentially use the following parameters: game_ignore_func, teams, df_games,
df_components, game_ignore_kwargs and function-specific **kwargs.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import algos.game_ignore_functions as gif


# ---------------------------------
# MAIN FUNCTION
# ---------------------------------

def get_rank_fit(opt, game_ignore_func, teams, df_games, df_components,
                 game_ignore_kwargs, **kwargs):
    """
    Wrapper function to apply specific rank-fit function ('opt') on the given Games dataset.
    Option 'opt' can be either string or function reference or basically any identifier that is defined here.
    game_ignore_kwargs are specified in {GamesDataset}.game_ignore_params.
    **kwargs are specified in {GamesDataset}.rank_fit_params.
    If new rank-fit function is specified, it has to be also added here.
    In addition to ratings, it returns also is_ignored and team_rank_diff info for better output information.
    """
    df_games = df_games.copy()
    if opt in ['regression', regression_rank_fit_function]:
        ratings = regression_rank_fit_function(teams, df_games, df_components, **kwargs)
    elif opt in ['iteration', iteration_rank_fit_function]:
        ratings = iteration_rank_fit_function(teams, df_games, game_ignore_func, game_ignore_kwargs, **kwargs)
    #
    df_games['Team_1_Rank'] = ratings.reindex(df_games['Team_1']).values
    df_games['Team_2_Rank'] = ratings.reindex(df_games['Team_2']).values
    df_games['Team_Rank_Diff'] = df_games['Team_1_Rank'] - df_games['Team_2_Rank']
    is_ignored = gif.get_ignored_games(game_ignore_func, df_games, ratings, **game_ignore_kwargs)

    return ratings, is_ignored, df_games['Team_Rank_Diff']


# ---------------------------------
# INDIVIDUAL FUNCTIONS
# ---------------------------------

def regression_rank_fit_function(teams, df_games, df_components, mean_rating=0, n_round=10):
    """
    Fitting with standard (weighted) linear regression. Used in Windmill algo.
    Does not support game_ignore_func at the moment.
    """
    df_coeff = pd.Series(index=teams)
    for i_comp, df_comp in df_components.reset_index().groupby('Component'):
        teams_comp = df_comp['Team']
        df_temp = df_games.loc[df_games['Team_1'].isin(teams_comp) | df_games['Team_2'].isin(teams_comp)].reset_index()
        df_temp['const'] = 1
        df_plus = df_temp.pivot(index='index', columns='Team_1', values='const').fillna(0)
        df_minus = df_temp.pivot(index='index', columns='Team_2', values='const').fillna(0)
        x = df_plus.subtract(df_minus, fill_value=0)[teams_comp]
        y = df_temp['Game_Rank_Diff']
        wghts = df_temp['Game_Wght']
        lr = LinearRegression(fit_intercept=False).fit(x, y, sample_weight=wghts)
        #
        df_coeff[teams_comp] = (lr.coef_ - lr.coef_.mean() + mean_rating).round(n_round)

    return df_coeff


# ----------

def iteration_rank_fit_function(teams, df_games, game_ignore_func, game_ignore_kwargs,
                                n_iter=1000, dist_tol=1e-8, rating_start=1000, verbose=True, n_round=10):
    """
    Fitting with iteration procedure. In every iteration, we do the following:
        * get ignored games based on ratings_start
        * get ratings_new as the weighted average of the ratings for each game
        * set ratings to ratings_start + 0.5*(ratings_new - ratings_start) - better convergence than using ratings_new
    Repeat until desired convergence is reached.
    """
    df_ratings_iter = pd.DataFrame(rating_start, index=teams, columns=range(n_iter + 1))
    for i in range(n_iter):
        ratings_iter = df_ratings_iter[i]
        df_games_iter = df_games.copy()
        df_games_iter['Team_1_Rank'] = ratings_iter.reindex(df_games_iter['Team_1']).values
        df_games_iter['Team_2_Rank'] = ratings_iter.reindex(df_games_iter['Team_2']).values
        df_games_iter['Team_Rank_Diff'] = df_games_iter['Team_1_Rank'] - df_games_iter['Team_2_Rank']
        df_games_iter['Is_Ignored'] = gif.get_ignored_games(game_ignore_func, df_games_iter, ratings_iter,
                                                            **game_ignore_kwargs)
        df_games_iter.loc[df_games_iter['Is_Ignored'] == 1, 'Game_Wght'] = 0
        df_games_iter['Rank_1'] = df_games_iter['Game_Rank_Diff'] + df_games_iter['Team_2_Rank']
        df_games_iter['Rank_2'] = df_games_iter['Team_1_Rank'] - df_games_iter['Game_Rank_Diff']
        # duplicate the games, such that we can make the weighted average of game ranks with one groupby call
        df_games_extended = pd.concat([df_games_iter.rename(columns={'Team_1': 'Team', 'Rank_1': 'Rank'}),
                                       df_games_iter.rename(columns={'Team_2': 'Team', 'Rank_2': 'Rank'})])
        ratings_new = df_games_extended.groupby('Team').apply(lambda x: safe_wma(x['Rank'], x['Game_Wght']))
        df_ratings_iter[i + 1] = (ratings_iter + 0.5*(ratings_new - ratings_iter)).round(n_round)
        dist_iter = np.sqrt(((df_ratings_iter[i + 1] - df_ratings_iter[i]) ** 2).mean())
        if verbose:
            print('{}/{}, Change: {:.7f}, Rating 1: {:.4f}'.format(
                i + 1, n_iter, dist_iter, df_ratings_iter[i + 1].max()))
        if dist_iter < dist_tol:
            n_iter = i + 1
            break
    df_ratings_iter = df_ratings_iter[range(n_iter + 1)]

    return df_ratings_iter[n_iter]


# ------------------------------
# HELPFUNCTIONS
# ------------------------------

def safe_wma(vals, wghts):
    """
    Weighted moving average, which does not throw error when applied on the empty list and does not take Nans
    into account.
    """
    if vals.shape[0] == 0 or wghts.sum() == 0:
        return np.nan
    else:
        return np.average(vals[~np.isnan(vals)], weights=wghts[~np.isnan(vals)])
