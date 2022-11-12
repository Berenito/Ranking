# -*- coding: utf-8 -*-
"""
Sigmoid algo.
"""

import helpfunctions.helpfunctions_dataset as hf_d
import numpy as np
import pandas as pd
import scipy.optimize as optimize

# -----


def sigmoid_function(x, max_score):
    return (2*max_score)*np.exp(-np.logaddexp(0, -x/100))-max_score


def obj_func(x, fcn, score_1, score_2, idx_1, idx_2, bdd_sigmoid=False, game_cap=15, game_score_bound=11, **kwargs):
    """
    Now you can just change fcn(x, **kwargs) to anything you define
    """

    if bdd_sigmoid:
        bound = min(game_cap, game_score_bound)
        score_diff = np.minimum(score_1- score_2, bound)
        resid = sigmoid_function(x[idx_1] - x[idx_2], bound) - (score_1 - score_2)
        #resid = sigmoid_function(x[idx_1] - x[idx_2], bound=bound) - min(bound, score_1 - score_2)
    else:
        resid = fcn(x[idx_1] - x[idx_2], **kwargs) - (score_1 - score_2)

    rmse = np.sqrt(np.mean(resid**2))
    print("  RMSE: {}".format(rmse))

    return rmse


def get_sigmoid_ratings(df_games, max_score=15, rating_start=0, ):
    """
    Sigmoid rating function.
    """

    # Preprocess team and game data
    teams = hf_d.get_teams_in_dataset(df_games)
    ind_teams = pd.Series(range(len(teams)), index=teams)

    score_1 = df_games["Score_1"].values
    score_2 = df_games["Score_2"].values
    team_1_index = ind_teams.reindex(df_games["Team_1"]).values
    team_2_index = ind_teams.reindex(df_games["Team_2"]).values

    # Find team ratings that minimize the total error
    fcn = lambda x: obj_func(x, sigmoid_function, score_1, score_2, team_1_index, team_2_index, max_score=max_score, bdd_sigmoid=True)

    ratings_np = optimize.minimize(fcn, np.zeros(teams.size), method='Powell', tol=1e-3).x
    ratings = pd.Series(ratings_np, index=teams).sort_values(ascending=False)

    return ratings
