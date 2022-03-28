# -*- coding: utf-8 -*-
"""
Sigmoid algo.
"""

import helpfunctions.helpfunctions_dataset as hf_d
import numpy as np
import scipy.optimize as optimize

# -----

def get_sigmoid_ratings(df_games):
    """
    Sigmopid rating function.
    """

    teams = hf_d.get_teams_in_dataset(df_games)#.to_numpy()
    teamlist =list(teams)

    df_games = df_games.copy()

    team_to_index = {}
    index_to_team = {}
    for i in range(teams.size):
        team_to_index[teams.iloc[i]] = i
        index_to_team[i] = teams.iloc[i]
    df_games['Team_1_index'] = df_games['Team_1'].apply(lambda t: team_to_index[t])
    df_games['Team_2_index'] = df_games['Team_2'].apply(lambda t: team_to_index[t])

    def err(x):

        def sigmoid_function(x):
            return 15*np.exp(-np.logaddexp(0, -x/100))

        # Compute error
        df_games['Error'] =  df_games.apply(lambda r: (sigmoid_function(x[r['Team_1_index']] - x[r['Team_2_index']]) - (r['Score_1'] - r['Score_2']) )**2, axis=1)
        total_error = sum(df_games['Error'])
        print("  Total error: {}".format(total_error))
        return total_error

    rating = optimize.minimize(err, np.zeros(teams.size), method='Powell', tol=1)#, options={'maxiter': 3})

    #print([(i, teamlist[i]) for i in range(len(teamlist))])

    return rating
