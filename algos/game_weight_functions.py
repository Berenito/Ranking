"""
Functions used in the stage 2 of the BlockRankingAlgorithm class ranking procedure - obtaining game weight
for each game in the dataset. (They are not intended to be called directly)
During the ranking procedure, get_game_weight wrapper function is called and it will call desired game-weight function.
Other, specific game-weight functions should have format game_wght = game_weight_func(**kwargs).
"""

import numpy as np
import datetime


# ---------------------------------
# MAIN FUNCTION
# ---------------------------------

def get_game_weight(opt, rw, **kwargs):
    """
    Wrapper function to apply specific game-weight function ('opt') on the given row ('rw') in the Games dataset.
    Option 'opt' can be either string or function reference or basically any identifier that is defined here.
    Scores are read from rw, theoretically also some other info can be obtained from rw.
    **kwargs are specified in {GamesDataset}.game_weight_params
    If new game-weight function is specified, it has to be also added here.
    """
    if opt in ['uniform', None, uniform_game_weight_function]:
        game_wght = uniform_game_weight_function()
    elif opt in ['usau', usau_game_weight_function]:
        # Use Monday as part of the previous week to resolve the issue of tournaments with some games played on Monday
        w_num = (datetime.datetime.strptime(rw['Date'], '%Y-%m-%d') - datetime.timedelta(days=1)).isocalendar()[1]
        game_wght = usau_game_weight_function(rw['Score_1'], rw['Score_2'], w_num, **kwargs)

    return game_wght


# ---------------------------------
# INDIVIDUAL FUNCTIONS
# ---------------------------------

def uniform_game_weight_function():
    """
    All games have weight 1.
    """
    game_wght = 1

    return game_wght


# ----------

def usau_game_weight_function(score_w, score_l, w_num, w0=0.5, w_first=24, w_last=42):
    """
    USAU game-weight function, source https://play.usaultimate.org/teams/events/rankings/.
    """
    score_wght = np.min([1, np.sqrt((score_w + np.max([score_l, np.floor(0.5 * (score_w - 1))])) / 19)])
    if w_num >= w_last:
        date_wght = 1
    else:
        date_wght = w0 * ((1 / w0) ** (1 / (w_last - w_first))) ** (w_num - w_first)
    game_wght = score_wght * date_wght

    return game_wght
