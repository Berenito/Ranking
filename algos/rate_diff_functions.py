"""
Functions used in the stage 1 of the BlockRankingAlgorithm class ranking procedure - obtaining rank difference
(rank-diff) for each game in the dataset. (They are not intended to be called directly)
During the ranking procedure, get_rank_diff wrapper function is called and it will call desired rank-diff function.
Other, specific rank-diff functions should have format r_diff = rank_diff_func(score_w, score_l, **kwargs).
"""

import numpy as np


# --------------------------------
# MAIN FUNCTION
# --------------------------------

def get_rank_diff(opt, rw, **kwargs):
    """
    Wrapper function to apply specific rank-diff function ('opt') on the given row ('rw') in the Games dataset.
    Option 'opt' can be either string or function reference or basically any identifier that is defined here.
    Scores are read from rw, theoretically also some other info can be obtained from rw.
    **kwargs are specified in {GamesDataset}.rank_diff_params.
    If new rank-diff function is specified, it has to be also added here.
    """
    score_w, score_l = rw['Score_1'], rw['Score_2']
    if opt in ['win_lose', win_lose_rank_diff_function]:
        r_diff = win_lose_rank_diff_function(score_w, score_l)
    elif opt in ['score_diff', 'windmill', score_diff_rank_diff_function]:
        r_diff = score_diff_rank_diff_function(score_w, score_l)
    elif opt in ['usau', usau_rank_diff_function]:
        r_diff = usau_rank_diff_function(score_w, score_l)
    else:
        raise ValueError("Unknown rank-diff option, see algos/rank_diff_functions.py for more info.")

    return r_diff


# --------------------------------
# INDIVIDUAL FUNCTIONS
# --------------------------------

def win_lose_rank_diff_function(score_w, score_l):
    """
    Always 1 for the winner (unless it is draw), regardless of the score.
    """
    r_diff = np.sign(score_w - score_l)

    return r_diff


def score_diff_rank_diff_function(score_w, score_l):
    """
    Basic score difference, used in Windmill algo.
    """
    r_diff = score_w - score_l

    return r_diff


def usau_rank_diff_function(score_w, score_l):
    """
    USAU rank-diff function, source https://play.usaultimate.org/teams/events/rankings/.
    """
    if score_w == score_l:
        r_diff = 0
    else:
        r = score_l / (score_w - 1) if score_w != 1 else 1
        r_diff = 125 + 475 * (np.sin(np.min([1, 2 * (1 - r)]) * 0.4 * np.pi) / np.sin(0.4 * np.pi))

    return r_diff
