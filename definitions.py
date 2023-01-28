"""
Define all important constants in this file.
"""
from classes.block_ranking_algorithm import BlockRankingAlgorithm

# Ranking eligibility requirements
MIN_TOURNAMENTS = 2
MIN_GAMES = 10
MAX_COMPONENT_REQUIRED = True


# Algorithm definitions
USAU_ALGO = BlockRankingAlgorithm(
    algo_name="usau",
    rank_diff_func="usau",
    game_weight_func="usau_no_date",
    rank_fit_func="iteration",
    game_ignore_func="blowout",
    game_weight_params={"w0": 0.5, "w_first": 29, "w_last": 42},
    rank_fit_params={"rating_start": 0, "n_round": 2, "n_iter": 1000, "verbose": True},
    game_ignore_params={"min_valid": MIN_GAMES},
)

WINDMILL_ALGO = BlockRankingAlgorithm(
    algo_name='windmill',
    rank_diff_func='score_diff',
    game_weight_func='uniform',
    rank_fit_func='regression',
    rank_fit_params={'n_round': 2}
)
