"""
Define all important constants in this file.
"""
from algos.bradley_terry import get_bradley_terry_ratings
from classes.block_ranking_algorithm import BlockRankingAlgorithm
from classes.ranking_algorithm import RankingAlgorithm

# Ranking eligibility requirements
MIN_TOURNAMENTS = 2
MIN_GAMES = 10
MIN_INTERCONNECTIVITY = 30
MAX_COMPONENT_REQUIRED = True


# Algorithm definitions
USAU_ALGO = BlockRankingAlgorithm(
    algo_name="USAU",
    rank_diff_func="usau",
    game_weight_func="usau_no_date",
    rank_fit_func="iteration",
    game_ignore_func="blowout",
    rank_fit_params={"rating_start": 0, "n_round": 2, "n_iter": 1000, "verbose": True},
)

WINDMILL_ALGO = BlockRankingAlgorithm(
    algo_name="Windmill",
    rank_diff_func="score_diff",
    game_weight_func="uniform",
    rank_fit_func="regression",
    rank_fit_params={"n_round": 2},
)

# TODO - update sigmoid algo with better rank-diff & game-weight functions, check the fairness
SIGMOID_ALGO = BlockRankingAlgorithm(
    algo_name="Sigmoid",
    rank_diff_func="score_diff",
    game_weight_func="uniform",
    rank_fit_func="minimization",
    rank_transform_func="sigmoid",
    rank_fit_params={"n_round": 2},
)

BRADLEY_TERRY_ALGO = RankingAlgorithm(
    rating_func=get_bradley_terry_ratings, algo_name="Bradley-Terry"
)

DIVISIONS = ["mixed", "open", "women"]

DIVISION_ALIASES = {
    "open": ["open", "men"],
    "women": ["women"],
    "mixed": ["mixed", "mix"],
}

ALGORITHMS = [USAU_ALGO, BRADLEY_TERRY_ALGO]
