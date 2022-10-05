"""
Example to show the intended usage of the implemented GamesDataset and BlockRankingAlgorithm classes.
"""

from ranking_classes import GamesDataset, BlockRankingAlgorithm, ROOT_DIR

division = 'Men'
weekly_ratings = False

# Prepare dataset
dataset_name = 'USAU_2021_{}'.format(division)
dataset_path = '{}/data/games_usau_cody_2021_{}.csv'.format(ROOT_DIR, division.lower())
usau_dataset = GamesDataset(dataset_path, dataset_name)

# Define algos
windmill_algo = BlockRankingAlgorithm(
    algo_name='Windmill',
    rank_diff_func='score_diff',
    game_weight_func='uniform',
    rank_fit_func='regression',
    rank_fit_params={'n_round': 2}
)
usau_algo = BlockRankingAlgorithm(
    algo_name='USAU',
    rank_diff_func='usau',
    game_weight_func='usau',
    rank_fit_func='iteration',
    game_ignore_func='blowout',
    game_weight_params={'w0': 0.5, 'w_first': 29, 'w_last': 42},
    rank_fit_params={'rating_start': 1000, 'n_round': 2, 'n_iter': 1000}
)
c_plot_list = ['Rating_USAU', 'Rating_Windmill', 'Games', 'W_Ratio', 'Opponent_W_Ratio', 'Avg_Point_Diff']

# Apply algos & export
usau_dataset.get_ratings(windmill_algo, block_algo=True)
usau_dataset.get_ratings(usau_algo, block_algo=True)
if weekly_ratings:
    usau_dataset.get_weekly_ratings(windmill_algo, verbose=True)
    usau_dataset.get_weekly_ratings(usau_algo, verbose=True)  # this takes some time
    usau_dataset.export_to_excel(include_weekly=True)
    usau_dataset.plot_bar_race_fig(c_plot_list, include_weekly=True)
else:
    usau_dataset.export_to_excel()
    usau_dataset.plot_bar_race_fig(c_plot_list)
