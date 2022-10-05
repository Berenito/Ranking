"""
Example to show the intended usage of the implemented GamesDataset and BlockRankingAlgorithm classes.
"""

from ranking_classes import GamesDataset, BlockRankingAlgorithm, ROOT_DIR


division = 'Open'
weekly_ratings = True

# Prepare dataset
dataset_name = 'EUF_2019_{}'.format(division)
dataset_path = '{}/data/games_euf_2019_{}.csv'.format(ROOT_DIR, division.lower())
euf_dataset = GamesDataset(dataset_path, dataset_name)

# Define algos
windmill_algo = BlockRankingAlgorithm(
    algo_name='Windmill',
    rank_diff_func='score_diff',
    game_weight_func='uniform',
    rank_fit_func='regression',
    rank_fit_params={'n_round': 2}
)
usau_algo_no_date_weight = BlockRankingAlgorithm(
    algo_name='USAU',
    rank_diff_func='usau',
    game_weight_func='usau_no_date',
    rank_fit_func='iteration',
    game_ignore_func='blowout',
    rank_fit_params={'rating_start': 1000, 'n_round': 2, 'n_iter': 1000}
)
c_plot_list = ['Rating_USAU', 'Rating_Windmill', 'Games', 'W_Ratio', 'Opponent_W_Ratio', 'Avg_Point_Diff']

# Apply algos & export
euf_dataset.get_ratings(windmill_algo, block_algo=True)
euf_dataset.get_ratings(usau_algo_no_date_weight, block_algo=True)
if weekly_ratings:
    euf_dataset.get_weekly_ratings(windmill_algo, verbose=True)
    euf_dataset.get_weekly_ratings(usau_algo_no_date_weight, verbose=True)  # this takes some time
    euf_dataset.export_to_excel(include_weekly=True)
    euf_dataset.plot_bar_race_fig(c_plot_list, include_weekly=True)
else:
    euf_dataset.export_to_excel()
    euf_dataset.plot_bar_race_fig(c_plot_list)
