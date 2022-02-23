"""
Example to show the intended usage of the implemented GamesDataset and RankingAlgorithm classes.
"""

from ranking_classes import GamesDataset, RankingAlgorithm
from algos.usau_algo import get_usau_ratings
from algos.example_algo import get_example_ratings

# ------------------------------

division = 'Men'

# Prepare dataset
dataset_name = 'USAU_2021_{}'.format(division)
dataset_path = 'data/games_usau_cody_2021_{}.csv'.format(division.lower())

#
usau_dataset = GamesDataset(dataset_path, dataset_name)

# Export Results
usau_dataset.export_to_excel(include_weekly=True)

# Apply Example Algo
example_algo = RankingAlgorithm(get_example_ratings, 'Example_Algo',
                                lb=100, ub=2000)
usau_dataset.get_ratings(example_algo)
usau_dataset.get_weekly_ratings(example_algo)

# Apply USAU Algo 
usau_algo = RankingAlgorithm(get_usau_ratings, 'USAU_Algo',
                             rating_start=1000, w0=0.5, w_first=29, w_last=42)
usau_dataset.get_ratings(usau_algo)
usau_dataset.get_weekly_ratings(usau_algo, verbose=True)  # this takes some time

# Export Results
# usau_dataset.export_to_excel()
usau_dataset.export_to_excel(include_weekly=True)
#
c_plot_list = ['Rating_USAU_Algo', 'Rating_Example_Algo', 'Games',
               'W_Ratio', 'Opponent_W_Ratio', 'Avg_Point_Diff']
# usau_dataset.plot_bar_race_fig(c_plot_list)
usau_dataset.plot_bar_race_fig(c_plot_list, include_weekly=True)

