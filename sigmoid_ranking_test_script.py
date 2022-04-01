"""
Example to show the intended usage of the implemented GamesDataset and RankingAlgorithm classes.
"""

from ranking_classes import GamesDataset, RankingAlgorithm
from algos.sigmoid_algo import get_sigmoid_ratings

# ------------------------------

division = 'Men'

# Prepare dataset
dataset_name = 'USAU_2021_{}'.format(division)
dataset_path = 'data/games_usau_cody_2021_{}.csv'.format(division.lower())

#dataset_path = 'data/test_data.csv'
#dataset_name = 'Test_Data'

usau_dataset = GamesDataset(dataset_path, dataset_name)

# Apply Sigmoid Algo
sigmoid_algo = RankingAlgorithm(get_sigmoid_ratings, 'Sigmoid_Algo')
usau_dataset.get_ratings(sigmoid_algo)
#usau_dataset.get_weekly_ratings(sigmoid_algo)

#c_plot_list = ['Rating_Sigmoid_Algo', 'Games', 'W_Ratio', 'Opponent_W_Ratio', 'Avg_Point_Diff']

#usau_dataset.plot_bar_race_fig(c_plot_list, include_weekly=False)
