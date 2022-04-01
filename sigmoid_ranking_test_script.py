"""
Example to show the intended usage of the implemented GamesDataset and RankingAlgorithm classes.
"""

from ranking_classes import GamesDataset, RankingAlgorithm
from algos.usau_algo import get_usau_ratings
from algos.sigmoid_algo import get_sigmoid_ratings
# from helpfunctions.helpfunctions_excel import export_to_excel

# ------------------------------

division = 'Men'

# Prepare dataset
dataset_name = 'USAU_2021_{}'.format(division)
dataset_path = 'data/games_usau_cody_2021_{}.csv'.format(division.lower())

usau_dataset = GamesDataset(dataset_path, dataset_name)

# # Apply USAU Algo
# usau_algo = RankingAlgorithm(get_usau_ratings, 'USAU_Algo',
#                              rating_start=1000, w0=0.5, w_first=29, w_last=42)
# usau_dataset.get_ratings(usau_algo)

# Apply Sigmoid Algo
sigmoid_algo = RankingAlgorithm(get_sigmoid_ratings, 'Sigmoid_Algo')
usau_dataset.get_ratings(sigmoid_algo)

#
# export_to_excel(usau_dataset)
usau_dataset.summary.to_csv("temp_report_sigmoid.csv")
