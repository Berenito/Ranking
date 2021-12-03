"""
Example to show the intended usage of the implemented GamesDataset and RankingAlgorithm classes.
"""

from ranking_classes import GamesDataset, RankingAlgorithm
from algos.usau_algo import get_usau_ratings
from algos.example_algo import get_example_ratings

# ------------------------------

# Prepare dataset
dataset_name = 'USAU_2021_Men'
dataset_path = 'data/games_usau_cody_2021_men.csv'
# dataset_name = 'USAU_2021_Women'
# dataset_path = 'data/games_usau_cody_2021_women.csv'
# dataset_name = 'USAU_2021_Mixed'
# dataset_path = 'data/games_usau_cody_2021_mixed.csv'
#
usau_dataset = GamesDataset(dataset_path, dataset_name)
usau_dataset.get_summary(save=True)
usau_dataset.get_weekly_summary(save=True)

# Apply Example Algo
example_algo = RankingAlgorithm(get_example_ratings, 'Example_Algo', 
                                lb=100, ub=2000)
usau_dataset.get_ratings(example_algo, save=True)
usau_dataset.get_weekly_ratings(example_algo, save=True)

# Apply USAU Algo 
usau_algo = RankingAlgorithm(get_usau_ratings, 'USAU_Algo', 
                             rating_start=1000, w0=0.5, w_first=29, w_last=42)
usau_dataset.get_ratings(usau_algo, save=True)
usau_dataset.get_weekly_ratings(usau_algo, save=True, verbose=True) # this takes some time

# Check Results 
summary = usau_dataset.summary
weekly_summary = usau_dataset.weekly_summary

# Export Results
# usau_dataset.export_to_excel()
usau_dataset.export_to_excel(include_weekly=True)
#
c_plot_list = ['Rating_USAU_Algo', 'Rating_Example_Algo', 'Games',
               'W_Ratio', 'Opponent_W_Ratio', 'Avg_Point_Diff']
# usau_dataset.plot_fig(c_plot_list)
usau_dataset.plot_fig(c_plot_list, include_weekly=True)

