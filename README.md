# Ranking (Development of EUF ranking system)
Version: 2022-10-05

The goal of this code is to provide a structure to the implementation of the ranking algorithms, 
so they can be immediately applied to any available dataset. It also provides a possibility to gain 
insights about the games datasets and the way to export the results to reasonable formats.

#### Folder structure
* `algos` - specific ranking algorithms scripts 
* `data` - csv files with the games datasets
* `figures` - place for exported figures
* `reports` - place for exported excel files
* `helpfunctions` - various utility functions
* `examples` - scripts showing the intended use of the ranking code
* `tests` - scripts validating the implemented algorithms 

#### Dataset format
Games dataset should be saved in a csv file with the following 6 columns:
* `Tournament` - name of the event
* `Date` - game date in YYYY-MM-DD format
* `Team_1`, `Team_2` - participating teams
* `Score_1`, `Score_2` - resulting scores

#### Algo format
* `RankingAlgorithm(algo_function, algo_name, **kwargs)`
  * `algo_function(df_games, **kwargs) -> ratings`
* `BlockRankingAlgorithm(rank_diff_func, game_weight_func, rank_fit_func, game_ignore_func, algo_name,
rank_diff_params, game_weight_params, rank_fit_params, game_ignore_params)`

#### How to use it
* clone the repo using `git clone https://github.com/Berenito/Ranking.git`
* create python env, install packages to it using `pip install requirements.txt`
* run the example script `example_ranking_script_euf.py`, possibly with new data (change the paths appropriately)
* play with the algo settings


