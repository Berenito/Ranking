"""
Definitions of classes used for EUF ranking system development.
"""

import os
import pandas as pd
import numpy as np
import helpfunctions.helpfunctions_dataset as hf_d
import helpfunctions.helpfunctions_plotly as hf_p
#import helpfunctions.helpfunctions_excel as hf_e
import algos.rank_diff_functions as rdf
import algos.game_weight_functions as gwf
import algos.rank_fit_functions as rff


# --------------------------------------------
# CLASS FOR WORKING WITH DATASET WITH GAMES
# --------------------------------------------

class GamesDataset:
    """
    Class to work with dataset of games.
    """

    def __init__(self, games, name='Unknown_Dataset', calculate_weekly=True):
        """
        Initialize the dataset.
        -----
        Input:
            games - pd.DataFrame with games or a path to the csv file.
            name - used for naming exported files
            calculate_weekly - whether to calculate weekly statistics (set to False for single-tournament data)
        Output:
            initialized GamesDataset object (all dataset-specific metrics are calculated automatically)
        Examples:
            dataset_from_df = GamesDataset(df_games, name='Dataset_1')
            dataset_from_csv = GamesDataset(csv_path, name='Dataset_2')
            dataset_single_tournament = GamesDataset(df_games, name='Dataset_3', calculate_weekly=False)
        """
        self.name = name
        if isinstance(games, str):
            df_games = pd.read_csv(games, index_col=0)
        elif isinstance(games, pd.DataFrame):
            df_games = games
        self.games = hf_d.process_dataset(df_games)
        self.teams = hf_d.get_teams_in_dataset(self.games)
        self.tournaments = hf_d.get_summary_of_tournaments(self.games)
        self.graph = hf_d.get_games_graph(self.games)
        self.summary = self.get_summary()
        self.n_games = self.games.shape[0]
        self.n_teams = self.teams.shape[0]
        self.n_tournaments = self.tournaments.shape[0]
        self.n_components = self.summary['Component'].nunique()
        self.max_component_size = self.summary['Component'].value_counts().max()
        self.date_first, self.date_last = self.games['Date'].min(), self.games['Date'].max()
        if calculate_weekly:
            self.calendar = hf_d.get_calendar_summary(self.games)
            self.week_ends = [dt for dt, ng in zip(self.calendar['Date_End'], self.calendar['N_Games']) if ng > 0]
            self.weekly_graph = {dt: hf_d.get_games_graph(self.games, dt) for dt in self.week_ends}
            self.weekly_summary = {dt: self.get_summary(dt) for dt in self.week_ends}
            self.add_components_to_calendar()

    # -----

    def get_summary(self, date=None):
        """
        Create summary table from the games dataset.
        Used during the initialization.
        """
        df_summary = hf_d.get_summary_of_games(self.games, date)
        if date is None:
            df_summary['Component'] = hf_d.get_graph_components(self.graph, self.teams)
        else:
            df_summary['Component'] = hf_d.get_graph_components(self.weekly_graph.get(date), self.teams)

        return df_summary

    # -----

    def get_weekly_summary(self):
        """
        Create summary table from the games dataset for every reasonable (new games played) Sunday of the dataset span.
        Used during the initialization.
        """
        dict_summary = {dt: self.get_summary(dt) for dt, ng in zip(self.calendar['Date_End'], self.calendar['N_Games'])
                        if ng > 0}

        return dict_summary

    # -----

    def add_components_to_calendar(self):
        """
        Function to add information about the graph components to the calendar dataframe.
        New added columns:
            N_Components - number of components (based on the teams that already played)
            N_Components_All - number of components (based on all teams in the dataset)
            Max_Component_Size - size of the biggest component
        """
        self.calendar[['N_Components', 'N_Components_All', 'Max_Component_Size']] = np.nan
        for dt, df_summary in self.weekly_summary.items():
            self.calendar.loc[self.calendar['Date_End'] == dt, 'N_Components'] = df_summary['Component'].nunique()
            self.calendar.loc[self.calendar['Date_End'] == dt, 'Max_Component_Size'] = \
                df_summary['Component'].value_counts().max()
            self.calendar.loc[self.calendar['Date_End'] == dt, 'N_Components_All'] = \
                df_summary['Component'].nunique() + self.n_teams - \
                self.calendar.loc[self.calendar['Date_End'] == dt, 'N_Teams_Cum']
        self.calendar[['N_Components', 'N_Components_All', 'Max_Component_Size']] = \
            self.calendar[['N_Components', 'N_Components_All', 'Max_Component_Size']].fillna(method='ffill')

    # -----

    def filter_games(self, date=None, team=None, tournament=None):
        """
        Return games for given team / tournament up to the given date.
        """
        if date is not None:
            df_games = self.games.loc[self.games['Date'] <= date]
        else:
            df_games = self.games
        if team is not None:
            df_out = df_games.loc[(df_games['Team_1'] == team) | (df_games['Team_2'] == team)]
        elif tournament is not None:
            df_out = df_games.loc[df_games['Tournament'] == tournament]

        return df_out.reset_index(drop=True)

    # -----

    def get_ratings(self, ranking_algo, sort=True, block_algo=False):
        """
        Calculate ratings based on provide RankingAlgorithm or object.
        -----
        Input:
            ranking_algo - RankingAlgorithm object
            sort - whether to sort the output by the ratings
            block_algo - whether to use BlockRankingAlgorithm functionality (fill the ranking-procedure information also
            to self.games, not just self.summary)
        Output:
            updates self.summary and potentially self.games
        Examples:
            dataset.get_ratings(usau_algo)
            dataset.get_ratings(usau_block_algo, block_algo=True)
        """
        if block_algo:
            ratings, self.games = ranking_algo.get_ratings(self, return_games=True)
        else:
            ratings = ranking_algo.get_ratings(self)
        self.summary = pd.concat([ratings, self.summary], axis=1)
        if sort:
            self.summary = self.summary.sort_values(by='Rating_{}'.format(ranking_algo.name),
                                                    ascending=False)

    # ----------

    def get_weekly_ratings(self, ranking_algo, sort=True, verbose=False):
        """
        Calculate weekly ratings for every Sunday of the dataset span.
        Does not support BlockRankingAlgorithm additional functionality (i.e. you can use BlockRankingAlgorithm to
        obtain weekly ratings, but the ranking-procedure information for the individual games cannot be exported yet)
        -----
        Input:
            ranking_algo - RankingAlgorithm object
            sort - whether to sort the output by the ratings
            verbose - whether to print the information about the progress
        Output:
            new column in self.weekly_summary
        Examples:
            dataset.get_weekly_ratings(usau_algo, verbose=True)
        """
        weekly_ratings = {}
        for dt in self.week_ends:
            if verbose:
                print(dt)
            weekly_ratings[dt] = ranking_algo.get_ratings(self, date=dt)
        for dt in self.weekly_summary.keys():
            self.weekly_summary[dt] = pd.concat([weekly_ratings.get(dt), self.weekly_summary.get(dt)], axis=1)
            if sort:
                self.weekly_summary[dt] = self.weekly_summary.get(dt).sort_values(
                    by='Rating_{}'.format(ranking_algo.name),
                    ascending=False)

    # ----------

    def export_to_excel(self, filename=None, include_weekly=False):
        """
        Export dataset information to excel (make sure all the information are calculated).
        At the moment will return 4 sheets anytime:
            Games (with possible ranking-procedure information included),
            Summary (with possible ratings included)
            Tournaments
            Calendar
        + optionally sheet for every week's summary (with possible ratings included).
        -----
        Input:
            filename - filename to save, None -> will be saved in reports folder (make sure to create it)
                       (unfortunately, it does not work properly with relative paths)
            include_weekly - whether to include also weekly summary
        Output:
            saved xlsx file
        Examples:
            dataset.export_to_excel(filename_to_save)
            dataset.export_to_excel(include_weekly=True)
        """
        sfx = '_weekly' if include_weekly else ''
        fl = os.path.join(os.getcwd(), 'reports', 'data_{}{}.xlsx'.format(self.name.lower().replace(' ', '_'),
                                                                          sfx)) if filename is None else filename
        df_list = [self.games.set_index('Tournament'), self.summary, self.tournaments,
                   self.calendar.reset_index().set_index('Year')]
        sheet_names = ['{} {}'.format(k, self.name) for k in ['Games', 'Summary', 'Tournaments', 'Calendar']]
        if include_weekly:
            df_list.extend([s for _, s in self.weekly_summary.items()])
            sheet_names.extend(['Summary {}'.format(dt) for dt, _ in self.weekly_summary.items()])
        hf_e.create_excel_file_from_df_list(fl, df_list, sheet_names=sheet_names)

    # ----------

    def plot_bar_race_fig(self, c_plot_list, filename=None, include_weekly=False):
        """
        Export to experimental visualization of the dataset progress (top 20 teams).
        -----
        Input:
            c_plot_list - columns to plot (max 5-6)
                          will be sorted by the first element
            filename - filename to save, None -> will be saved in figures folder (make sure to create it)
            include_weekly - whether to include also weekly summary
        Output:
            saved html figure
        Examples:
            dataset.plot_bar_race_fig(['W_Ratio', 'Games', 'W_Ratio', 'Opponent_W_Ratio'], include_weekly=True)
        """
        sfx = '-weekly' if include_weekly else ''
        fl = os.path.join(os.getcwd(), 'figures', 'fig-{}{}.html'.format(
            self.name.lower().replace(' ', '-').replace('_', '-'), sfx)) if filename is None else filename
        dict_plot = self.weekly_summary if include_weekly else {'All Games': self.summary}
        hf_p.plot_bar_race_chart(dict_plot, c_plot_list, fl, self.name)


# ---------------------------------------------------
# CLASS FOR WORKING WITH RANKING ALGORITHM
# ---------------------------------------------------

class RankingAlgorithm:
    """
    Class to work with ranking algorithm. Closely connected to GamesDataset class.
    """

    def __init__(self, rating_func, algo_name='Unknown_Algo', **kwargs):
        """
        Initialize the algorithm.
        -----
        Input:
            rating_func - function (df_games, **kwargs) -> ratings
            algo_name - used for naming exported stuff
            kwargs - additional inputs to rating_func
        Output:
            initialized RankingAlgorithm object
        Examples:
            example_algo = RankingAlgorithm(example_func, algo_name='Example_Algo', p1=100, p2=200)
            usau_algo = RankingAlgorithm(get_usau_ratings, algo_name='USAU_Algo')
        """
        self.name = algo_name
        self.rating_func = rating_func
        self.params = kwargs

    # -----

    def get_ratings(self, dataset, date=None):
        """
        Calculate the ratings of the provided dataset.
        -----
        Input:
            dataset - GamesDataset object or df_games
            date - only games up to date will be included, None -> include all
        Output:
            ratings - series with the calculated ratings
        Examples:
            ratings = example_algo.get_ratings(dataset_1)
        """
        if isinstance(dataset, pd.DataFrame):
            df_games = dataset.copy()
        elif isinstance(dataset, GamesDataset):
            df_games = dataset.games.copy()
        if date is not None:
            df_games = df_games.loc[df_games['Date'] <= date]
        ratings = self.rating_func(df_games, **self.params)
        ratings = ratings.rename('Rating_{}'.format(self.name))

        return ratings


# ---------------------------------------------------
# CLASS FOR WORKING WITH BLOCK RANKING ALGORITHM
# ---------------------------------------------------

class BlockRankingAlgorithm:
    """
    Class to work with ranking algorithm defined with the help of blocks. Closely connected to GamesDataset class.
    """

    def __init__(self, rank_diff_func='win_lose', game_weight_func='uniform', rank_fit_func='regression',
                 game_ignore_func=None, algo_name='Unknown_Algo', rank_diff_params={},
                 game_weight_params={}, rank_fit_params={}, game_ignore_params={}):
        """
        Initialize the algorithm.
        -----
        Input:
            rank_diff_func - rank-diff function identifier
            game_weight_func - game-weight function identifier
            rank_fit_func - rank-fit function identifier
            game_ignore_func - game-ignore function identifier
            algo_name - used for naming exported stuff
            rank_diff_params - additional inputs to rank_diff_func
            game_weight_params - additional inputs to game_weight_func
            rank_fit_params - additional inputs to rank_fit_func
            game_ignore_params - additional inputs to game_ignore_func
        Output:
            initialized BlockRankingAlgorithm object
        Examples:
            windmill_algo = BlockRankingAlgorithm(algo_name='Windmill', rank_diff_func='score_diff',
                                                  game_weight_func='uniform', rank_fit_func='regression',
                                                  rank_fit_params={'n_round': 2})
            usau_algo = BlockRankingAlgorithm(algo_name='USAU', rank_diff_func='usau', game_weight_func='usau',
                                              rank_fit_func='iteration', game_ignore_func='blowout',
                                              game_weight_params={'w0': 0.5, 'w_first': 29, 'w_last': 42},
                                              rank_fit_params={'rating_start': 1000, 'n_round': 2, 'n_iter': 1000})
        """
        self.name = algo_name
        self.rank_diff_func = rank_diff_func
        self.game_weight_func = game_weight_func
        self.rank_fit_func = rank_fit_func
        self.game_ignore_func = game_ignore_func
        self.rank_diff_params = rank_diff_params
        self.game_weight_params = game_weight_params
        self.rank_fit_params = rank_fit_params
        self.game_ignore_params = game_ignore_params

    # -----

    def get_ratings(self, dataset, return_games=False, date=None):
        """
        Calculate the ratings of the provided dataset.
        -----
        Input:
            dataset - GamesDataset object or df_games
            date - only games up to date will be included, None -> include all
            return_games - return also df_games with ranking-procedure information
                           (such that {GamesDataset}.get_ratings(algo, block_algo=True) works)
        Output:
            ratings - series with the calculated ratings
        Examples:
            ratings = example_algo.get_ratings(dataset_1)
            ratings, df_games = example_algo.get_ratings(dataset_1, return_games=True)
        """
        if isinstance(dataset, pd.DataFrame):
            df_games = dataset.copy()
            teams = hf_d.get_teams_in_dataset(df_games)
            graph = hf_d.get_games_graph(df_games)
            df_components = hf_d.get_graph_components(graph, teams)
        elif isinstance(dataset, GamesDataset):
            df_games = dataset.games.copy()
            df_components = dataset.summary['Component']
            teams = dataset.teams
        if date is not None:  # to support also {GamesDataset}.get_weekly_ratings method
            df_games = df_games.loc[df_games['Date'] <= date]
            teams = hf_d.get_teams_in_dataset(df_games)
            graph = hf_d.get_games_graph(df_games)
            df_components = hf_d.get_graph_components(graph, teams)
        df_games['Game_Rank_Diff'] = df_games.apply(lambda rw: rdf.get_rank_diff(self.rank_diff_func, rw,
                                                                                 **self.rank_diff_params), axis=1)
        df_games['Game_Wght'] = df_games.apply(lambda rw: gwf.get_game_weight(self.game_weight_func, rw,
                                                                              **self.game_weight_params), axis=1)
        ratings, df_games['Is_Ignored'], df_games['Team_Rank_Diff'] = \
            rff.get_rank_fit(self.rank_fit_func, self.game_ignore_func, teams, df_games, df_components,
                             self.game_ignore_params, **self.rank_fit_params)
        ratings = ratings.rename('Rating_{}'.format(self.name))
        if return_games:
            df_games = df_games.rename(columns={'Game_Rank_Diff': 'Game_Rank_Diff_{}'.format(self.name),
                                                'Team_Rank_Diff': 'Team_Rank_Diff_{}'.format(self.name),
                                                'Game_Wght': 'Game_Wght_{}'.format(self.name),
                                                'Is_Ignored': 'Is_Ignored_{}'.format(self.name)})
            return ratings, df_games
        else:
            return ratings
