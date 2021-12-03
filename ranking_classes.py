"""
Definitions of classes used for EUF ranking system development.
-----
Last Update: 2021-12-03
"""

import os
import pandas as pd
import helpfunctions_dataset as hf_d
import helpfunctions_plotly as hf_p
import helpfunctions_excel as hf_e  

# --------------------------------------------
# CLASS FOR WORKING WITH DATASET WITH GAMES
# --------------------------------------------

class GamesDataset():
    """
    Class to work with dataset of games.
    """
    def __init__(self, games, name='Unknown_Dataset'):
        """
        Initialize the dataset.
        -----
        Input:
            games - pd.DataFrame with games or a path to the csv file.
            name - used for naming exported files
        Output:
            initialized GamesDataset object 
        Examples:
            dataset_from_df = GamesDataset(df_games, name='Dataset_1')
            dataset_from_csv = GamesDataset(csv_path, name='Dataset_2')
        """
        self.name = name
        if isinstance(games, str):
            df_games = pd.read_csv(games, index_col=0)
        elif isinstance(games, pd.DataFrame):
            df_games = games
        self.games = hf_d.process_dataset(df_games)
        self.teams = hf_d.get_teams_in_dataset(self.games)
        date_first, date_last = self.games['Date'].min(), self.games['Date'].max()
        self.date_range = pd.date_range(date_first, date_last, freq='W').strftime('%Y-%m-%d')
        self.summary, self.weekly_summary = None, None

    # -----

    def get_summary(self, date=None, save=False):
        """
        Create summary table from the games dataset.
        -----
        Input:
            date - only games up to date will be included, None -> include all 
            save - False -> return summary dataframe, True -> save to self.summary attribute
                   (can be saved to self.summary only if date is None)
        Output:
            df_summary (self.summary) - pd.DataFrame with columns 'Games', 'Wins', 'Losses', 
                'W_Ratio', 'Opponent_W_Ratio', 'Goals_For', 'Goals_Against', 'Avg_Point_Diff'
        Examples:
            df_summary = dataset.get_summary(date='2021-07-31')
            dataset.get_summary(save=True)
        """
        df_summary = hf_d.get_summary_of_games(self.games, date)
        if save and date is None: # only summary of whole dataset can be saved
            self.summary = df_summary
        else:
            return df_summary
        
    # -----
    
    def get_weekly_summary(self, save=False):
        """
        Create summary table from the games dataset for every Sunday of the dataset span.
        -----
        Input:
            save - False -> return weekly_summary dict, True -> save to self.weekly_summary attribute
        Output:
            dict_summary (self.weekly_summary) - dictionary of summaries for many dates
        Examples:
            dict_summary = get_weekly_summary()
            get_weekly_summary(save=True)
        """
        dict_summary = {dt: self.get_summary(dt) for dt in self.date_range}
        if save:
            self.weekly_summary = dict_summary
        else:
            return dict_summary
    
    # -----
    
    def get_ratings(self, ranking_algo, date=None, save=False, sort=True):
        """
        Calculate ratings based on provide RankingAlgorithm object.
        -----
        Input:
            ranking_algo - RankingAlgorithm object
            date - only games up to date will be included, None -> include all 
            save - False -> return rating series, True -> add to self.summary attribute
                   (can be saved to self.summary only if date is None)
            sort - whether to sort the output by the ratings
        Output:
            ratings series or new column in self.summary
        Examples:
            ratings = dataset.get_ratings(example_algo, date='2021-07-31')
            dataset.get_ratings(usau_algo, save=True)
        """
        ratings = ranking_algo.get_ratings(self, date)
        if save and date is None: # only summary of whole dataset can be saved
            if self.summary is None:
                self.get_summary(save=True)
            self.summary = pd.concat([ratings, self.summary], axis=1)
            if sort:
                self.summary = self.summary.sort_values(by='Rating_{}'.format(ranking_algo.name),
                                                        ascending=False)
        else:
            if sort:
                ratings = ratings.sort_values(ascending=False)
            return ratings
        
    # ----------
    
    def get_weekly_ratings(self, ranking_algo, save=False, sort=True, verbose=False):
        """
        Calculate weekly ratings for every Sunday of the dataset span.
        -----
        Input:
            ranking_algo - RankingAlgorithm object
            save - False -> return weekly_ratings dict, True -> add to self.weekly_summary attribute
            sort - whether to sort the output by the ratings
            verbose - whether to print the information about the progress
        Output:
            weekly_ratings dict or new column in self.weekly_summary
        Examples:
            dict_weekly_ratings = dataset.get_weekly_ratings(example_algo)
            dataset.get_ratings(usau_algo, save=True)
        """
        weekly_ratings = {}
        for dt in self.date_range:                
            if verbose:
                    print(dt)
            weekly_ratings[dt] = ranking_algo.get_ratings(self, dt)
        if save:
            if self.weekly_summary is None:
                self.get_weekly_summary(save=True)
            for dt in self.date_range:
                self.weekly_summary[dt] = pd.concat([weekly_ratings.get(dt), self.weekly_summary[dt]], axis=1)
                if sort:
                    self.weekly_summary[dt] = self.weekly_summary.get(dt).sort_values(by='Rating_{}'.format(ranking_algo.name),
                                                                                          ascending=False)    
        else:
            if sort:
                weekly_ratings = {dt: rt.sort_values(ascending=False) for dt, rt in weekly_ratings.items()}
            return weekly_ratings
    
    # ----------
    
    def export_to_excel(self, filename=None, include_weekly=False):
        """
        Export dataset information to excel (make sure all the information are calculated).
        At the moment will return 2 sheets anytime: Games, Summary (with possible ratings included)
        + optionally sheet for every week's summary.
        -----
        Input:
            filename - filename to save, None -> will be saved in reports folder (make sure to create it)
                       (unfortunately, it does not work properly with relative paths)
            include_weekly - whether to include also weekly summary
        Output:
            saved xlsx file
        Examples:
            dataset.export_to_excel(filename_to_save)
            dataset.export_to_excel()
        """
        sfx = '_weekly' if include_weekly else ''
        fl = os.path.join(os.getcwd(), 'reports', 'data_{}{}.xlsx'.format(self.name.lower().replace(' ', '_'), sfx)) if filename is None else filename
        df_list = [self.games.set_index('Tournament'), self.summary]
        sheet_names = ['{} {}'.format(k, self.name) for k in ['Games', 'Summary']]
        if include_weekly:
            df_list.extend([s for _, s in self.weekly_summary.items()])
            sheet_names.extend(['Summary {}'.format(dt) for dt, _ in self.weekly_summary.items()])
        hf_e.create_excel_file_from_df_list(fl, df_list, sheet_names=sheet_names)
    
    # ----------
    
    def plot_fig(self, c_plot_list, filename=None, include_weekly=False):
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
            dataset.plot_weekly_bar_race_chart(['W_Ratio', 'Games', 'W_Ratio', 'Opponent_W_Ratio'])
        """
        sfx = '-weekly' if include_weekly else ''
        fl = os.path.join(os.getcwd(), 'figures', 'fig-{}{}.html'.format(self.name.lower().replace(' ', '-'), sfx)) if filename is None else filename
        dict_plot = self.weekly_summary if include_weekly else {'All Games': self.summary}
        hf_p.plot_bar_race_chart(dict_plot, c_plot_list, fl, self.name)
    
# ---------------------------------------------------
# CLASS FOR WORKING WITH RANKING ALGORITHM
# ---------------------------------------------------

class RankingAlgorithm():
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
            dataset - GamesDataset object
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
        ratings = self.rating_func(df_games, **self.params).rename('Rating_{}'.format(self.name))
        
        return ratings
    