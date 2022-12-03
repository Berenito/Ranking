"""
Define the GamesDataset class.
"""
import typing as t

import numpy as np
import pandas as pd

from utils import dataset


class GamesDataset:
    """
    Class to work with the Games Dataset - extension of the Games Table.
    """

    def __init__(self, games: t.Union[pd.DataFrame, str], name: str = "Unknown_Dataset", calculate_weekly: bool = True):
        """
        Initialize the dataset.
        
        Examples:
            dataset_from_df = GamesDataset(df_games, name="Dataset_1")
            dataset_from_csv = GamesDataset(csv_path, name="Dataset_2")
            dataset_single_tournament = GamesDataset(df_games, name="Dataset_3", calculate_weekly=False)
        
        :param games: Games Table or a path to its CSV file.
        :param name: Dataset name (for exporting purposes)
        :param calculate_weekly: Whether to calculate weekly statistics (set to False for single-tournament data)
        :return: Initialized GamesDataset object (all dataset-specific metrics are calculated automatically)
        """
        self.name = name
        if isinstance(games, str):
            df_games = pd.read_csv(games, index_col=0)
        elif isinstance(games, pd.DataFrame):
            df_games = games
        self.games = dataset.process_games(df_games)
        self.teams = dataset.get_teams_in_games(self.games)
        self.tournaments = dataset.get_summary_of_tournaments(self.games)
        self.graph = dataset.get_games_graph(self.games)
        self.summary = self.get_summary()
        self.n_games = self.games.shape[0]
        self.n_teams = self.teams.shape[0]
        self.n_tournaments = self.tournaments.shape[0]
        self.n_components = self.summary["Component"].nunique()
        self.max_component_size = self.summary["Component"].value_counts().max()
        self.date_first, self.date_last = self.games["Date"].min(), self.games["Date"].max()
        if calculate_weekly:
            self.calendar = dataset.get_calendar_summary(self.games)
            self.week_ends = [dt for dt, ng in zip(self.calendar["Date_End"], self.calendar["N_Games"]) if ng > 0]
            self.weekly_graph = {dt: dataset.get_games_graph(self.games, dt) for dt in self.week_ends}
            self.weekly_summary = {dt: self.get_summary(dt) for dt in self.week_ends}
            self.add_components_to_calendar()

    def get_summary(self, date: t.Optional[str] = None) -> pd.DataFrame:
        """
        Create summary of the Games Dataset (used during the initialization).
        
        :param date: Date in YYYY-MM-DD format
        :return: Summary DataFrame
        """
        df_summary = dataset.get_summary_of_games(self.games, date)
        # Add graph-component for each team
        if date is None:
            df_summary["Component"] = dataset.get_graph_components(self.graph, self.teams)
        else:
            df_summary["Component"] = dataset.get_graph_components(self.weekly_graph.get(date), self.teams)
        # TODO: Add if the requirements are fulfilled
        return df_summary

    def add_components_to_calendar(self):
        """
        Function to add information about the graph components to the calendar DataFrame.
        
        New added columns:
            N_Components - number of components (based on the teams that already played)
            N_Components_All - number of components (based on all teams in the dataset)
            Max_Component_Size - size of the biggest component
        """
        self.calendar[["N_Components", "N_Components_All", "Max_Component_Size"]] = np.nan
        for date, df_summary in self.weekly_summary.items():
            is_date = self.calendar["Date_End"] == date
            self.calendar.loc[is_date, "N_Components"] = df_summary["Component"].nunique()
            self.calendar.loc[is_date, "Max_Component_Size"] = df_summary["Component"].value_counts().max()
        self.calendar["N_Components_All"] = self.calendar["N_Components"] + self.n_teams - self.calendar["N_Teams_Cum"]
        self.calendar[["N_Components", "N_Components_All", "Max_Component_Size"]] = self.calendar[
            ["N_Components", "N_Components_All", "Max_Component_Size"]
        ].fillna(method="ffill").astype("int")

    def filter_games(
        self, date: t.Optional[str] = None, team: t.Optional[str] = None, tournament: t.Optional[str] = None
    ) -> pd.DataFrame:
        """
        Return games for given team / tournament / up to the given date.
        :param date: Date in YYYY-MM-DD format
        :param team: Team name
        :param tournament: Tournament name
        :return: Filtered Games Table
        """
        if date is not None:
            df_games = self.games.loc[self.games["Date"] <= date]
        else:
            df_games = self.games
        if team is not None:
            df_games = dataset.get_games_for_teams(df_games, team, "any")
        elif tournament is not None:
            df_games = df_games.loc[df_games["Tournament"] == tournament]
        return df_games.reset_index(drop=True)
