import pandas as pd
from tqdm import tqdm
from utils.dataset import get_points_won_matrix


def get_bradley_terry_ratings(
    df_games: pd.DataFrame, iterations: int = 10000
) -> pd.Series:
    _df = df_games[["Team_1", "Team_2", "Score_1", "Score_2"]]

    points_won_matrix = get_points_won_matrix(_df)

    teams = points_won_matrix.index
    n_teams = len(teams)

    ratings = pd.Series([1] * n_teams, index=teams, name="Bradley-Terry rating")
    for i in tqdm(range(n_teams * iterations)):
        team = teams[i % n_teams]
        ratings[team] = _calculate_new_rating(
            ratings=ratings, points_won_matrix=points_won_matrix, team=team
        )
    ratings = _normalize_ratings(ratings)

    return ratings


def _normalize_ratings(ratings: pd.Series) -> pd.Series:
    """Normalize ratings so that the product of the ratings is 1."""
    return ratings / ratings.product() ** (1 / len(ratings))


def _calculate_new_rating(
    ratings: pd.Series, points_won_matrix: pd.DataFrame, team: str
) -> float:
    """
    Calculate the new rating for a team based on the Bradley-Terry model.
    r_i = sum(r_j * w_ij / (r_i + r_j)) / sum(w_ij / (r_i + r_j))

    :param ratings: Series of the current ratings (r)
    :param points_won_matrix: pivot table of won points (w)
    :param team: Team for which the new rating should be calculated
    :return: new rating for the team
    """
    nominator = (points_won_matrix.T[team] * ratings / (ratings + ratings[team])).sum()
    denominator = (points_won_matrix[team] / (ratings + ratings[team])).sum()
    return nominator / denominator
