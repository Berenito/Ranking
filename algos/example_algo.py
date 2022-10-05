"""
Possible easy example of ranking algo usage.
"""

import helpfunctions.helpfunctions_dataset as hf_d


def get_example_ratings(df_games, lb, ub):
    """
    Just some easy example of rating function, lb and ub should be defined 
    during RankingAlgorithm initialization.
    """
    df_summary = hf_d.get_summary_of_games(df_games)
    ratings = df_summary['W_Ratio'] * df_summary['Opponent_W_Ratio']
    ratings = lb + ratings*(ub - lb)
    
    return ratings
