"""
Functions used during the rank-fitting stage of the BlockRankingAlgorithm class ranking procedure 
(they are not intended to be called directly).
During the ranking procedure, get_ignored_games wrapper function is called and it will call desired game-ignore function
Other, specific game-ignore functions use input df_games, ratings and potentially function-specific **kwargs.
"""
import typing as t

import pandas as pd


def get_ignored_games(
    option: t.Optional[t.Union[str, t.Callable]], df_games: pd.DataFrame, ratings: pd.Series, **kwargs
) -> pd.Series:
    """
    Apply specific game-ignore function given by "option" on the given Games Dataset based on the current ratings.
    :param option: Game-ignore function identifier
    :param df_games: Games Table
    :param ratings: Current ratings
    :param kwargs: Additional parameters to the game-ignore function
    :return: Series with ignored games
    """
    df_games = df_games.copy()
    if option in [None, "none", no_game_ignore_function]:
        is_ignored = no_game_ignore_function(df_games)
    elif option in ["usau", "blowout", blowout_game_ignore_function]:
        is_ignored = blowout_game_ignore_function(df_games, ratings, **kwargs)
    else:
        raise ValueError("Unknown game-ignore option, make sure it is defined in algos/game_ignore_functions.py.")
    return is_ignored


def no_game_ignore_function(df_games: pd.DataFrame) -> pd.Series:
    """
    Use all games.
    """
    return pd.Series(0, name="Is_Ignored", index=df_games.index)


def blowout_game_ignore_function(
    df_games: pd.DataFrame,
    ratings: pd.Series,
    min_rank_diff: float = 600,
    min_valid: int = 5,
    is_blowout: t.Callable = lambda x: x["Score_1"] > 2 * x["Score_2"] + 1,
) -> pd.Series:
    """
    Ignore blowout wins that would be damaging for the winner.

    For a game to be ignored, the following conditions have to be satisfied:
        * it has to be a blowout (score_w > 2*score_l + 1) (blowout_fcn)
        * team rating difference have to be > 600 (min_rank_diff)
        * winner must have at least 5 (min_valid) un-ignored games
    In case only some of the games can be ignored (due to 5-games rule), the most damaging ones are ignored.
    Default case is equal to the ignore function used in the USAU algorithm, but now can be also generalized easily.

    :param df_games: Games Table
    :param ratings: Current ratings
    :param min_rank_diff: Minimum team rating difference
    :param min_valid: Minimum valid games
    :param is_blowout: Function to determine whether a game is considered a blowout
    """
    df_games["Is_Blowout"] = df_games.apply(is_blowout, axis=1)
    df_games["Is_Ignored"] = 1 * (
        df_games["Is_Blowout"] & ((df_games["Team_Rank_Diff"] > min_rank_diff) | df_games["Team_Rank_Diff"].isna())
    )
    for team in ratings.index:
        n_ignored = df_games.loc[df_games["Team_1"] == team, "Is_Ignored"].sum()
        n_valid = (1 - df_games.loc[(df_games["Team_1"] == team) | (df_games["Team_2"] == team), "Is_Ignored"]).sum()
        if n_valid < min_valid and n_ignored > 0:
            n_unignore = int(min(min_valid - n_valid, n_ignored))
            idx_ignored = df_games.loc[(df_games["Team_1"] == team) & (df_games["Is_Ignored"] == 1)].sort_values(
                by="Team_Rank_Diff"
            ).index.tolist()
            df_games.loc[idx_ignored[: n_unignore], "Is_Ignored"] = 0
    return df_games["Is_Ignored"]
