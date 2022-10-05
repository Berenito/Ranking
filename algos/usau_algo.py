"""
USAU Ranking Algo, source https://play.usaultimate.org/teams/events/rankings/.
With (hopefully) better iteration procedure, more robust to smaller components.
This is a version, which can be called from standard RankingAlgorithm class, there is also
block-based implementation, which can be called from BlockRankingAlgorithm class.
"""

import numpy as np
import pandas as pd
import datetime
import helpfunctions.helpfunctions_dataset as hf_d


def get_ranking_diff_and_game_weight(score_w, score_l):
    """
    See the source for the formula.
    Also the draw is implemented as there was at least 1 draw in the 2021 season.
    """
    if score_w == score_l:
        r_diff = 0
    else:
        r = score_l / (score_w - 1) if score_w != 1 else 1
        r_diff = 125 + 475 * (np.sin(np.min([1, 2 * (1 - r)]) * 0.4 * np.pi) / np.sin(0.4 * np.pi))
    g_wght = np.min([1, np.sqrt((score_w + np.max([score_l, np.floor(0.5 * (score_w - 1))])) / 19)])

    return r_diff, g_wght


def get_date_weight(w_num, w0=0.5, w_first=24, w_last=42):
    """
    Exponential interpolation between 0.5 (first week of the season) and 1 (last week of the season).
    The weeks are probably considered until Monday in case there is tournament finishing on Monday.
    """
    if w_num >= w_last:
        date_wght = 1
    else:
        date_wght = w0 * ((1 / w0) ** (1 / (w_last - w_first))) ** (w_num - w_first)
    return date_wght


def get_ignored_games(df_games, ratings):
    """
    For a game to be ignored, the following conditions have to be satisfied:
        - it has to be a blowout (score_w > 2*score_l + 1)
        - rating difference have to be > 600
        - winner must have at least 5 un-ignored games
    In case only some of the games can be ignored (due to 5-games rule), the most damaging ones are ignored.
    """
    df_games = df_games.copy()
    df_games['Team_Rank_Diff'] = ratings.reindex(df_games['Team_1']).values - ratings.reindex(
        df_games['Team_2']).values
    df_games['Is_Ignored'] = 1 * (df_games['Is_Blowout'] & ((df_games['Team_Rank_Diff'] > 600)
                                                            | df_games['Team_Rank_Diff'].isna()))
    for team in ratings.index:
        n_ignored = df_games.loc[df_games['Team_1'] == team, 'Is_Ignored'].sum()
        n_valid = (1 - df_games.loc[(df_games['Team_1'] == team) | (df_games['Team_2'] == team), 'Is_Ignored']).sum()
        if n_valid < 5 and n_ignored > 0:
            n_unignore = int(min(5 - n_valid, n_ignored))
            idx_ignored = df_games.loc[(df_games['Team_1'] == team) & (df_games['Is_Ignored'] == 1)].sort_values(
                by='Team_Rank_Diff').index.tolist()
            df_games.loc[idx_ignored[:n_unignore], 'Is_Ignored'] = 0

    return df_games['Is_Ignored']


def safe_wma(vals, wghts):
    """
    Weighted moving average, which does not throw error when applied on the empty list and does not take Nans
    into account.
    """
    if vals.shape[0] == 0 or wghts.sum() == 0:
        return np.nan
    else:
        return np.average(vals[~np.isnan(vals)], weights=wghts[~np.isnan(vals)])


def run_ranking_iteration(ratings_start, df_games, n_round=2):
    """
    Set new rating of each team as the weighted average of the ratings of their un-ignored games calculated using the
    current ratings (original version). In this version, you go to ratings_start + 0.5*(ratings_new - ratings_start),
    which has better convergence for smaller number of teams.
    """
    df_games = df_games.copy()
    df_games['Rank_1'] = df_games['Game_Rank_Diff'] + ratings_start.reindex(df_games['Team_2']).values
    df_games['Rank_2'] = ratings_start.reindex(df_games['Team_1']).values - df_games['Game_Rank_Diff']
    df_games['Is_Ignored'] = get_ignored_games(df_games, ratings_start)
    df_games.loc[df_games['Is_Ignored'] == 1, 'Game_Wght'] = 0
    df_games_extended = pd.concat([df_games.rename(columns={'Team_1': 'Team', 'Rank_1': 'Rank'}),
                                   df_games.rename(columns={'Team_2': 'Team', 'Rank_2': 'Rank'})])
    ratings_new = df_games_extended.groupby('Team').apply(lambda x: safe_wma(x['Rank'], x['Game_Wght']))
    # in the original version, 1 is used instead of 0.5
    ratings_new = (ratings_start + 0.5*(ratings_new - ratings_start)).round(n_round)

    return ratings_new


def get_usau_ratings(df_games, rating_start, w0, w_first, w_last):
    """
    Wrapper function to be called from RankingAlgorithm class.
    Params: rating_start (1000 in standard version),
            w0, w_first, w_last (date-weight related, optional, default values 0.5, 24, 42 as for 2021)
    """
    df_games[['Game_Rank_Diff', 'Score_Wght']] = df_games[['Score_1', 'Score_2']].apply(
        lambda x: get_ranking_diff_and_game_weight(x['Score_1'], x['Score_2']), axis=1, result_type='expand')
    df_games['Calendar_Week'] = df_games['Date'].apply(
        lambda x: (datetime.datetime.strptime(x, '%Y-%m-%d') - datetime.timedelta(days=1)).isocalendar()[1])
    df_games['Date_Wght'] = df_games['Calendar_Week'].apply(lambda x: get_date_weight(x, w0, w_first, w_last))
    df_games['Game_Wght'] = df_games['Score_Wght'] * df_games['Date_Wght']
    df_games['Is_Blowout'] = 1 * (df_games['Score_1'] > 2 * df_games['Score_2'] + 1)
    #
    teams = hf_d.get_teams_in_dataset(df_games)
    n_iter, dist_tol = 1000, 1e-5
    df_ratings_iter = pd.DataFrame(rating_start, index=teams, columns=range(n_iter + 1))
    for i in range(n_iter):
        df_ratings_iter[i + 1] = run_ranking_iteration(df_ratings_iter[i], df_games)
        dist_iter = np.sqrt(((df_ratings_iter[i + 1] - df_ratings_iter[i]) ** 2).mean())
        print('{}/{}, Change: {:.7f}, Rating 1: {:.4f}'.format(i + 1, n_iter, dist_iter, df_ratings_iter[i + 1].max()))
        if dist_iter < dist_tol:
            n_iter = i + 1
            break
    df_ratings_iter = df_ratings_iter[range(n_iter + 1)]
    ratings_final = df_ratings_iter[n_iter].sort_values(ascending=False)

    return ratings_final
