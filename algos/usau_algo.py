# -*- coding: utf-8 -*-
"""
USAU Ranking Algo.
This is still just WIP version, so don't take it too seriously,
there are some small differences from frisbee-rankings.com website.
"""

import numpy as np
import pandas as pd
import datetime
import helpfunctions_dataset as hf_d

# -------------------

def get_ranking_diff_and_game_weight(score_w, score_l):
    """
    """
    r = score_l / (score_w - 1) if score_w != 1 else 1
    r_diff = 125 + 475*(np.sin(np.min([1, 2*(1 - r)])*0.4*np.pi) / np.sin(0.4*np.pi))
    g_wght = np.min([1, np.sqrt((score_w + np.max([score_l, np.floor(0.5*(score_w - 1))]))/19)])
    
    return r_diff, g_wght

# -----

    
def get_date_weight(w_num, w0=0.5, w_first=24, w_last=42):
    """
    """
    if w_num >= w_last:
        date_wght = 1
    else:
        date_wght = w0 * ((1/w0)**(1/(w_last - w_first)))**(w_num - w_first)
    return date_wght

# -----

def get_ignored_games(df_games):
    """
    """
    df_games = df_games.copy()
    df_games['Is_Ignored'] = 1*(df_games['Is_Blowout'] & (df_games['Team_Rank_Diff'] > 600))
    df_teams = pd.DataFrame(index=df_games['Team_1'].unique())
    df_teams['N_Games'] = df_games.groupby('Team_1')['Is_Ignored'].count().add(df_games.groupby('Team_2')['Is_Ignored'].count(), fill_value=0)
    df_teams['N_Ignored'] = df_games.groupby('Team_1')['Is_Ignored'].sum()
    df_teams['N_Valid'] = df_teams['N_Games'] - df_teams['N_Ignored']
    df_bad_teams = df_teams.loc[(df_teams['N_Valid'] < 5) & (df_teams['N_Ignored'] > 0)]
    for team, rw_team in df_bad_teams.iterrows():
        n_unignore = int(min(5 - rw_team['N_Valid'], rw_team['N_Ignored']))
        idx_ignored = df_games.loc[((df_games['Team_1'] == team) | (df_games['Team_2'] == team)) & (df_games['Is_Ignored'] == 1)].sort_values(by='Team_Rank_Diff').index.tolist()
        for i in range(n_unignore):
            df_games.loc[idx_ignored[i], 'Is_Ignored'] = 0
            
    return df_games['Is_Ignored']
   
# -----

def safe_wma(vals, wghts):
    if vals.shape[0] == 0 or wghts.sum() == 0:
        return np.nan
    else:
        return np.average(vals, weights=wghts) 

# -----
    
def run_ranking_iteration(ratings_start, df_games, return_game_info=False):
    """
    """
    df_games['Rank_1'] = (df_games['Game_Rank_Diff'] + ratings_start.reindex(df_games['Team_2']).values).fillna(-1000)
    df_games['Rank_2'] = ratings_start.reindex(df_games['Team_1']).values - df_games['Game_Rank_Diff']
    df_games['Team_Rank_Diff'] = ratings_start.reindex(df_games['Team_1']).values - ratings_start.reindex(df_games['Team_2']).values
    df_games['Team_Rank_Diff'] = df_games['Team_Rank_Diff'].fillna(1000)
    df_games['Is_Ignored'] = get_ignored_games(df_games)
    df_games.loc[df_games['Is_Ignored']==1, 'Game_Wght'] = 0
    df_games_extended = pd.concat([df_games.rename(columns={'Team_1': 'Team', 'Team_2': 'Opponent', 'Rank_1': 'Rank', 'Score_1': 'Score_T', 'Score_2': 'Score_O'}),
                                   df_games.rename(columns={'Team_2': 'Team', 'Team_1': 'Opponent', 'Rank_2': 'Rank', 'Score_2': 'Score_T', 'Score_1': 'Score_O'})])
    df_games_extended = df_games_extended[['Tournament', 'Date', 'Team', 'Opponent', 'Score_T', 'Score_O', 'Game_Rank_Diff', 'Team_Rank_Diff', 'Game_Wght', 'Is_Ignored', 'Rank']]
    ratings_new = df_games_extended.groupby('Team').apply(lambda x: safe_wma(x['Rank'], x['Game_Wght']))
    if return_game_info:
        dict_games = {k: df for k, df in df_games_extended.groupby('Team')}
        return ratings_new, dict_games
    else:
        return ratings_new
    
# -----

def get_usau_ratings(df_games, rating_start, w0, w_first, w_last):
    """
    """
    df_games[['Game_Rank_Diff', 'Score_Wght']] = df_games[['Score_1', 'Score_2']].apply(lambda x: get_ranking_diff_and_game_weight(x['Score_1'], x['Score_2']), axis=1, result_type='expand')
    df_games['Calendar_Week'] = df_games['Date'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').isocalendar()[1])
    df_games['Date_Wght'] = df_games['Calendar_Week'].apply(lambda x: get_date_weight(x, w0, w_first, w_last))
    df_games['Game_Wght'] = df_games['Score_Wght'] * df_games['Date_Wght']
    df_games['Is_Blowout'] = 1*(df_games['Score_1'] > 2*df_games['Score_2'] + 1)
    #
    n_iter, dist_tol = 1000, 1e-5
    df_ratings_iter = pd.DataFrame(rating_start, index=hf_d.get_teams_in_dataset(df_games), columns=range(n_iter+1))
    for i in range(n_iter):
        df_ratings_iter[i+1] = run_ranking_iteration(df_ratings_iter[i], df_games)
        dist_iter = np.sqrt(((df_ratings_iter[i+1] - df_ratings_iter[i])**2).mean())
        # print('{}/{}, Change: {:.7f}, Rating 1: {:.4f}'.format(i+1, n_iter, dist_iter, df_ratings_iter[i+1].max()))
        if dist_iter < dist_tol:
            n_iter = i + 1
            break
    df_ratings_iter = df_ratings_iter[range(n_iter + 1)]
    _, dict_games = run_ranking_iteration(df_ratings_iter[n_iter], df_games, return_game_info=True)
    ratings_final = df_ratings_iter[n_iter].sort_values(ascending=False)
    
    return ratings_final