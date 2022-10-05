"""
Helpfunctions regarding making figures with Plotly.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot
import plotly.express as px


def get_default_layout():
    """
    Helpfunction to specify default layout used in all the figures.
    """
    lay_dict = dict(title_x=0.5)
    return lay_dict


def get_bold(x):
    """
    Display given text as bold by adding HTML tags.
    """
    return '<b>{}</b>'.format(x)


def add_slider_to_fig(fig, data_plot_info, prefix, num_tr_fix=0, fields=['title.text']):
    """
    Helpfunction to make generation of figures with sliders easier.
    data_plot_info -> [n_traces, lbl, {attributes specified in fields input}]
    """
    data_plot_info = [[k[0], k[1], k[2:]] for k in data_plot_info]
    v_list = [[True]*num_tr_fix + [k for l in [[i==j]*dt[0] for i, dt in enumerate(data_plot_info)] for k in l] for j, _ in enumerate(data_plot_info)]
    sld_list = [dict(args=[{'visible': v}, {fld: fld_val for fld, fld_val in zip(fields, fld_val_list)}],
                     label=lbl,
                     method='update') for v, (_, lbl, fld_val_list) in zip(v_list, data_plot_info)]
    sliders = [dict(active=0, currentvalue={'prefix': prefix}, pad={'t': 30}, steps=sld_list)]
    fig.update_layout(sliders=sliders)
    [fig.update_layout({fld.replace('.', '_'): fld_val for fld, fld_val in zip(fields, data_plot_info[0][2])})]


def plot_bar_race_chart(dict_summary, c_plot_list, fl, name):
    """
    Make figure that visualize weekly progress of the rankings / dataset summary.
    """
    # c_plot_list = ['W_Ratio', 'Games', 'W_Ratio', 'Opponent_W_Ratio']
    f_dict = dict(zip(['Games', 'Wins', 'Losses', 'W_Ratio', 'Opponent_W_Ratio'],
                      ['%{x}', '%{x}', '%{x}', '%{x:.3f}', '%{x:.3f}']))
    f_plot_list = [f_dict.get(k, '%{x:.1f}') for k in c_plot_list]
    n_cols = len(c_plot_list)
    n_plot = 20
    y_ax = list(range(n_plot))
    cl = px.colors.qualitative.Alphabet
    n_cl = len(cl)
    teams_color_order = dict_summary.get(list(dict_summary.keys())[-1]).sort_values(by=c_plot_list[0]).index
    df_cl = pd.Series({t: cl[k % n_cl] for k, t in enumerate(teams_color_order)})
    data_plot_info = []
    fig = go.Figure(layout=get_default_layout()).set_subplots(rows=1, cols=n_cols, shared_yaxes=True,
                                                              horizontal_spacing=0.02,
                                                              subplot_titles=[get_bold(k) for k in c_plot_list])
    for i_dt, (dt, df_dt) in enumerate(dict_summary.items()):
        df_plot = df_dt.sort_values(by=c_plot_list[0], ascending=False).iloc[:n_plot].iloc[::-1]
        n_teams, n_games = df_dt.shape[0], int(df_dt['Games'].sum()/2)
        vis = i_dt == 0
        for i_c, (c_plot, f_plot) in enumerate(zip(c_plot_list, f_plot_list)):
            ht = 'Team: {}<br>{}: {}<extra></extra>'.format('%{customdata[0]}', c_plot, f_plot)
            cd = [[k] for k in df_plot.index]
            fig.add_trace(go.Bar(y=y_ax, x=df_plot[c_plot], marker_color=df_cl.loc[df_plot.index], orientation='h',
                                 texttemplate=get_bold(f_plot), textposition='inside', textfont_size=15,
                                 hovertemplate=ht, customdata=cd, visible=vis), row=1, col=i_c+1)
        data_plot_info.append([n_cols, dt, get_bold('Data: {}, Date: {}, N Teams: {}, N Games: {}'.format(name, dt, n_teams, n_games)),
                               [get_bold('({}) {}'.format(df_plot.shape[0] - i, team)) for i, team in enumerate(df_plot.index)]])
    add_slider_to_fig(fig, data_plot_info, prefix='Date: ', fields=['title.text', 'yaxis.ticktext'])
    fig.update_layout(showlegend=False, yaxis_tickmode='array', yaxis_tickvals=y_ax, yaxis_tickfont_size=15,
                      plot_bgcolor='rgb(255, 255, 255)')
    fig.update_xaxes(visible=False)
    plot(fig, filename=fl)


def get_games_and_score_summary_fig(dataset):
    cl = px.colors.qualitative.Plotly
    games_per_team = pd.Series(0, index=range(1, dataset.summary['Games'].max() + 1))
    games_per_team.update(dataset.summary['Games'].value_counts())
    fig = go.Figure(layout=get_default_layout()).set_subplots(
        rows=2, cols=2, horizontal_spacing=0.04, vertical_spacing=0.07, shared_xaxes=True,
        subplot_titles=['Games per Team', 'Score Analysis (W-Blue, L-Red, Diff-Green)',
                        'Games per Team Cumulative', 'Cumulative Score Analysis (W-Blue, L-Red, Diff-Green)'])
    # games per team
    ht = '%{y} teams (%{customdata:.1f}%) with exactly %{x} games played<extra></extra>'
    fig.add_trace(go.Bar(x=games_per_team.index, y=games_per_team, hovertemplate=ht,
                         customdata=100 * games_per_team / dataset.n_teams), row=1, col=1)
    # games per team cumulative
    games_per_team_cum = (dataset.n_teams - games_per_team.cumsum()).shift(fill_value=dataset.n_teams)
    ht = '%{y} teams (%{customdata:.1f}%) with at least %{x} games played<extra></extra>'
    fig.add_trace(go.Scatter(x=games_per_team_cum.index, y=games_per_team_cum, mode='lines',
                             hovertemplate=ht, customdata=100 * games_per_team_cum / dataset.n_teams), row=2, col=1)
    fig.add_trace(go.Scatter(x=[10], y=[games_per_team_cum[10]], mode='markers', marker_color=cl[1], marker_size=15,
                             hovertemplate=ht, customdata=[100 * games_per_team_cum[10] / dataset.n_teams]), row=2,
                  col=1)
    # score analysis
    df_score = dataset.games[['Score_1', 'Score_2']].copy()
    df_score['Score_Diff'] = df_score['Score_1'] - df_score['Score_2']
    w_score = pd.Series(0, index=range(df_score['Score_1'].max() + 1))
    l_score = pd.Series(0, index=range(df_score['Score_1'].max() + 1))
    diff_score = pd.Series(0, index=range(df_score['Score_1'].max() + 1))
    w_score.update(df_score['Score_1'].value_counts())
    l_score.update(df_score['Score_2'].value_counts())
    diff_score.update(df_score['Score_Diff'].value_counts())
    ht = '%{y} games (%{customdata:.1f}%) with winning score exactly %{x}<extra></extra>'
    fig.add_trace(go.Bar(x=w_score.index, y=w_score, marker_color=cl[0], hovertemplate=ht,
                         customdata=100 * w_score / dataset.n_games), row=1, col=2)
    ht = '%{y} games (%{customdata:.1f}%) with losing score exactly %{x}<extra></extra>'
    fig.add_trace(go.Bar(x=l_score.index, y=l_score, marker_color=cl[1], hovertemplate=ht,
                         customdata=100 * l_score / dataset.n_games), row=1, col=2)
    ht = '%{y} games (%{customdata:.1f}%) with score difference exactly %{x}<extra></extra>'
    fig.add_trace(go.Bar(x=diff_score.index, y=diff_score, marker_color=cl[2], hovertemplate=ht,
                         customdata=100 * diff_score / dataset.n_games), row=1, col=2)
    # cumulative score analysis
    w_score_cum = (dataset.n_games - w_score.cumsum()).shift(fill_value=dataset.n_games)
    l_score_cum = (dataset.n_games - l_score.cumsum()).shift(fill_value=dataset.n_games)
    diff_score_cum = (dataset.n_games - diff_score.cumsum()).shift(fill_value=dataset.n_games)
    ht = '%{y} games (%{customdata:.1f}%) with winning score at least %{x}<extra></extra>'
    fig.add_trace(
        go.Scatter(x=w_score_cum.index, y=w_score_cum, mode='lines+markers', line_color=cl[0], hovertemplate=ht,
                   customdata=100 * w_score_cum / dataset.n_games), row=2, col=2)
    ht = '%{y} games (%{customdata:.1f}%) with losing score at least %{x}<extra></extra>'
    fig.add_trace(
        go.Scatter(x=l_score_cum.index, y=l_score_cum, mode='lines+markers', line_color=cl[1], hovertemplate=ht,
                   customdata=100 * l_score_cum / dataset.n_games), row=2, col=2)
    ht = '%{y} games (%{customdata:.1f}%) with score difference at least %{x}<extra></extra>'
    fig.add_trace(
        go.Scatter(x=diff_score_cum.index, y=diff_score_cum, mode='lines+markers', line_color=cl[2], hovertemplate=ht,
                   customdata=100 * diff_score_cum / dataset.n_games), row=2, col=2)
    #
    fig.update_layout(margin=dict(l=20, r=20, b=10, t=20), showlegend=False)

    return fig


def get_calendar_summary_fig(dataset):
    x_ax = dataset.calendar['Date_End']
    cl = px.colors.qualitative.Plotly
    fig = go.Figure(layout=get_default_layout()).set_subplots(rows=2, cols=2, horizontal_spacing=0.04,
                                                              vertical_spacing=0.07, shared_xaxes=True,
                                                              subplot_titles=['No. Teams', 'No. Games',
                                                                              'Components', 'No. Tournaments'])
    # Teams
    ht = 'No. Teams up to given date: %{y}<extra></extra>'
    fig.add_trace(go.Scatter(x=x_ax, y=dataset.calendar['N_Teams_Cum'], mode='lines+markers', line_color=cl[0],
                             hovertemplate=ht), row=1, col=1)
    ht = 'No. Teams active in given week: %{y}<extra></extra>'
    fig.add_trace(go.Bar(x=x_ax, y=dataset.calendar['N_Teams'], marker_color=cl[1],
                         hovertemplate=ht), row=1, col=1)
    # Games
    ht = 'No. Games up to given date: %{y}<extra></extra>'
    fig.add_trace(go.Scatter(x=x_ax, y=dataset.calendar['N_Games_Cum'], mode='lines+markers', line_color=cl[0],
                             hovertemplate=ht), row=1, col=2)
    ht = 'No. Games in given week: %{y}<extra></extra>'
    fig.add_trace(go.Bar(x=x_ax, y=dataset.calendar['N_Games'], marker_color=cl[1],
                         hovertemplate=ht), row=1, col=2)
    # Components
    ht = 'No. Components (all teams): %{y}<extra></extra>'
    fig.add_trace(go.Scatter(x=x_ax, y=dataset.calendar['N_Components_All'], line_color=cl[0],
                             hovertemplate=ht), row=2, col=1)
    ht = 'Max Component Size: %{y}<extra></extra>'
    fig.add_trace(go.Scatter(x=x_ax, y=dataset.calendar['Max_Component_Size'], line_color=cl[1],
                             hovertemplate=ht), row=2, col=1)
    ht = 'No. Components (teams with some games): %{y}<extra></extra>'
    fig.add_trace(go.Scatter(x=x_ax, y=dataset.calendar['N_Components'], line_color=cl[2],
                             hovertemplate=ht), row=2, col=1)
    # Tournaments
    ht = 'No. Tournaments up to given date: %{y}<extra></extra>'
    fig.add_trace(go.Scatter(x=x_ax, y=dataset.calendar['N_Tournaments_Cum'], mode='lines+markers', line_color=cl[0],
                             hovertemplate=ht), row=2, col=2)
    ht = 'No. Tournaments in given week: %{y}<extra></extra>'
    fig.add_trace(go.Bar(x=x_ax, y=dataset.calendar['N_Tournaments'], marker_color=cl[1],
                         hovertemplate=ht), row=2, col=2)
    fig.update_layout(showlegend=False, margin=dict(l=20, r=20, b=10, t=20), hovermode='x')

    return fig
