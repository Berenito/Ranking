# -*- coding: utf-8 -*-
"""
Helpfunctions regarding making figures with Plotly.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot
import plotly.express as px
import os
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

# ----------

def get_default_layout():
    """
    Helpfunction to specify default layout used in all the figures.
    """
    lay_dict = dict(title_x=0.5)
    return lay_dict


# -----

def get_bold(x):
    """
    Display given text as bold by adding HTML tags.
    """
    return '<b>{}</b>'.format(x)


# -----

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


# -----

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


# ----------

def plot_bar_race_fig(dataset, c_plot_list, filename=None, include_weekly=False):
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
    fl = os.path.join(ROOT_DIR, 'figures', 'fig-{}{}.html'.format(
        dataset.name.lower().replace(' ', '-').replace('_', '-'), sfx)) if filename is None else filename
    dict_plot = dataset.weekly_summary if include_weekly else {'All Games': dataset.summary}
    plot_bar_race_chart(dict_plot, c_plot_list, fl, dataset.name)
