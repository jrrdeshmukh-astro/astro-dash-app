# -*- coding: utf-8 -*-
import json
import base64
import datetime
from astropy.table import Table 
import numpy as np
import astropy.units as u
import thejoker as tj
from astropy.time import Time
import math
import pandas as pd
import flask
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.plotly as py
import plotly.graph_objs as go

from dash.dependencies import Input, Output, State
from plotly import tools


app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}],
)
app.title = "R147 Catalog"

server = app.server

# Currency pairs

DATA_PATH = 'data'

# Loading historical tick data

# Currency pairs

data = Table.read('r147catalog.rtf', format="ascii.cds")
data_df = pd.DataFrame()
data_df['Teff'] = pd.to_numeric(pd.Series(data['Teff']), errors='coerce')
data_df['eJ-K'] = pd.to_numeric(pd.Series(data['eJ-K']), errors='coerce')
data_df['Kmag'] = pd.to_numeric(pd.Series(data['Kmag']), errors='coerce')
data_df['Jmag'] = pd.to_numeric(pd.Series(data['Jmag']), errors='coerce')
data_df['Mass'] = pd.to_numeric(pd.Series(data['Mass']), errors='coerce')
data_df['pmRA'] = pd.to_numeric(pd.Series(data['pmRA']), errors='coerce')
data_df['pmDE'] = pd.to_numeric(pd.Series(data['pmDE']), errors='coerce')
data_df['RAdeg'] = pd.to_numeric(pd.Series(data['RAdeg']), errors='coerce')
data_df['DEdeg'] = pd.to_numeric(pd.Series(data['DEdeg']), errors='coerce')
data_df = data_df.dropna()
data_df['Photo-Binary'] = pd.Series(data['Photo-Binary'])
data_df['Wide-Binary'] = pd.Series(data['Wide-Binary'])
data_df['Spec-Binary'] = pd.Series(data['Spec-Binary'])

data_df['NOMAD'] = pd.Series(data['NOMAD'])
data_df = data_df[data_df['eJ-K']!=-999]
data_df = data_df[data_df['NOMAD']!='...']
data_df = data_df[data_df['Teff']>0]
rv_df = pd.read_csv('R147-RVs_from_TRES-2021July14.txt')
cols = list(rv_df.columns)
cols_1=[]
for col in cols:
    cols_1.append(col.replace('.','_').strip())

rv_df.columns = cols_1
rv_df = rv_df.drop('Candidates_CandName', axis=1)
rv_df = rv_df.drop('Exposures_SpecRes', axis=1)
rv_df.columns = cols_1[2:]
rv_df = rv_df.reset_index()
rv_df['index'] = 'NID 0' + rv_df['index'].str.replace('m','-')
# API Requests for news div

data_not_available = ['739m0787796', '739m0791953','745m0797925']


more_stars = ['726m1072568', '735m0805329', 
'738m0796217','728m1067260','735m0805486',
'738m0796395','730m0984375','735m0805566',
'738m0796593','731m0918987','735m0805730',
'738m0797354','731m0921419','735m0805823',
'738m0798147','731m0922128','735m0806221',
'731m0926114','736m0792228','739m0787930',
'732m0862104','736m0792853','739m0789763',
'732m0870794','737m0793390','739m0790113',
'733m0833927','737m0796108','739m0791715']

stars = pd.Series(['733m0837663','737m0796188',
'733m0838535','737m0796189','740m0785901',
'734m0819782','737m0796287','740m0786197',
'734m0820974','737m0796524','742m0809706',
'734m0821086','737m0797111',
'734m0821900','737m0798739','746m0784682',
'734m0822598','738m0795743','746m0785235',
'735m0804892','738m0796054','746m0790867',
])

def format_columns(column):
    format_column = column[(column!='') & (column!=None)]
    return format_column

rv_data = pd.DataFrame()
for star in stars:
    rv = pd.read_csv('Ruprecht/'+ star+ '.vzero.txt', header=None)
    rv = rv[0].str.split('  ',expand=True)
    rv_2 = pd.DataFrame()
    rv_1 = rv.transpose()
    for column in rv_1.columns:
        rv_2[column]=(format_columns(rv_1[column]).dropna()).reset_index(drop=True)
    rv_2 = rv_2.transpose().astype('float')
    rv_2.columns = ['BJD','RV','ERR']
    rv_2['index'] = 'NID 0' + star.replace('m','-')
    rv_data = rv_data.append(rv_2)


stars = 'NID 0' + stars.str.replace('m','-')
currencies =  stars.unique().tolist()

# API Requests for news div


# API Call to update news
def update_news():
    json_data = news_requests.json()["articles"]
    df = pd.DataFrame(json_data)
    df = pd.DataFrame(df[["title", "url"]])
    max_rows = 10
    return html.Div(
        children=[
            html.P(className="p-news", children="Headlines"),
            html.P(
                className="p-news float-right",
                children="Last update : "
                + datetime.datetime.now().strftime("%H:%M:%S"),
            ),
            html.Table(
                className="table-news",
                children=[
                    html.Tr(
                        children=[
                            html.Td(
                                children=[
                                    html.A(
                                        className="td-link",
                                        children=df.iloc[i]["title"],
                                        href=df.iloc[i]["url"],
                                        target="_blank",
                                    )
                                ]
                            )
                        ]
                    )
                    for i in range(min(len(df), max_rows))
                ],
            ),
        ]
    )


# Returns dataset for currency pair with nearest datetime to current time

def star_spec(star):
    df_row = data_df[data_df['NOMAD']==star].reset_index(drop=True).loc[0].to_dict()
    return [data_df[data_df['NOMAD']==star].index ,df_row]


def first_ask_bid(currency_pair, t):
    t = t.replace(year=2016, month=1, day=5)
    items = currency_pair_data[currency_pair]
    dates = items.index.to_pydatetime()
    index = min(dates, key=lambda x: abs(x - t))
    df_row = items.loc[index]
    int_index = items.index.get_loc(index)
    return [df_row, int_index]  # returns dataset row and index of row


def get_row(data):
    idx = data[0]
    current_row = data[1]
    return html.Div(
        children=[
            # Summary
            html.Div(
                id=str(current_row['NOMAD']) + "summary",
                className="row summary",
                n_clicks=0,
                children=[
                    html.Div(
                        id=str(current_row['NOMAD']) + "row",
                        className="row",
                        children=[
                            html.P(
                                current_row['NOMAD'],  # currency pair name
                                id=str(current_row['NOMAD']),
                                className="three-col",
                            ),
                            html.P(
                                current_row['Teff'].round(2),  # Bid value
                                id=str(current_row['NOMAD']) + "Teff",
                                className="three-col",
                            ),
                            html.P(
                                current_row['Mass'].round(2),  # Ask value
                                id=str(current_row['NOMAD'])+ "Mass",
                                className="three-col",
                            ),
                        
                            
                        ],
                    )
                ],
            ),
            # Contents
            html.Div(
                id=current_row['NOMAD'] + "contents",
                className="row details",
                children=[

                    # Button to display currency pair chart
                    
                    html.Div(
                        className="button-buy-sell-chart",
                        children=[
                            html.Button(
                                id=current_row['NOMAD'] + "Buy",
                                children="Get Orbit",
                                n_clicks=0
                            )
                        ],
                    ),
                    html.Div(
                        className="button-buy-sell-chart-right",
                        children=[
                            html.Button(
                                id=current_row['NOMAD'] + "Button_chart",
                                children="Get Info",
                                n_clicks=0
                                
                            )
                        ],
                    ),
                ],
            ),
        ]
    )


# color of Bid & Ask rates
def get_color(a, b):
    if a == b:
        return "white"
    elif a > b:
        return "#45df7e"
    else:
        return "#da5657"


# Replace ask_bid row for currency pair with colored values
def replace_row(currency_pair, index, bid, ask):
    index = index + 1  # index of new data row
    new_row = (
        currency_pair_data[currency_pair].iloc[index]
        if index != len(currency_pair_data[currency_pair])
        else first_ask_bid(currency_pair, datetime.datetime.now())
    )  # if not the end of the dataset we retrieve next dataset row

    return [
        html.P(
            currency_pair, id=currency_pair, className="three-col"  # currency pair name
        ),
        html.P(
            new_row[1].round(5),  # Bid value
            id=new_row[0] + "bid",
            className="three-col",
            style={"color": get_color(new_row[1], bid)},
        ),
        html.P(
            new_row[2].round(5),  # Ask value
            className="three-col",
            id=new_row[0] + "ask",
            style={"color": get_color(new_row[2], ask)},
        ),
        html.Div(
            index, id=currency_pair + "index", style={"display": "none"}
        ),  # save index in hidden div
    ]


# Display big numbers in readable format
def human_format(num):
    try:
        num = float(num)
        # If value is 0
        if num == 0:
            return 0
        # Else value is a number
        if num < 1000000:
            return num
        magnitude = int(math.log(num, 1000))
        mantissa = str(int(num / (1000 ** magnitude)))
        return mantissa + ["", "K", "M", "G", "T", "P"][magnitude]
    except:
        return num


# Returns Top cell bar for header area
def get_top_bar_cell(cellTitle, cellValue):
    return html.Div(
        className="two-col",
        children=[
            html.P(className="p-top-bar", children=cellTitle),
            html.P(id=cellTitle, className="display-none", children=cellValue),
            html.P(children=human_format(cellValue)),
        ],
    )
    


# Returns HTML Top Bar for app layout
def get_top_bar(
    star
):
    row = star_spec(star)[1]
    return html.Div(children=[
        get_top_bar_cell("NOMAD", row['NOMAD']),
        get_top_bar_cell("Teff", row['Teff']),
        get_top_bar_cell("Photo-Binary", row['Photo-Binary']),
        get_top_bar_cell("Wide-Binary", row['Wide-Binary']),
        get_top_bar_cell("Spec-Binary", row['Spec-Binary']),
        get_top_bar_cell("Mass", row['Mass']),
    ])

def get_orbit_bar(
    star
):
    print('Star', star)
    df_2 = rv_data[rv_data['index']==star]
    t=Time(df_2['BJD'], format='mjd')
    rv=list(df_2['RV'])* u.m/u.s
    err=list(df_2['ERR'])* u.m/u.s
    data = tj.RVData(t=t, rv=rv, rv_err=err)
    prior = tj.JokerPrior.default(P_min=1*u.day, P_max=256*u.day,
                            sigma_K0=30*u.km/u.s,
                            sigma_v=100*u.km/u.s)
    joker = tj.TheJoker(prior)
    prior_samples = prior.sample(size=100_000)
    samples = joker.rejection_sample(data, prior_samples) 
    row = pd.DataFrame(samples.pack()[0])
    row = row.mean()
    return html.Div(children=[
        get_top_bar_cell("NOMAD", star),
        get_top_bar_cell("P", round(row[0],3)),
        get_top_bar_cell("M0", round(row[3],3)),
        get_top_bar_cell("e", round(row[1],3)),
        get_top_bar_cell("Omega", round(row[2],3)),
        
    ])

####### STUDIES TRACES ######

# Moving average
def moving_average_trace(df, fig):
    df2 = df.rolling(window=5).mean()
    trace = go.Scatter(
        x=df2.index, y=df2["close"], mode="lines", showlegend=False, name="MA"
    )
    fig.append_trace(trace, 1, 1)  # plot in first row
    return fig


# Exponential moving average
def e_moving_average_trace(df, fig):
    df2 = df.rolling(window=20).mean()
    trace = go.Scatter(
        x=df2.index, y=df2["close"], mode="lines", showlegend=False, name="EMA"
    )
    fig.append_trace(trace, 1, 1)  # plot in first row
    return fig


# Bollinger Bands
def bollinger_trace(df, fig, window_size=10, num_of_std=5):
    price = df["close"]
    rolling_mean = price.rolling(window=window_size).mean()
    rolling_std = price.rolling(window=window_size).std()
    upper_band = rolling_mean + (rolling_std * num_of_std)
    lower_band = rolling_mean - (rolling_std * num_of_std)

    trace = go.Scatter(
        x=df.index, y=upper_band, mode="lines", showlegend=False, name="BB_upper"
    )

    trace2 = go.Scatter(
        x=df.index, y=rolling_mean, mode="lines", showlegend=False, name="BB_mean"
    )

    trace3 = go.Scatter(
        x=df.index, y=lower_band, mode="lines", showlegend=False, name="BB_lower"
    )

    fig.append_trace(trace, 1, 1)  # plot in first row
    fig.append_trace(trace2, 1, 1)  # plot in first row
    fig.append_trace(trace3, 1, 1)  # plot in first row
    return fig


# Accumulation Distribution
def accumulation_trace(df):
    df["volume"] = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (
        df["high"] - df["low"]
    )
    trace = go.Scatter(
        x=df.index, y=df["volume"], mode="lines", showlegend=False, name="Accumulation"
    )
    return trace


# Commodity Channel Index
def cci_trace(df, ndays=5):
    TP = (df["high"] + df["low"] + df["close"]) / 3
    CCI = pd.Series(
        (TP - TP.rolling(window=10, center=False).mean())
        / (0.015 * TP.rolling(window=10, center=False).std()),
        name="cci",
    )
    trace = go.Scatter(x=df.index, y=CCI, mode="lines", showlegend=False, name="CCI")
    return trace


# Price Rate of Change
def roc_trace(df, ndays=5):
    N = df["close"].diff(ndays)
    D = df["close"].shift(ndays)
    ROC = pd.Series(N / D, name="roc")
    trace = go.Scatter(x=df.index, y=ROC, mode="lines", showlegend=False, name="ROC")
    return trace


# Stochastic oscillator %K
def stoc_trace(df):
    SOk = pd.Series((df["close"] - df["low"]) / (df["high"] - df["low"]), name="SO%k")
    trace = go.Scatter(x=df.index, y=SOk, mode="lines", showlegend=False, name="SO%k")
    return trace


# Momentum
def mom_trace(df, n=5):
    M = pd.Series(df["close"].diff(n), name="Momentum_" + str(n))
    trace = go.Scatter(x=df.index, y=M, mode="lines", showlegend=False, name="MOM")
    return trace


# Pivot points
def pp_trace(df, fig):
    PP = pd.Series((df["high"] + df["low"] + df["close"]) / 3)
    R1 = pd.Series(2 * PP - df["low"])
    S1 = pd.Series(2 * PP - df["high"])
    R2 = pd.Series(PP + df["high"] - df["low"])
    S2 = pd.Series(PP - df["high"] + df["low"])
    R3 = pd.Series(df["high"] + 2 * (PP - df["low"]))
    S3 = pd.Series(df["low"] - 2 * (df["high"] - PP))
    trace = go.Scatter(x=df.index, y=PP, mode="lines", showlegend=False, name="PP")
    trace1 = go.Scatter(x=df.index, y=R1, mode="lines", showlegend=False, name="R1")
    trace2 = go.Scatter(x=df.index, y=S1, mode="lines", showlegend=False, name="S1")
    trace3 = go.Scatter(x=df.index, y=R2, mode="lines", showlegend=False, name="R2")
    trace4 = go.Scatter(x=df.index, y=S2, mode="lines", showlegend=False, name="S2")
    trace5 = go.Scatter(x=df.index, y=R3, mode="lines", showlegend=False, name="R3")
    trace6 = go.Scatter(x=df.index, y=S3, mode="lines", showlegend=False, name="S3")
    fig.append_trace(trace, 1, 1)
    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 1, 1)
    fig.append_trace(trace3, 1, 1)
    fig.append_trace(trace4, 1, 1)
    fig.append_trace(trace5, 1, 1)
    fig.append_trace(trace6, 1, 1)
    return fig


def line_trace_pmra(df):
    trace = go.Scatter(
        x=[df['pmRA']], y=[df['pmDE']], showlegend=False, name="PM"
    )
    #trace.add_scatter(x=data_df['pmRA'], y=data_df["pmDE"], showlegend=False, name="PM_All")
    
    return trace

def line_trace_pos(df):
    trace = go.Scatter(
        x=[df['RAdeg']], y=[df['DEdeg']], showlegend=False, name="PM"
    )
    #trace.add_scatter(x=data_df['pmRA'], y=data_df["pmDE"], showlegend=False, name="PM_All")
    
    return trace

def line_trace_hr(df):
    trace = go.Scatter(
        x=[df['Teff']], y=[df['Jmag']], showlegend=False, name="PM"
    )
    #trace.add_scatter(x=data_df['pmRA'], y=data_df["pmDE"], showlegend=False, name="PM_All")
    
    return trace

# MAIN CHART TRACES (STYLE tab)
def line_trace(df):
    trace = go.Scatter(
        x=df['BJD'], y=df["RV"], mode="markers", showlegend=False, name="RV Curve"
    )
    return trace


def area_trace(df):
    trace = go.Scatter(
        x=df.index, y=df["close"], showlegend=False, fill="toself", name="area"
    )
    return trace


def bar_trace(df):
    return go.Ohlc(
        x=df.index,
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        increasing=dict(line=dict(color="#888888")),
        decreasing=dict(line=dict(color="#888888")),
        showlegend=False,
        name="bar",
    )


def colored_bar_trace(df):
    return go.Ohlc(
        x=df.index,
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        showlegend=False,
        name="colored bar",
    )


def candlestick_trace(df):
    return go.Candlestick(
        x=df.index,
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        increasing=dict(line=dict(color="#00ff00")),
        decreasing=dict(line=dict(color="white")),
        showlegend=False,
        name="candlestick",
    )


# For buy/sell modal
def ask_modal_trace(currency_pair, index):
    df = currency_pair_data[currency_pair].iloc[index - 10 : index]  # returns ten rows
    return go.Scatter(x=df.index, y=df["Ask"], mode="lines", showlegend=False)


# For buy/sell modal
def bid_modal_trace(currency_pair, index):
    df = currency_pair_data[currency_pair].iloc[index - 10 : index]  # returns ten rows
    return go.Scatter(x=df.index, y=df["Bid"], mode="lines", showlegend=False)


# returns modal figure for a currency pair
def get_modal_fig(currency_pair, index):
    fig = tools.make_subplots(
        rows=2, shared_xaxes=True, shared_yaxes=False, cols=1, print_grid=False
    )

    fig.append_trace(ask_modal_trace(currency_pair, index), 1, 1)
    fig.append_trace(bid_modal_trace(currency_pair, index), 2, 1)

    fig["layout"]["autosize"] = True
    fig["layout"]["height"] = 375
    fig["layout"]["margin"] = {"t": 5, "l": 50, "b": 0, "r": 5}
    fig["layout"]["yaxis"]["showgrid"] = True
    fig["layout"]["yaxis"]["gridcolor"] = "#3E3F40"
    fig["layout"]["yaxis"]["gridwidth"] = 1
    fig["layout"].update(paper_bgcolor="#21252C", plot_bgcolor="#21252C")

    return fig


# Returns graph figure
def get_fig(currency_pair):
    # Get OHLC data
    
    pm_df = star_spec(currency_pair)[1]
    rv_df_star = rv_data[rv_data['index']==currency_pair]    
    row = 1  # number of subplots

    
    fig = tools.make_subplots(
        rows=row,
        cols=1,
        shared_xaxes=True,
        shared_yaxes=True,
        print_grid=False,
        vertical_spacing=0.12,
    )
    # Add main trace (style) to figure
    fig.append_trace(line_trace(rv_df_star), 1, 1)
    #fig.append_trace(line_trace_pmra(pm_df), 2, 1)
    
    # Add trace(s) on fig's first row

    # Plot trace on new row


    fig["layout"][
        "uirevision"
    ] = "The User is always right"  # Ensures zoom on graph is the same on update
    fig["layout"]["autosize"] = True
    fig["layout"]["height"] = 400
    fig["layout"]["xaxis"]["rangeslider"]["visible"] = False
    fig["layout"]["yaxis"]["showgrid"] = True
    fig['layout']['yaxis']['title'] = "RV"
    fig['layout']['xaxis']['title'] = "JD"
    fig["layout"]["yaxis"]["gridcolor"] = "#3E3F40"
    fig["layout"]["yaxis"]["gridwidth"] = 1
    fig["layout"].update(paper_bgcolor="#21252C", plot_bgcolor="#21252C")

    return fig


def get_fig_pos(currency_pair):
    # Get OHLC data
    
    pm_df = star_spec(currency_pair)[1]    
    row = 1  # number of subplots

    
    fig = tools.make_subplots(
        rows=row,
        cols=1,
        shared_xaxes=True,
        shared_yaxes=True,
        print_grid=False,
        vertical_spacing=0.12,
    )
    # Add main trace (style) to figure
    fig.add_trace(go.Scatter(
        x=data_df[data_df['NOMAD']!=currency_pair]['RAdeg'], y=data_df[data_df['NOMAD']!=currency_pair]['DEdeg'], mode='markers', showlegend=False, name="PM_All"
    ), 1, 1)
    fig.add_trace(line_trace_pos(pm_df), 1, 1)
    
    
    # Add trace(s) on fig's first row

    # Plot trace on new row


    fig["layout"][
        "uirevision"
    ] = "The User is always right"  # Ensures zoom on graph is the same on update
    fig["layout"]["autosize"] = True
    fig["layout"]["height"] = 400
    fig['layout']['xaxis']['title'] = "RAdeg"
    fig['layout']['yaxis']['title'] = "DEdeg"
    fig["layout"]["xaxis"]["rangeslider"]["visible"] = False
    fig["layout"]["yaxis"]["showgrid"] = True
    fig["layout"]["yaxis"]["gridcolor"] = "#3E3F40"
    fig["layout"]["yaxis"]["gridwidth"] = 1
    fig["layout"].update(paper_bgcolor="#21252C", plot_bgcolor="#21252C")

    return fig


def get_fig_hr(currency_pair):
    # Get OHLC data
    
    pm_df = star_spec(currency_pair)[1]    
    row = 1  # number of subplots

    
    fig = tools.make_subplots(
        rows=row,
        cols=1,
        shared_xaxes=True,
        shared_yaxes=True,
        print_grid=False,
        vertical_spacing=0.12,
    )
    # Add main trace (style) to figure
    fig.add_trace(go.Scatter(
        x=data_df[data_df['NOMAD']!=currency_pair]['Teff'], y=data_df[data_df['NOMAD']!=currency_pair]['Jmag'], mode='markers', showlegend=False, name="PM_All"
    ), 1, 1)
    fig.add_trace(line_trace_hr(pm_df), 1, 1)
    
    
    # Add trace(s) on fig's first row

    # Plot trace on new row


    fig["layout"][
        "uirevision"
    ] = "The User is always right"  # Ensures zoom on graph is the same on update
    fig["layout"]["autosize"] = True
    fig["layout"]["height"] = 400
    fig["layout"]["xaxis"]["rangeslider"]["visible"] = False
    fig["layout"]["yaxis"]["showgrid"] = True
    fig["layout"]["yaxis"]["gridcolor"] = "#3E3F40"
    fig["layout"]["yaxis"]["gridwidth"] = 1
    fig['layout']['yaxis']['autorange'] = "reversed"
    fig['layout']['yaxis']['title'] = "Jmag"
    fig['layout']['xaxis']['autorange'] = "reversed"
    fig['layout']['xaxis']['title'] = "Teff"
    fig["layout"].update(paper_bgcolor="#21252C", plot_bgcolor="#21252C")

    return fig
    
def get_fig_pm(currency_pair):
    # Get OHLC data
    
    pm_df = star_spec(currency_pair)[1]    
    row = 1  # number of subplots

    
    fig = tools.make_subplots(
        rows=row,
        cols=1,
        shared_xaxes=True,
        shared_yaxes=True,
        print_grid=False,
        vertical_spacing=0.12,
    )
    # Add main trace (style) to figure
    fig.add_trace(go.Scatter(
        x=data_df[data_df['NOMAD']!=currency_pair]['pmRA'], y=data_df[data_df['NOMAD']!=currency_pair]['pmDE'], mode='markers', showlegend=False, name="PM_All"
    ), 1, 1)
    fig.add_trace(line_trace_pmra(pm_df), 1, 1)
    
    
    # Add trace(s) on fig's first row

    # Plot trace on new row


    fig["layout"][
        "uirevision"
    ] = "The User is always right"  # Ensures zoom on graph is the same on update
    fig["layout"]["autosize"] = True
    fig["layout"]["height"] = 400
    fig["layout"]["xaxis"]["rangeslider"]["visible"] = False
    fig["layout"]["yaxis"]["showgrid"] = True
    fig['layout']['xaxis']['title'] = "pmRA"
    fig['layout']['yaxis']['title'] = "pmDE"
    fig["layout"]["yaxis"]["gridcolor"] = "#3E3F40"
    fig["layout"]["yaxis"]["gridwidth"] = 1
    fig["layout"].update(paper_bgcolor="#21252C", plot_bgcolor="#21252C")

    return fig


# returns chart div
def chart_div(pair):
    return html.Div(
        id=pair + "graph_div",
        className="display-none",
        children=[
            # Menu for Currency Graph
            
            # Chart Top Bar
            html.Div(
                className="row chart-top-bar",
                children=[
                    html.Span(
                        id=pair + "menu_button_pos",
                        className="inline-block chart-title",
                        children=f"{pair} Position",
                        n_clicks=0,
                    ),
                    # Dropdown and close button float right
                    
                ],
            ),
                
            # Graph div
            html.Div(
                dcc.Graph(
                    id=pair + "chart-pos",
                    className="chart-graph",
                    config={"displayModeBar": False, "scrollZoom": True},
                )
            ),

            html.Div(
                className="row chart-top-bar",
                children=[
                    html.Span(
                        id=pair + "menu_button_pm",
                        className="inline-block chart-title",
                        children=f"{pair} Proper Motion",
                        n_clicks=0,
                    ),
                    # Dropdown and close button float right
                    
                ],
            ),
                
            # Graph div
            html.Div(
                dcc.Graph(
                    id=pair + "chart-pm",
                    className="chart-graph",
                    config={"displayModeBar": False, "scrollZoom": True},
                )
            ),

            html.Div(
                className="row chart-top-bar",
                children=[
                    html.Span(
                        id=pair + "menu_button_hr",
                        className="inline-block chart-title",
                        children=f"{pair} HR Diagram",
                        n_clicks=0,
                    ),
                    # Dropdown and close button float right
                    
                ],
            ),
                
            # Graph div
            html.Div(
                dcc.Graph(
                    id=pair + "chart-hr",
                    className="chart-graph",
                    config={"displayModeBar": False, "scrollZoom": True},
                )
            ),

            html.Div(
                className="row chart-top-bar",
                children=[
                    html.Span(
                        id=pair + "menu_button",
                        className="inline-block chart-title",
                        children=f"{pair} RV Curve",
                        n_clicks=0,
                    ),
                    # Dropdown and close button float right
                    html.Div(
                        className="graph-top-right inline-block",
                        children=[
                            
                            html.Span(
                                id=pair + "close",
                                className="chart-close inline-block float-right",
                                children="×",
                                n_clicks=0,
                            ),
                        ],
                    ),
                ],
            ),
                
            # Graph div
            html.Div(
                dcc.Graph(
                    id=pair + "chart",
                    className="chart-graph",
                    config={"displayModeBar": False, "scrollZoom": True},
                )
            ),
            
        ],
    )




# returns modal Buy/Sell
def modal(pair):
    return html.Div(
        id=pair + "modal",
        className="modal",
        style={"display": "none"},
        children=[
            html.Div(
                className="modal-content",
                children=[
                    html.Span(
                        id=pair + "closeModal", className="modal-close", children="×"
                    ),
                    html.P(id="modal" + pair, children=pair),
                    # row div with two div
                    html.Div(
                        className="row",
                        children=[
                            # graph div
                            html.Div(
                                className="six columns",
                                children=[
                                    dcc.Graph(
                                        id=pair + "modal_graph",
                                        config={"displayModeBar": False},
                                    )
                                ],
                            ),
                            # order values div
                            html.Div(
                                className="six columns modal-user-control",
                                children=[
                                    html.Div(
                                        children=[
                                            html.P("Volume"),
                                            dcc.Input(
                                                id=pair + "volume",
                                                className="modal-input",
                                                type="number",
                                                value=0.1,
                                                min=0,
                                                step=0.1,
                                            ),
                                        ]
                                    ),
                                    html.Div(
                                        children=[
                                            html.P("Type"),
                                            dcc.RadioItems(
                                                id=pair + "trade_type",
                                                options=[
                                                    {"label": "Buy", "value": "buy"},
                                                    {"label": "Sell", "value": "sell"},
                                                ],
                                                value="buy",
                                                labelStyle={"display": "inline-block"},
                                            ),
                                        ]
                                    ),
                                    html.Div(
                                        children=[
                                            html.P("SL TPS"),
                                            dcc.Input(
                                                id=pair + "SL",
                                                type="number",
                                                min=0,
                                                step=1,
                                            ),
                                        ]
                                    ),
                                    html.Div(
                                        children=[
                                            html.P("TP TPS"),
                                            dcc.Input(
                                                id=pair + "TP",
                                                type="number",
                                                min=0,
                                                step=1,
                                            ),
                                        ]
                                    ),
                                ],
                            ),
                        ],
                    ),
                    html.Div(
                        className="modal-order-btn",
                        children=html.Button(
                            "Order", id=pair + "button_order", n_clicks=0
                        ),
                    ),
                ],
            )
        ],
    )


# Dash App Layout
app.layout = html.Div(
    className="row",
    children=[

        # Left Panel Div
        html.Div(
            className="three columns div-left-panel",
            children=[
                # Div for Left Panel App Info
                html.Div(
                    className="div-info",
                    children=[
                        html.A(
                            html.Img(
                                className="logo",
                                src=app.get_asset_url("dash-logo-new.png"),
                            ),
                            
                        ),  
                        html.H6(className="title-header", children="Astro"),
                        html.H6(className="title-header", children="R147"),
                        dcc.Markdown(
                            """
                            This app queries R147 Catalog and RV data
                            """
                        ),
                    ],
                ),
                # Ask Bid Currency Div
                html.Div(
                    className="div-currency-toggles",
                    children=[
                        html.P(className="three-col", children="NOMAD"),
                        html.P(className="three-col", children="Teff"),
                        html.P(className="three-col", children="Mass"),
                        html.Div(
                            id="pairs",
                            className="div-bid-ask",
                            children=[get_row(star_spec(pair)) for pair in currencies]
                        ),
                        
                        

                        
                
                    ],
                ),
                # Div for News Headlines
                
            ],
        ),
        
        # Right Panel Div
        html.Div(
            className="nine columns div-right-panel",
            children=[
                # Top Bar Div - Displays Balance, Equity, ... , Open P/L
                html.Div(
                    id="top_bar", className="row div-top-bar"
                ),
                html.Div(
                    id="orbit_bar", className="row div-top-bar"
                ),
                # Charts Div
                html.Div(
                    id="charts",
                    className="row",
                    children=[chart_div(pair) for pair in currencies]
                ),
                

                # Panel for orders
                
            ],
        ),
        # Hidden div that stores all clicked charts (EURUSD, USDCHF, etc.)
        html.Div(id="charts_clicked", style={"display": "none"}),
        html.Div(id="orbit_clicked", style={"display": "none"}),
        html.Div(id="currencies", style={"display": "none"}),
        html.Div(id="currencies-prev", style={"display": "none"}),
        # Hidden div for each pair that stores orders
        
    ],
)

# Dynamic Callbacks

# Replace currency pair row
def generate_ask_bid_row_callback(pair):
    def output_callback(n, i, bid, ask):
        return replace_row(pair, int(i), float(bid), float(ask))

    return output_callback


# returns string containing clicked charts
def generate_chart_button_callback():
    def chart_button_callback(*args):
        pairs = ""
        for i in range(len(currencies)):
            if (args[i] % 2) !=0:
                pair = currencies[i]
                if pairs:
                    pairs = pairs + "," + pair
                else:
                    pairs = pair
        return pairs

    return chart_button_callback





# Function to update Graph Figure
def generate_figure_callback(pair):
    def chart_fig_callback(pairs, old_fig):
        if pairs is None:
            return {"layout": {}, "data": {}}

        pairs = pairs.split(",")
        if pair not in pairs:
            return {"layout": {}, "data": []}

        if old_fig is None or old_fig == {"layout": {}, "data": {}}:
            return get_fig(pair)

        fig = get_fig(pair)
        return fig

    return chart_fig_callback

def generate_figure_callback_pm(pair):
    def chart_fig_callback(pairs, old_fig):
        if pairs is None:
            return {"layout": {}, "data": {}}

        pairs = pairs.split(",")
        if pair not in pairs:
            return {"layout": {}, "data": []}

        if old_fig is None or old_fig == {"layout": {}, "data": {}}:
            return get_fig_pm(pair)

        fig = get_fig_pm(pair)
        return fig

    return chart_fig_callback

def generate_figure_callback_hr(pair):
    def chart_fig_callback(pairs, old_fig):
        if pairs is None:
            return {"layout": {}, "data": {}}

        pairs = pairs.split(",")
        if pair not in pairs:
            return {"layout": {}, "data": []}

        if old_fig is None or old_fig == {"layout": {}, "data": {}}:
            return get_fig_hr(pair)

        fig = get_fig_hr(pair)
        return fig

    return chart_fig_callback

def generate_figure_callback_pos(pair):
    def chart_fig_callback(pairs, old_fig):
        if pairs is None:
            return {"layout": {}, "data": {}}

        pairs = pairs.split(",")
        if pair not in pairs:
            return {"layout": {}, "data": []}

        if old_fig is None or old_fig == {"layout": {}, "data": {}}:
            return get_fig_pos(pair)

        fig = get_fig_pos(pair)
        return fig

    return chart_fig_callback


# Function to close currency pair graph
def generate_close_graph_callback():
    def close_callback(n, n2):
        if n == 0:
            if n2 == 1:
                return 1
            return 0
        return 0

    return close_callback


# Function to open or close STYLE or STUDIES menu
def generate_open_close_menu_callback():
    def open_close_menu(n, className):
        if n == 0:
            return "not_visible"
        if className == "visible":
            return "not_visible"
        else:
            return "visible"

    return open_close_menu


# Function for hidden div that stores the last clicked menu tab
# Also updates style and studies menu headers
def generate_active_menu_tab_callback():
    def update_current_tab_name(n_style, n_studies):
        if n_style >= n_studies:
            return "Style", "span-menu selected", "span-menu"
        return "Studies", "span-menu", "span-menu selected"

    return update_current_tab_name


# Function show or hide studies menu for chart
def generate_studies_content_tab_callback():
    def studies_tab(current_tab):
        if current_tab == "Studies":
            return {"display": "block", "textAlign": "left", "marginTop": "30"}
        return {"display": "none"}

    return studies_tab


# Function show or hide style menu for chart
def generate_style_content_tab_callback():
    def style_tab(current_tab):
        if current_tab == "Style":
            return {"display": "block", "textAlign": "left", "marginTop": "30"}
        return {"display": "none"}

    return style_tab


# Open Modal
def generate_modal_open_callback():
    def open_modal(n):
        if n > 0:
            return {"display": "block"}
        else:
            return {"display": "none"}

    return open_modal


# Function to close modal
def generate_modal_close_callback():
    def close_modal(n, n2):
        return 0

    return close_modal


# Function for modal graph - set modal SL value to none
def generate_clean_sl_callback():
    def clean_sl(n):
        return 0

    return clean_sl


# Function for modal graph - set modal SL value to none
def generate_clean_tp_callback():
    def clean_tp(n):
        return 0

    return clean_tp


# Function to create figure for Buy/Sell Modal
def generate_modal_figure_callback(pair):
    def figure_modal(index, n, old_fig):
        if (n == 0 and old_fig is None) or n == 1:
            return get_modal_fig(pair, index)
        return old_fig  # avoid to compute new figure when the modal is hidden

    return figure_modal


# Function updates the pair orders div
def generate_order_button_callback(pair):
    def order_callback(n, vol, type_order, sl, tp, pair_orders, ask, bid):
        if n > 0:
            t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            l = [] if pair_orders is None else json.loads(pair_orders)
            price = bid if type_order == "sell" else ask

            if tp != 0:
                tp = (
                    price + tp * 0.001
                    if tp != 0 and pair[3:] == "JPY"
                    else price + tp * 0.00001
                )

            if sl != 0:
                sl = price - sl * 0.001 if pair[3:] == "JPY" else price + sl * 0.00001

            order = {
                "id": pair + str(len(l)),
                "time": t,
                "type": type_order,
                "volume": vol,
                "symbol": pair,
                "tp": tp,
                "sl": sl,
                "price": price,
                "profit": 0.00,
                "status": "open",
            }
            l.append(order)

            return json.dumps(l)

        return json.dumps([])

    return order_callback


# Function to update orders
def update_orders(orders, current_bids, current_asks, id_to_close):
    for order in orders:
        if order["status"] == "open":
            type_order = order["type"]
            current_bid = current_bids[currencies.index(order["symbol"])]
            current_ask = current_asks[currencies.index(order["symbol"])]

            profit = (
                order["volume"]
                * 100000
                * ((current_bid - order["price"]) / order["price"])
                if type_order == "buy"
                else (
                    order["volume"]
                    * 100000
                    * ((order["price"] - current_ask) / order["price"])
                )
            )

            order["profit"] = "%.2f" % profit
            price = current_bid if order["type"] == "buy" else current_ask

            if order["id"] == id_to_close:
                order["status"] = "closed"
                order["close Time"] = datetime.datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                order["close Price"] = price

            if order["tp"] != 0 and price >= order["tp"]:
                order["status"] = "closed"
                order["close Time"] = datetime.datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                order["close Price"] = price

            if order["sl"] != 0 and order["sl"] >= price:
                order["status"] = "closed"
                order["close Time"] = datetime.datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                order["close Price"] = price
    return orders


# Function to update orders div
def generate_update_orders_div_callback():
    def update_orders_callback(*args):
        orders = []
        current_orders = args[-1]
        close_id = args[-2]
        args = args[:-2]  # contains list of orders for each pair + asks + bids
        len_args = len(args)
        current_bids = args[len_args // 3 : 2 * len_args]
        current_asks = args[2 * len_args // 3 : len_args]
        args = args[: len_args // 3]
        ids = []

        if current_orders is not None:
            orders = json.loads(current_orders)
            for order in orders:
                ids.append(
                    order["id"]  # ids that allready have been added to current orders
                )

        for list_order in args:  # each currency pair has its list of orders
            if list_order != "[]":
                list_order = json.loads(list_order)
                for order in list_order:
                    if order["id"] not in ids:  # only add new orders
                        orders.append(order)
        if len(orders) == 0:
            return None

        # we update status and profit of orders
        orders = update_orders(orders, current_bids, current_asks, close_id)
        return json.dumps(orders)

    return update_orders_callback


# Resize pair div according to the number of charts displayed
def generate_show_hide_graph_div_callback(pair):
    def show_graph_div_callback(charts_clicked):
        if pair not in charts_clicked:
            return "display-none"

        charts_clicked = charts_clicked.split(",")  # [:4] max of 4 graph
        len_list = len(charts_clicked)

        classes = "chart-style"
        if len_list % 2 == 0:
            classes = classes + " six columns"
        elif len_list == 3:
            classes = classes + " four columns"
        else:
            classes = classes + " twelve columns"
        return classes

    return show_graph_div_callback


# Generate Buy/Sell and Chart Buttons for Left Panel
def generate_contents_for_left_panel():
    def show_contents(n_clicks):
        if n_clicks is None:
            return "display-none", "row summary"
        elif n_clicks % 2 == 0:
            return "display-none", "row summary"
        return "row details", "row summary-open"

    return show_contents


# Loop through all currencies

def generate_next_button_callback():
    def next_button_callback(*args):
        num = max(0,args[0]-args[1])
        currencies = data_df['NOMAD'].unique().tolist()[num*10:(num+1)*10]
        return [
        chart_div(pair) for pair in currencies
        ]

    return next_button_callback

def generate_next_button_callback_rows():
    def next_button_callback_rows(*args):
        num = max(0,args[0]-args[1])
        currencies = data_df['NOMAD'].unique().tolist()[num*10:(num+1)*10]
        print(args, currencies)
        result = [html.Button(
                        id="previous",
                        children="Previous",
                        n_clicks=args[1]
                    ),
                    html.Button(
                        id="next",
                        children="Next",
                        n_clicks=args[0]

                    )]
        return [
        get_row(star_spec(pair)) for pair in currencies
        ] + result

    return next_button_callback_rows

"""
def generate_next_button_callback_currencies():
    def next_button_callback_currencies(*args):
        num = max(0,args[0]-args[1])
        currencies = data_df['NOMAD'].unique().tolist()[num*10:(num+1)*10]
        return currencies

    return next_button_callback_currencies

app.callback(
    Output("charts", 'children'),
    [Input("next", "n_clicks"),
    Input("previous", "n_clicks")]
)(generate_next_button_callback())

app.callback(
    Output("currencies", 'children'),
    [Input("next", "n_clicks"),
    Input("previous", "n_clicks")]
)(generate_next_button_callback_currencies())


app.callback(
    Output("pairs", 'children'),
    [Input("next", "n_clicks"),
    Input("previous", "n_clicks")]
)(generate_next_button_callback_rows())

@app.callback(Output("currencies-prev", 'children'),[Input("currencies", "children")])
def update_currencies(currencies):
    globals()['currencies'] = currencies[0]
    return currencies
"""


for pair in currencies:

    # Callback for Buy/Sell and Chart Buttons for Left Panel
    app.callback(
        [Output(pair + "contents", "className"), Output(pair + "summary", "className")],
        [Input(pair + "summary", "n_clicks")],
    )(generate_contents_for_left_panel())
    

    # Callback for className of div for graphs
    app.callback(
        Output(pair + "graph_div", "className"), [Input("charts_clicked", "children")]
    )(generate_show_hide_graph_div_callback(pair))

    # Callback to update the actual graph
    app.callback(
        Output(pair + "chart", "figure"),
        [
            Input("charts_clicked", "children"),
        ],
        [   
            State(pair + "chart", "figure"),
        ],
    )(generate_figure_callback(pair))

    app.callback(
        Output(pair + "chart-pm", "figure"),
        [
            Input("charts_clicked", "children"),
        ],
        [   
            State(pair + "chart-pm", "figure"),
        ],
    )(generate_figure_callback_pm(pair))

    app.callback(
        Output(pair + "chart-hr", "figure"),
        [
            Input("charts_clicked", "children"),
        ],
        [   
            State(pair + "chart-hr", "figure"),
        ],
    )(generate_figure_callback_hr(pair))

    app.callback(
        Output(pair + "chart-pos", "figure"),
        [
            Input("charts_clicked", "children"),
        ],
        [   
            State(pair + "chart-pos", "figure"),
        ],
    )(generate_figure_callback_pos(pair))

    # updates the ask and bid prices
    

    # close graph by setting to 0 n_clicks property
    app.callback(
        Output(pair + "Button_chart", "n_clicks"),
        [Input(pair + "close", "n_clicks")],
        [State(pair + "Button_chart", "n_clicks")],
    )(generate_close_graph_callback())



    # show or hide graph menu


# updates hidden div with all the clicked charts
app.callback(
    Output("charts_clicked", "children"),
    [Input(pair + "Button_chart", "n_clicks") for pair in currencies],
    [State("charts_clicked", "children")],
)(generate_chart_button_callback())


app.callback(
    Output("orbit_clicked", "children"),
    [Input(pair + "Buy", "n_clicks") for pair in currencies],
    [State("orbit_clicked", "children")],
)(generate_chart_button_callback())


@app.callback(Output("orbit_bar", "children"), [Input("orbit_clicked", "children")])
def update_orbit_bar(orbit_clicked):
    orbit_clicked = orbit_clicked.split(",")
    print(orbit_clicked)
    if orbit_clicked[0]=='':
        return []
    print('here')
    return html.Div(children=[get_orbit_bar(oc) for oc in orbit_clicked])

# Callback to update Top Bar values
@app.callback(Output("top_bar", "children"), [Input("charts_clicked", "children")])
def update_top_bar(charts_clicked):
    charts_clicked = charts_clicked.split(",")
    if charts_clicked[0]=='':
        return []
    return html.Div(children=[get_top_bar(chart_clicked) for chart_clicked in charts_clicked])


"""
# Callback to update live clock
@app.callback(Output("live_clock", "children"), [Input("interval", "n_intervals")])
def update_time(n):
    return datetime.datetime.now().strftime("%H:%M:%S")
"""




if __name__ == "__main__":
    """
    df_2 = rv_data[rv_data['index']=='NID 0737-0797111']
    t=Time(df_2['BJD'], format='mjd')
    rv=list(df_2['RV'])* u.m/u.s
    err=list(df_2['ERR'])* u.m/u.s
    data = tj.RVData(t=t, rv=rv, rv_err=err)
    prior = tj.JokerPrior.default(P_min=2*u.day, P_max=512*u.day,
                              sigma_K0=30*u.km/u.s,
                              sigma_v=100*u.km/u.s)
    joker = tj.TheJoker(prior)
    prior_samples = prior.sample(size=100_000)
    samples = joker.rejection_sample(data, prior_samples) 
    row = pd.DataFrame(samples.pack()[0])
    print(row)
    """
    app.run_server(debug=True, host='127.0.0.1', port='8000')