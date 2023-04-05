import pandas as pd
import datetime
import yfinance as yf
import numpy as np 
import warnings
import matplotlib.pyplot as plt
from datetime import datetime
from calendar import monthrange
from dateutil.relativedelta import relativedelta
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
#from optimizer import optimizer_backtest
from Trend_Following import ret, Start, End, number_of_iter, asset_classes, rsi_df, dummy_L_df #, rolling_long_df, df_Long_short
warnings.filterwarnings("ignore")
############################################################
# Variables and setup
############################################################


#setup (1 = True):
ls        = 1
monte     = 1
trend     = 'sma'
Rf        = 0.2
benchmark = ['VTI','BND']
Scalar    = 1 #50

date1 = datetime.strptime(Start, "%Y-%m-%d")
date2 = datetime.strptime(End, "%Y-%m-%d")
diff = relativedelta(date2, date1)

Start_bench = date1 + relativedelta(months=1)

months_between = (diff.years)*12 + diff.months + 1

############################################################
# Setting up empty DFs
############################################################

merged_df = sharpe_array = df_dummy_sum = df_dummy_sum =this_month_weight = pd.DataFrame([])

############################################################
# Monte carlo
############################################################

def monte_carlo(Y):
    log_return  = np.log(Y/Y.shift(1))
    sample      = Y.shape[0]
    num_ports   = 5# number_of_iter * Scalar 
    all_weights = np.zeros((num_ports, len(Y.columns)))
    ret_arr     = np.zeros(num_ports)
    vol_arr     = np.zeros(num_ports)
    sharpe_arr  = np.zeros(num_ports)
    for ind in range(num_ports): 
        # weights 
        weights = np.random.dirichlet(np.ones(len(Y.columns)), size=1)
        weights[weights < 0.2] = 0

        weights = np.squeeze(weights)
        weights = weights/np.sum(weights)
        all_weights[ind,:] = weights
        
        # expected return 
        ret_arr[ind] = np.sum((log_return.mean()*weights)*sample)

        # expected volatility 
        vol_arr[ind] = np.sqrt(np.dot(weights.T,np.dot(log_return.cov()*sample, weights)))

        # Sharpe Ratio 
        sharpe_arr[ind] = (ret_arr[ind] - (Rf/12))/vol_arr[ind]
    max_sh = sharpe_arr.argmax()
    #plot_frontier(vol_arr,ret_arr,sharpe_arr)
    sharpe_ratio = (ret_arr[max_sh]- (Rf/12))/vol_arr[max_sh]
    print(all_weights)
    return all_weights[max_sh,:], sharpe_ratio

############################################################

def plot_frontier(vol_arr,ret_arr,sharpe_arr):
    plt.figure(figsize=(12,8))
    plt.scatter(vol_arr,ret_arr,c=sharpe_arr,cmap='plasma')
    plt.colorbar(label='Sharpe Ratio')
    max_sr_ret = ret_arr[sharpe_arr.argmax()]
    max_sr_vol = vol_arr[sharpe_arr.argmax()]
    max_sr_sr  = sharpe_arr[sharpe_arr.argmax()]
    # plot the dataplt.figure(figsize=(12,8))
    plt.xlabel('Volatility')
    plt.ylabel('Return')
    plt.scatter(max_sr_vol, max_sr_ret, c='red', s=50, edgecolors='black')
############################################################
# Building a loop that estimate optimal portfolios on
# rebalancing dates
############################################################

rng_start = pd.date_range(Start, periods=months_between, freq='MS')

def next_month(i):
    i_str = i.strftime('%Y-%m')
    dt = datetime.strptime(i_str, '%Y-%m')
    next_month = dt + relativedelta(months=1)
    next_i = datetime(next_month.year, next_month.month, 1)
    next_b = pd.date_range(start=next_i, periods=1, freq='M')
    next_b = next_b[0]
    return next_i,next_b

############################################################
# Calculate sharpe for next month
############################################################

def next_sharpe(weights, log_return, sharpe_list):
    sample = log_return.shape[0]
    ret_arr2 = np.sum((log_return.mean()*weights)*sample)
    # expected volatility 
    vol_arr2 = np.sqrt(np.dot(weights.T,np.dot(log_return.cov()*sample, weights)))
    sharpe_arr2 = ret_arr2/vol_arr2
    return sharpe_arr2

def forfrontier(arr, i):
    arr = pd.DataFrame(arr).T
    arr['index'] = i
    arr.set_index('index', inplace=True)
    return arr
############################################################
# Backtesting
############################################################
def backtest(rng_start, ret, ret_pct, trend_df):
    print("Iterating: ", number_of_iter * Scalar)
    y_next = portfolio_return_concat = portfolio_return = weight_concat = sharpe_array_concat = pd.DataFrame([])
    for i in rng_start:
        rng_end = pd.date_range(i, periods=1, freq='M')
        for b in rng_end:
            # cleanup here
            if rng_start[-1] == i and prev_i is not None and prev_b is not None:
                print(f"Last month {i}")
                Y = ret[prev_i:prev_b]
                w, sharpe_ratio = monte_carlo(Y_adjusted) #Long

            else:
                Y = ret[i:b]
                Y_adjusted = asset_trimmer(b, trend_df, Y)
                if not Y_adjusted.empty:
                    w, sharpe_ratio = monte_carlo(Y_adjusted) #Long
                    next_i,next_b = next_month(i)
                    weight_concat = weightings(w, Y_adjusted, next_i, weight_concat, sharpe_array_concat, sharpe_ratio)
                    y_next = ret_pct[next_i:next_b]
                    Y_adjusted_next_L = asset_trimmer(b, trend_df, y_next) #Long
                    portfolio_return = portfolio_returns(w, Y_adjusted_next_L) #Long

                prev_i = i
                prev_b = b
                portfolio_return_concat = pd.concat([portfolio_return_concat, portfolio_return], axis=0) #Long
    portfolio_return_concat = pd.DataFrame(portfolio_return_concat)
    return portfolio_return_concat, weight_concat

# Function to drop if the asset is not trending.
def asset_trimmer(b, trend_df, Y):
    df_split_monthly = trend_df[b:b]
    cols_to_drop = [col for col in df_split_monthly.columns if df_split_monthly[col].max() < 0.8]
    Y = Y.drop(columns=cols_to_drop)
    return Y

# Function to manage weights.

def weightings(w, Y_adjusted, i, weight_concat, sharpe_array_concat, sharpe_ratio):
    w_df = pd.DataFrame(w).T
    w_df.columns = Y_adjusted.columns
    w_df['date'] = w_df.index
    w_df['date'] = i
    w_df.set_index('date', inplace=True)
    sharpe_array = w_df
    sharpe_array['sharpe'] = sharpe_ratio
    sharpe_array_concat = pd.concat([sharpe_array_concat, sharpe_array])
    weight_concat = pd.concat([weight_concat,w_df]).fillna(0)
    return weight_concat

# Function to calculate portfolio returns

def portfolio_returns(w, Y_adjusted_next):
    df_daily_return = w.T*Y_adjusted_next
    df_portfolio_return = pd.DataFrame(df_daily_return.sum(axis=1), columns=['portfolio_return'])
    return df_portfolio_return

# Correlation matrix used in the plotly dash.

def correlation_matrix(sharpe_array, column):
    corr_matrix = sharpe_array.corr()
    corr_matrix = corr_matrix[f'{column}']
    return corr_matrix

############################################################
# Calling my functions
############################################################

if   trend == 'rsi':
    rolling_long_df = rsi_df
elif trend == 'sma':
    rolling_long_df = dummy_L_df
# Data management of weights and returns.
portfolio_return_concat, weight_concat = backtest(rng_start, ret, ret.pct_change(), rolling_long_df)

sharpe_array = weight_concat.copy()
weight_concat.drop('sharpe', axis=1, inplace=True)

this_month_weight = weight_concat.iloc[-1]
this_month_weight = pd.DataFrame([this_month_weight])
weight_concat = weight_concat.drop(index=weight_concat.index[-1])
############################################################
# Spy returns & portfolio returns
############################################################

Bench_start = portfolio_return_concat.index.min()

def bench(Bench_start, benchmark):
    Bench_W = Bench = pd.DataFrame([])
    for i in benchmark:
        if i == 'VTI':
            Bench_W = yf.download(i, start=Bench_start, Bench_end=End)['Adj Close'].pct_change() * 0.6
        else:
            Bench_W = yf.download(i, start=Bench_start, Bench_end=End)['Adj Close'].pct_change() * 0.4
        Bench = pd.concat([Bench, Bench_W], axis=1)
    Bench = pd.DataFrame(pd.DataFrame(Bench)).sum(axis=1)
    Bench.iloc[0] = 0
    Bench = (1 + Bench).cumprod() * 10000
    Bench = pd.DataFrame(Bench)
    Bench.columns = ['Bench_Return']

    return Bench

#Bench = pd.DataFrame.set_axis('Bench_Return', axis=1)
Bench = bench(Bench_start, benchmark)
benchmark = 'Bench_Return'

merged_df = portfolio_return_concat
merged_df.iloc[0] = 0
merged_df = (1 + merged_df).cumprod() * 10000
def long_names(asset_classes, weight):
    mapping_dict = dict(zip(asset_classes['Asset'], asset_classes['Full_name']))
    weight_long = weight.rename(columns=mapping_dict)
    return weight_long

# Generate the table of weights
def df_merger(weights_df, weight_long):
    for asset_df, asset_long in zip(weights_df, weight_long):
        column_name = f"{asset_long}"
        weights_df.rename(columns={asset_df: column_name}, inplace=True)
        weight_long.rename(columns={asset_long: column_name}, inplace=True)
    return weights_df, weight_long

def generate_weights_table(weights_df, asset_classes):
    weight_long = long_names(asset_classes, weights_df)
    weights_df2 = weights_df.copy()
    weights_df, weight_long = df_merger(weights_df, weight_long)
    weights_table = html.Table(
        style={'border': '1px solid black', 'padding': '10px'},
        children=[
            # create table header row
            html.Tr(
                style={'background-color': 'grey',                              # Header
                       'color': 'white',
                       'border': '120px solid black',
                       'padding': '120px',
                       'font-family': 'Arial',
                       'font-size': '14px'},
                children=[
                    html.Th('Date:'),
                    *[html.Th(col, style={'text-align': 'center'}) for col in weights_df2.columns]
                ]
            ),
            # create table body rows
            *[html.Tr(
                children=[
                    html.Td(index, style={'font-weight': 'bold',                # Left index
                                          'border': '1px solid black',
                                          'padding': '1px',
                                          'font-family': 'Arial',
                                          'font-size': '14px',}),
                    *[html.Td(round(weights_df.loc[index, col], 4),
                              style={'text-align': 'center',
                                     'border': '1px solid grey',
                                     'padding': '1px',
                                     'font-family': 'Arial',
                                     'font-size': '12px',
                                     'background-color': '#0DBF00' if weights_df.loc[index, col] > 0.5 
                                       else '#9ACD32' if weights_df.loc[index, col] > 0.2 
                                       else '#6FD17A' if weights_df.loc[index, col] > 0.1
                                       else '#D6FF97' if weights_df.loc[index, col] > 0.04
                                       else 'white',
                                       },
                                       title=col,
                                ) for col in weights_df.columns],
                ]
            ) for index in weights_df.index.strftime('%Y-%m-%d')]
        ]
    )
    return weights_table

# Create the plotly dash

def portfolio_data(df, col, num_days, average_number_days):
    Net_Returns = df[f'{col}'].mean()* num_days
    Average_Returns = df[f'{col}'].mean() * average_number_days
    std = df[f'{col}'].std() * average_number_days
    Sharpe_Ratio =  np.sqrt(average_number_days) * (Average_Returns / std)
    return Net_Returns, std, Sharpe_Ratio

def last_month_data(df, col):
    last_month_returns = df.resample('M').mean().iloc[-2]
    last_month_std_returns = df.resample('M').std().iloc[-2]
    last_month_sharpe_ratio = np.sqrt(12) * (last_month_returns / last_month_std_returns)
    last_month_sharpe_ratio = last_month_sharpe_ratio.astype(np.float64).values
    return last_month_sharpe_ratio

def portfolio_returns_app(returns_df, weights_df, this_month_weight, sharpe_array, Bench):
    # Calculate summary statistics for portfolio returns
    num_years = (returns_df.index.max() - returns_df.index.min()).days / 365
    num_days = len(returns_df)
    average_number_days = num_days/num_years
    returns = returns_df.pct_change()
    returns.dropna(inplace=True)
    # Portfolio data:
    Portfolio_Net_Returns, Portfolio_std, Portfolio_Sharpe_Ratio = portfolio_data(returns, 'portfolio_return', num_days, average_number_days)
    last_month_sharpe_ratio = last_month_data(returns, 'portfolio_return')
    # Bench data:
    Bench_Net_Returns, Bench_std, Bench_Sharpe_Ratio = portfolio_data(Bench.pct_change(), f'{benchmark}', num_days, average_number_days)
    last_month_sharpe_ratio_bench = last_month_data(Bench.pct_change(), f'{benchmark}')
 
    # Create a line chart of portfolio and benchmark returns
    fig = go.Figure()

    returns_df = returns_df.sort_index(ascending=False)
    fig.add_trace(go.Scatter(x=returns_df.index, y=returns_df['portfolio_return'], mode='lines', name='Portfolio Return'))
    fig.add_trace(go.Scatter(x=Bench.index, y=Bench[f'{benchmark}'], mode='lines', name=f'{benchmark}'))

    corr_matrix = correlation_matrix(sharpe_array, 'sharpe')
    corr_matrix = corr_matrix.to_frame()
    corr_matrix = corr_matrix.sort_values(by='sharpe', ascending=True)
    corr_matrix_long = long_names(asset_classes, corr_matrix.T).T
    corr_matrix, corr_matrix_long = df_merger(corr_matrix, corr_matrix_long)
    data = [
        go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix_long.index,
            colorscale='RdBu',
            hoverongaps=False,
            hovertemplate='%{y} <br>Correlation: %{z:.2f}<br>',
            showscale=True,
            zmin=-1,
            zmax=1,
            text=corr_matrix.round(2).values.astype(str),
            texttemplate="%{text}",
            textfont={"size":10},
            name = ''
        )
    ]

    layout = go.Layout(
        title='Sharpe ratio correlations',
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(corr_matrix.index))),
            ticktext=corr_matrix.index.to_list()
        )
    )
    # Create a table of summary statistics for portfolio and benchmark returns
    returns_table = html.Table(children=[
            html.Tr(children=[
                html.Th('Statistic'),
                html.Th('Portfolio'),
                html.Th(benchmark)
            ]),
            html.Tr(children=[
                html.Td('Net Returns'),
                html.Td(round(Portfolio_Net_Returns, 4)),
                html.Td(round(Bench_Net_Returns, 4)),
            ]),
            html.Tr(children=[
                html.Td('Avg Yr Returns'),
                html.Td(round(Portfolio_Net_Returns / num_years, 4)),
                html.Td(round(Bench_Net_Returns / num_years, 4))
            ]),
            html.Tr(children=[
                html.Td('Std Returns'),
                html.Td(round(Portfolio_std, 4)),
                html.Td(round(Bench_std, 4))
            ]),
            html.Tr(children=[
                html.Td('Sharpe Ratio'),
                html.Td(str(round(Portfolio_Sharpe_Ratio, 4))),
                html.Td(str(round(Bench_Sharpe_Ratio, 4)))
            ]),
            html.Tr(children=[
                html.Td('L/M sharpe Ratio'),
                html.Td(str(round(float(last_month_sharpe_ratio), 4))),
                html.Td(str(round(float(last_month_sharpe_ratio_bench), 4)))
            ])
        ])
       
    app = dash.Dash(__name__)

    app.layout = html.Div(children=[
    html.H1(children='Portfolio Returns'),

    dcc.Graph(
        id='returns-chart',
        figure=fig
    ),
    html.H2(children='Weights'),

    generate_weights_table(weights_df, asset_classes),

    html.H2(children="Next Month Weights"),
    
    generate_weights_table(this_month_weight, asset_classes),

    html.H2(children='Summary Statistics', style={'font-size': '24px'}),
    returns_table,
    #I would like the index to be the ticker, and the hover on the chart to be the full asset name, it would also be nice in the weights table.
    html.H2(children='Correlation Matrix'),
    dcc.Graph(id='correlation-matrix', figure={'data': data, 'layout':layout},
            style={'width': '40vh',
                   'height': '90vh',
                   'font-family': 'Arial',
                   'font-size': '12px',}
    )
    ])
    
    return app





app = portfolio_returns_app(merged_df, weight_concat, this_month_weight, sharpe_array, Bench)
app.run_server(debug=False)

'''
Next steps:
-More assets enabled.
-More asset selection culling.
-Incorporate the capm model for each assets expected returns.

-Incorporate a rally pivot concept, whereby it will pivot out of an asset if it its recent prices of the month are poor, e.g., a pivot.
-It is a bubble indicator.

New project:
-For each month, rate us on how well we selected assets based on the next months weightings, if the weightings are within a bounds then we are ok, if they are 
    below the previous month then take note that we were in too deep with this asset class, so next time we think of re-balancing by increasing this asset, we can essentially rate our scores.

###############

Do a manual back test, checking each month 1 by 1 and see if the previous month is similar on another run, so increment the months by one to check.
    
############### 
    '''