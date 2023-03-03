import pandas as pd
import datetime
import yfinance as yf
import numpy as np 
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from calendar import monthrange
from dateutil.relativedelta import relativedelta
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from scipy.optimize import minimize
from Trend_Following import dummy_L_df, ret, start, end, dummy_LS_df
warnings.filterwarnings("ignore")


# Load portfolio returns data into a D
#from datamanagement import excel_download, datamanagement_1, data_management_2

# Date range

counter = 4

Start = start
End = end

date1 = datetime.strptime(Start, "%Y-%m-%d")
date2 = datetime.strptime(End, "%Y-%m-%d")
diff = relativedelta(date2, date1)

Start_bench = date1 + relativedelta(months=1)

months_between = (diff.years)*12 + diff.months + 1
# Tickers of assets

Model='Classic' # Could be Classic (historical), BL (Black Litterman) or FM (Factor Model)
Rm = 'MV' # Risk measure used, this time will be variance
Obj = 'Sharpe' # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
Hist = True # Use historical scenarios for risk measures that depend on scenarios
Rf = 0.04 # Risk free rate
L = 1 # Risk aversion factor, only useful when obj is 'Utility'
Points = 50 # Number of points of the frontier
method_mu ='hist' # Method to estimate expected returns based on historical data.
method_cov ='hist' # Method to estimate covariance matrix based on historical data.

############################################################
# Calculate assets returns
############################################################

def optimize_risk_parity(Y, Ycov, counter, i):
    n = Y.shape[1]
    # Define the risk contribution as a constraint
    def risk_contribution(w):
        sigma = np.sqrt(np.matmul(np.matmul(w, Ycov), w))
        return (np.matmul(w, Ycov) * w) / sigma
    # Define the optimization objective
    def objective(w):
        return -np.sum(w * Y.mean())
    # Define the optimization constraints
    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'ineq', 'fun': lambda w: np.sum(risk_contribution(w)) - (1/n)},
            {'type': 'ineq', 'fun': lambda w: np.sum(w[:counter] > 0.05) -counter},
            {'type': 'ineq', 'fun': lambda w: 3 - np.sum(w > 0)},
            ]
    bounds = [(0, 1) for i in range(n)]
    # Call the optimization solver
    res = minimize(objective, np.ones(n)/n, constraints=cons, bounds=bounds, method='SLSQP',
                   options={'disp': False, 'eps': 1e-12, 'maxiter': 10000})
    print(res.message)

    print(res.success)
    return res.x

############################################################
# Monte carlo
############################################################

def monte_carlo(Y):
    log_return = np.log(Y/Y.shift(1))
    sample = Y.shape[0]
    num_ports = 1000
    all_weights = np.zeros((num_ports, len(Y.columns)))
    ret_arr = np.zeros(num_ports)
    vol_arr = np.zeros(num_ports)
    sharpe_arr = np.zeros(num_ports)
    """
    next step, allow short selling, so that the sum of weights must be [-1,1]

    To do this, I suspect I will need to impliment the short trend, whereby if the asset is in a short trend, then we can ONLY sell it.
    So, I will need to set up a new asset returns (Y) containing assets that can be both long and short, and another df outlining if they are long or short.

    """

    for ind in range(num_ports): 
        # weights 
        weights = np.random.dirichlet(np.ones(len(Y.columns)), size=1)
        weights = np.squeeze(weights)
        
        # Enforce minimum weight
        weights = np.maximum(weights, 0.05)
        weights = weights/np.sum(weights)
        all_weights[ind,:] = weights
        
        # expected return 
        ret_arr[ind] = np.sum((log_return.mean()*weights)*sample)

        # expected volatility 
        vol_arr[ind] = np.sqrt(np.dot(weights.T,np.dot(log_return.cov()*sample, weights)))

        # Sharpe Ratio 
        sharpe_arr[ind] = ret_arr[ind]/vol_arr[ind]
    max_sh = sharpe_arr.argmax()
    #plot_frontier(vol_arr,ret_arr,sharpe_arr)

    #To-do:
    #enable short selling
    #enable leverage
    #print(Y.columns)
    #print(all_weights[max_sh,:])
    return all_weights[max_sh,:]

def monte_carlo_SL(Y, short_df):
    log_return = np.log(Y/Y.shift(1))
    sample = Y.shape[0]
    num_ports = 1000
    all_weights = np.zeros((num_ports, len(Y.columns)))
    ret_arr = np.zeros(num_ports)
    vol_arr = np.zeros(num_ports)
    sharpe_arr = np.zeros(num_ports)

    for ind in range(num_ports): 
        # weights 
        weights = np.array(np.random.random(len(Y.columns))) 

        # set maximum weight of 1 for assets with value of 1 or greater in short_df
        mask = short_df >= 1
        weights[mask] = weights[mask] / np.sum(weights[mask])
        weights[~mask] = weights[~mask] / np.sum(np.abs(weights[~mask]))

        # save the weights
        all_weights[ind,:] = weights

        # expected return 
        ret_arr[ind] = np.sum((log_return.mean()*weights)*sample)

        # expected volatility 
        vol_arr[ind] = np.sqrt(np.dot(weights.T,np.dot(log_return.cov()*sample, weights)))

        # Sharpe Ratio 
        sharpe_arr[ind] = ret_arr[ind]/vol_arr[ind]

    max_sh = sharpe_arr.argmax()

    return all_weights[max_sh,:]

############################################################

def plot_frontier(vol_arr,ret_arr,sharpe_arr):
    plt.figure(figsize=(12,8))
    plt.scatter(vol_arr,ret_arr,c=sharpe_arr,cmap='plasma')
    plt.colorbar(label='Sharpe Ratio')
    max_sr_ret = ret_arr[sharpe_arr.argmax()]
    max_sr_vol = vol_arr[sharpe_arr.argmax()]
    max_sr_sr  = sharpe_arr[sharpe_arr.argmax()]
    print("Max values", max_sr_ret,max_sr_vol, max_sr_sr, "Max possible Sharpe:", max(sharpe_arr))
    # plot the dataplt.figure(figsize=(12,8))
    plt.xlabel('Volatility')
    plt.ylabel('Return')

    # add a red dot for max_sr_vol & max_sr_ret
    plt.scatter(max_sr_vol, max_sr_ret, c='red', s=50, edgecolors='black')
    #plt.scatter(max_sr_sr, max(vol_arr), c='green', s=50, edgecolors='black')

############################################################
# Building a loop that estimate optimal portfolios on
# rebalancing dates
############################################################

rng_start = pd.date_range(start, periods=months_between, freq='MS')

def next_month(i):
    #next_i = i + pd.Timedelta(days=31)
    i_str = i.strftime('%Y-%m')
    dt = datetime.strptime(i_str, '%Y-%m')
    next_month = dt + relativedelta(months=1)
    next_i = datetime(next_month.year, next_month.month, 1)
    next_b = pd.date_range(start=next_i, periods=1, freq='M')
    next_b = next_b[0]
    return next_i,next_b

############################################################
# Setting up empty DFs
############################################################

merged_df = pd.DataFrame([])

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

############################################################
# Backtesting
############################################################
def backtest(rng_start, ret, ret_pct, dummy_L_df, dummy_LS_df, ls):
    y_next = pd.DataFrame([])
    portfolio_return_concat = pd.DataFrame([])
    portfolio_return  = pd.DataFrame([])
    weight_concat = pd.DataFrame([])    
    for i in rng_start:
        rng_end = pd.date_range(i, periods=1, freq='M')
        for b in rng_end:
            # cleanup here
            if rng_start[-1] == i:
                Y = ret[i:b]
                Y_adjusted = asset_trimmer(b, dummy_L_df, Y)
                w = monte_carlo(Y_adjusted)
                weight_concat = weightings(w, Y_adjusted, i, weight_concat)
            else:
                if ls == 0:
                    Y_LS = ret[i:b]
                    Y_adjusted_LS = asset_trimmer_LS(b, dummy_LS_df, Y_LS)
                    if not Y_adjusted_LS.empty:
                        w_SL = monte_carlo_SL(Y_adjusted_LS, dummy_LS_df)
                        next_i,next_b = next_month(i)
                        y_next = ret_pct[next_i:next_b]
                        Y_adjusted_next_SL = asset_trimmer_LS(b, dummy_LS_df, y_next)
                        portfolio_return = portfolio_returns(w_SL, Y_adjusted_next_SL, b)
                else:
                    Y = ret[i:b]
                    Y_adjusted = asset_trimmer(b, dummy_L_df, Y)
                    if not Y_adjusted.empty:
                        w = monte_carlo(Y_adjusted) #Long
                        next_i,next_b = next_month(i)
                        weight_concat = weightings(w, Y_adjusted, next_i, weight_concat)
                        y_next = ret_pct[next_i:next_b]
                        Y_adjusted_next_L = asset_trimmer(b, dummy_L_df, y_next) #Long
                        portfolio_return = portfolio_returns(w, Y_adjusted_next_L, b) #Long
                portfolio_return_concat = pd.concat([portfolio_return, portfolio_return_concat], axis=0) #Long

    return portfolio_return_concat, weight_concat

def asset_trimmer_LS(b, df_monthly, Y):
        df_split_monthly = df_monthly[b:b]
        print("are we here???")
        cols_to_drop = [col for col in df_split_monthly.columns if (-0.8 < df_split_monthly[col].max() < 0.8)]
        print("Trend DF", df_split_monthly.drop(columns=cols_to_drop))
        Y = Y.drop(columns=cols_to_drop)
        return Y

def asset_trimmer(b, df_monthly, Y):
        df_split_monthly = df_monthly[b:b]
        cols_to_drop = [col for col in df_split_monthly.columns if df_split_monthly[col].max() < 0.8]
        Y = Y.drop(columns=cols_to_drop)
        return Y

def weightings(w, Y_adjusted, i, weight_concat):
    w_df = pd.DataFrame(w).T
    w_df.columns = Y_adjusted.columns
    z = w_df
    w_df['date'] = w_df.index
    w_df['date'] = i
    w_df.set_index('date', inplace=True)
    weight_concat = pd.concat([weight_concat,w_df]).fillna(0)
    return weight_concat

def portfolio_returns(w, Y_adjusted_next, b):
    df_daily_return = w.T*Y_adjusted_next

    df_portfolio_return = pd.DataFrame(df_daily_return.sum(axis=1), columns=['portfolio_return'])
    
    return df_portfolio_return

def returns_functions():
    print("need to sort this out")

def sentiment_index():
    print("this is for building my sentiment index")
    #I need to build a sentiment index. 2021 was bull market for sure. For Jan 2022, I sold out so that is bear market.
    #For 2023 I think its a sideways market with 4k ES price.x

############################################################
# Correlation matrix
############################################################

def correlation_matrix(sharpe_array):
    corr_matrix = sharpe_array.corr()
    corr_matrix = corr_matrix['Sharpe_ratio']
    plt.figure(figsize=(6, 6))
    sns.heatmap(corr_matrix.to_frame(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

############################################################
# Calling my functions
############################################################
ret_pct = ret.pct_change()

# Need to determine how to merge these 2 dfs

df_dummy_sum = pd.DataFrame()
ls = 1
portfolio_return_concat, weight_concat = backtest(rng_start, ret, ret_pct, dummy_L_df, dummy_LS_df, ls)

print(weight_concat)
weight_concat = weight_concat.drop(index=weight_concat.index[-1])
############################################################
# Portfolio returns
############################################################

#def returns_normalizer(asset):

portfolio_return_concat = pd.DataFrame(pd.DataFrame(portfolio_return_concat))

############################################################
# Spy returns
############################################################

Bench_start = portfolio_return_concat.index.min()
Bench_end   = portfolio_return_concat.index.max()

SPY = yf.download('SPY', start=Bench_start, Bench_end=end)['Adj Close'].pct_change()
SPY = pd.DataFrame(pd.DataFrame(SPY))

merged_df = SPY.merge(portfolio_return_concat, left_index=True, right_index=True)
merged_df.iloc[0] = 0

merged_df = (1 + merged_df).cumprod() * 10000

merged_df = merged_df.rename(columns={'Adj Close': 'SPY_Return'})
print(merged_df)

#Something ain't right with something in the chart
def portfolio_returns_app(returns_df, weights_df):
    # Calculate summary statistics for portfolio returns
    returns = returns_df.pct_change()
    returns.dropna(inplace=True)
    portfolio_mean_returns = returns['portfolio_return'].mean()
    portfolio_std_returns = returns['portfolio_return'].std()
    portfolio_sharpe_ratio = np.sqrt(252) * (portfolio_mean_returns / portfolio_std_returns)
    portfolio_monthly_sharpe_ratio = np.sqrt(12) * (portfolio_mean_returns / portfolio_std_returns)

    # Create a line chart of portfolio and benchmark returns
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=returns_df.index, y=returns_df['portfolio_return'], mode='lines', name='Portfolio Return'))
    fig.add_trace(go.Scatter(x=returns_df.index, y=returns_df['SPY_Return'], mode='lines', name='SPY Returns'))

    # Create a table of summary statistics for portfolio and benchmark returns
    returns_table = html.Table(children=[
            html.Tr(children=[
                html.Th('Statistic'),
                html.Th('Portfolio Returns'),
                html.Th('SPY Returns')
            ]),
            html.Tr(children=[
                html.Td('Mean Returns'),
                html.Td(round(portfolio_mean_returns, 4)),
                html.Td(round(returns['SPY_Return'].mean(), 4))
            ]),
            html.Tr(children=[
                html.Td('Std Returns'),
                html.Td(round(portfolio_std_returns, 4)),
                html.Td(round(returns['SPY_Return'].std(), 4))
            ]),
            html.Tr(children=[
                html.Td('Sharpe Ratio'),
                html.Td('Portfolio: ' + str(round(portfolio_sharpe_ratio, 4))),
                html.Td('SPY: ' + str(round(np.sqrt(252) * (returns['SPY_Return'].mean() / returns['SPY_Return'].std()), 4)))
            ]),
            html.Tr(children=[
                html.Td('Monthly Sharpe Ratio'),
                html.Td('Portfolio: ' + str(round(portfolio_monthly_sharpe_ratio, 4))),
                html.Td('SPY: ' + str(round(np.sqrt(12) * (returns['SPY_Return'].mean() / returns['SPY_Return'].std()), 4)))
            ])
        ])
    
    # Create a table of weights
    weights_table = html.Table(children=[
            html.Tr(children=[
                html.Th('Date'),
                *[html.Th(col) for col in weights_df.columns]
            ]),
            *[html.Tr(children=[
                html.Td(date),
                *[html.Td(round(weights_df.loc[date, col], 4)) for col in weights_df.columns]
            ]) for date in weights_df.index]
        ])
    
    app = dash.Dash(__name__)
    app.layout = html.Div(children=[
        html.H1(children='Portfolio Returns'),

        dcc.Graph(
            id='returns-chart',
            figure=fig
        ),

        html.H2(children='Summary Statistics'),

        returns_table,

        html.H2(children='Weights'),

        weights_table
    ])

    # Run the app
    app.run_server(debug=False)

    return app

app = portfolio_returns_app(merged_df, weight_concat)
app.run_server(debug=True)