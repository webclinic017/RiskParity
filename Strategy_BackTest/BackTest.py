import pandas as pd
import datetime
import yfinance as yf
import backtrader as bt
import numpy as np 
import warnings
import riskfolio as rp
import requests
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import timedelta
from calendar import monthrange
from dateutil.relativedelta import relativedelta
from scipy.optimize import minimize
from Trend_Following import df_monthly, ret, start, end
warnings.filterwarnings("ignore")
#from datamanagement import excel_download, datamanagement_1, data_management_2

# Date range

counter = 4

Start = start
End = end

date1 = datetime.datetime.strptime(Start, "%Y-%m-%d")
date2 = datetime.datetime.strptime(End, "%Y-%m-%d")
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
    # Somewhere here I need to set the weights of an asset = 0 if it is not trending.
    log_return = np.log(Y/Y.shift(1))
    sample = Y.shape[0]
    num_ports = 10000
    all_weights = np.zeros((num_ports, len(Y.columns)))
    ret_arr = np.zeros(num_ports)
    vol_arr = np.zeros(num_ports)
    sharpe_arr = np.zeros(num_ports)
    for ind in range(num_ports): 
        # weights 
        weights = np.array(np.random.random(len(Y.columns))) 
        weights = weights/np.sum(weights)  
       
        # save the weights
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
    next_i = i + pd.Timedelta(days=31)
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
def backtest(rng_start, ret, ret_pct, df_monthly):
    y_next = pd.DataFrame([])
    portfolio_return_concat = pd.DataFrame([])
    portfolio_return  = pd.DataFrame([])
    for i in rng_start:
        rng_end = pd.date_range(i, periods=1, freq='M')
        for b in rng_end:
            # I need to set weight = 0 if a column in df_monthly < 0.8, for example (I could run a monte carlo to get this parameter!!!)
            if rng_start[-1] == i:
                print("last month")
            else:
                Y = ret[i:b]
                Y_adjusted = asset_trimmer(b, df_monthly, Y)
                if not Y_adjusted.empty:
                    w = monte_carlo(Y_adjusted)
                    next_i,next_b = next_month(i)
                    y_next = ret_pct[next_i:next_b]
                    Y_adjusted_next = asset_trimmer(b, df_monthly, y_next)
                    portfolio_return = portfolio_returns(w, Y_adjusted_next)
                    #need the concat shit...
                    portfolio_return_concat = pd.concat([portfolio_return, portfolio_return_concat], axis=0)
    return portfolio_return_concat

def asset_trimmer(b, df_monthly, Y):
        df_split_monthly = df_monthly[b:b]
        cols_to_drop = [col for col in df_split_monthly.columns if df_split_monthly[col].max() < 0.3]
        Y = Y.drop(columns=cols_to_drop)
        return Y

def portfolio_returns(w, Y_adjusted_next):
    df_daily_return = w.T*Y_adjusted_next
    df_portfolio_return = pd.DataFrame(df_daily_return.sum(axis=1), columns=['portfolio_return'])\
    
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
print(df_monthly)
portfolio_return_concat = backtest(rng_start, ret, ret_pct, df_monthly)

############################################################
# To normalize the charts to the same dfs.
############################################################
class YearNormalize:
    def __init__(self, wght):
        self.wght = wght
        self.specific_month_1, self.specific_year, self.specific_month = self.year_normalize()

    def year_normalize(self):
        specific_year  = self.wght.index[0].year
        specific_month = self.wght.index[0].month
        if self.wght.index[0].month - 1 == 0:
            specific_month_1  = 12
        else:
            specific_month_1 = self.wght.index[0].month - 1
        specific_month_1 = specific_month_1
        return specific_year, specific_month, specific_month_1

############################################################
# Portfolio returns
############################################################

#def returns_normalizer(asset):


new_date = portfolio_return_concat.index[0] - timedelta(days=1)
new_row = pd.DataFrame({'portfolio_return': [0]}, index=[new_date])
portfolio_return_concat = pd.concat([new_row, portfolio_return_concat]) 
portfolio_return_concat = (1 + portfolio_return_concat['portfolio_return']).cumprod() * 10000
portfolio_return_concat.to_frame()
portfolio_return_concat = pd.DataFrame(pd.DataFrame(portfolio_return_concat))

############################################################
# Spy returns
############################################################

Bench_start = portfolio_return_concat.index.min()
Bench_end   = portfolio_return_concat.index.max()

SPY = yf.download('SPY', start=Bench_start, Bench_end=end)['Adj Close'].pct_change()
SPY = pd.DataFrame(pd.DataFrame(SPY))

merged_df = SPY.merge(portfolio_return_concat, left_index=True, right_index=True)
merged_df.iloc[0, 0] = 0

merged_df['Adj Close'] = (1 + merged_df['Adj Close']).cumprod() * 10000

merged_df = merged_df.rename(columns={'Adj Close': 'SPY_Return'})

#Something ain't right with something in the chart

print(merged_df.to_string())


merged_df.plot(y=['SPY_Return', 'portfolio_return'])

# Set the x-axis to show monthly ticks

plt.show()