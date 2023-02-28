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
import qgrid
import plotly.graph_objects as go
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

def excel_download():
    holdings_url = "https://github.com/ra6it/RiskParity/blob/main/RiskParity_Holdings_Constraints.xlsx?raw=true"
    holdings_url = requests.get(holdings_url).content
    assets = pd.read_excel(holdings_url,'Holdings',usecols="A:B", engine='openpyxl')
    assets = assets.reindex(columns=['Asset', 'Industry'])
    asset_classes = {'Asset': assets['Asset'].values.tolist(), 
                     'Industry': assets['Industry'].values.tolist()}
    asset_classes = pd.DataFrame(asset_classes)
    asset_classes = asset_classes.sort_values(by=['Asset'])
    asset = assets['Asset'].values.tolist()
    asset = [x for x in asset if str(x) != 'nan']
    return asset_classes, asset

def datamanagement_1(start, end):
    asset_classes, asset = excel_download()
    df_list = []
    asset = list(set(asset))
    for i in asset:
        asset_2 = yf.download(i, start=start, end=end)['Adj Close']
        df_list.append(pd.DataFrame(asset_2))
    prices = pd.concat(df_list, axis=1)
    prices.columns = asset
    return prices, asset_classes, asset

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

def monte_carlo(Y, df_split_monthly):
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

def data_management_2(prices, asset_classes, asset):
    returns = prices
    valid_assets = asset_classes['Asset'].isin(asset)
    asset_classes = asset_classes[valid_assets]
    asset_classes = pd.DataFrame(asset_classes)
    asset_classes = asset_classes.sort_values(by=['Asset'])
    return returns

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
def backtest(rng_start, ret, ret_pct, sharpe_list, df_monthly):
    wght = pd.DataFrame([])
    x = pd.DataFrame([])
    y_next = pd.DataFrame([])
    merged_array = pd.DataFrame([])

    # Need to set rng_start to the start of the sma df.

    for i in rng_start:
        rng_end = pd.date_range(i, periods=1, freq='M')
        for b in rng_end:
            Y = ret[i:b]
            # If 
            #print("B equals:", b)
            df_split_monthly = df_monthly[b:b]
            # I need to set weight = 0 if a column in df_monthly < 0.8, for example (I could run a monte carlo to get this parameter!!!)

            #Ycov = Y.cov()
            #optimized_weights = optimize_risk_parity(Y, Ycov, counter, i)
            #w = optimized_weights.round(6)
            if rng_start[-1] == i:
                print("last month")
            else:
                w = monte_carlo(Y, df_split_monthly)
                next_i,next_b = next_month(i)
                y_next = ret_pct[next_i:next_b]

                sharpe_array = next_sharpe(w, y_next, sharpe_list)

                #sharpe_df = pd.DataFrame(sharpe_array, columns=['Sharpe_Ratio'])
                weight_printer = pd.DataFrame(w).T
                weight_printer.columns = Y.columns.T
                wgt = pd.concat([weight_printer.T, pd.DataFrame(rng_end, index={"Date"})]).T
                wgt = wgt.set_index(wgt["Date"])
                wght = pd.concat([wght, wgt], axis = 0)
                #Convert the returns and weightings to numpy.
                myreturns = np.dot(w, y_next.T)
                myret = pd.DataFrame(myreturns.T, index = y_next.index)
                #Setting up my correlation matrices
                merged_df = weight_printer
                merged_df['Sharpe_ratio'] = sharpe_array
                merged_df['Date'] = pd.DataFrame(rng_end, index={"Date"})
                merged_df = merged_df.set_index(wgt["Date"])
                merged_df = merged_df.drop('Date', axis = 1)
                merged_array = pd.concat([merged_array, merged_df], axis = 0)
                x = pd.concat([x, myret], axis = 0)
    return wght, x, merged_array

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

def backtest_drop(wght):
    weights = wght.sum(axis=0)
    weights = weights.sort_values(ascending=False)
    top_5_weights = weights.head(5)
    wght_2 = ret[top_5_weights.index]
    return wght_2

sharpe_list = []
#prices, asset_classes, asset = datamanagement_1(start, end)
#ret = data_management_2(prices, asset_classes, asset)
ret_pct = ret.pct_change()
wght, x, sharpe_array = backtest(rng_start, ret, ret_pct, sharpe_list, df_monthly)
#correlation_matrix(sharpe_array)
wght_2 = backtest_drop(wght)
wght.drop(columns=['Date'], axis = 1, inplace = True)
wght.drop(wght.columns[wght.sum() == 0], axis=1, inplace=True)


############################################################
# Portfolio returns
############################################################

def portfolio_returns(x):
    cumret = (1 + x).cumprod() * 10000
    cumret = pd.DataFrame(cumret)
    cumret.columns = ['Returns total']
    return cumret

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
# Spy returns
############################################################

def SPY_ret_2(Start_bench, End):
    SPY_2 = yf.download("SPY", start=Start_bench, end=End)['Adj Close']
    SPY = SPY_2/SPY_2.iloc[0]
    SPY = SPY*10000
    return SPY

def SPY_ret(prices):
    SPY = prices['SPY'].dropna()
    mask = (SPY.index.year == YearNormalize(wght).specific_year) & (SPY.index.month == YearNormalize(wght).specific_month)
    SPY.drop(SPY[mask].index, inplace=True)
    mask_2 = (SPY.index.year == YearNormalize(wght).specific_year) & (SPY.index.month == YearNormalize(wght).specific_month_1)
    SPY.drop(SPY[mask_2].index, inplace=True)
    SPY = SPY/SPY.iloc[0]
    print(SPY)
    SPY.drop(SPY.index[0], inplace=True)
    #SPY.iloc[0,:] = 0
    return SPY*10000

############################################################
# Create 1 df
############################################################

SPY = SPY_ret_2(Start_bench, End)
SPY.columns = ['SPY']
cumret = portfolio_returns(x)

merged_df = pd.merge(SPY, cumret, left_index=True, right_index=True, how='inner')
############################################################
# Plot
############################################################

fig, ax = plt.subplots()
merged_df['Returns total'].plot(ax=ax, label='Portfolio Returns')
SPY.plot(ax=ax, label='SPY')

# Set the x-axis to show monthly ticks
ax.xaxis.set_major_locator(plt.MaxNLocator(months_between/4))

plt.legend()
#plt.show()