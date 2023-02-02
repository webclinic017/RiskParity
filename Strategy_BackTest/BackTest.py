import pandas as pd
import datetime
import yfinance as yf
import backtrader as bt
import numpy as np 
import warnings
import riskfolio as rp
import requests
import matplotlib as plt
import qgrid
import plotly.graph_objects as go
from calendar import monthrange
from dateutil.relativedelta import relativedelta
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

# Date range
Start = '2018-05-01'
End = '2022-06-30'
counter = 8

start = Start
end = End

date1 = datetime.datetime.strptime(Start, "%Y-%m-%d")
date2 = datetime.datetime.strptime(End, "%Y-%m-%d")
diff = relativedelta(date2, date1)

months_between = (diff.years)*12 + diff.months + 1
print(months_between)
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

asset_classes, asset = excel_download()

assets = asset
print(assets)
# Downloading data

################## Here, I need this to be in my backtesting loop, where start and end is called with each month.
df_list = []
for i in assets:
    asset_2 = yf.download(i, start=start, end=end)['Adj Close']
    df_list.append(pd.DataFrame(asset_2))

new_df = pd.concat(df_list, axis=1)
new_df.columns = assets
print(new_df)

prices = new_df

############################################################
# Calculate assets returns
############################################################

def optimize_risk_parity(Y, Ycov, counter):
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
            {'type': 'ineq', 'fun': lambda w: np.amax(w) - 0.8}]
    bounds = [(0, 1) for i in range(n)]
    # Call the optimization solver
    res = minimize(objective, np.ones(n)/n, constraints=cons, bounds=bounds, method='SLSQP',
                   options={'disp': False, 'eps': 1e-12})
    return res.x

data = prices

returns = data.pct_change()

valid_assets = asset_classes['Asset'].isin(asset)

asset_classes = asset_classes[valid_assets]

asset_classes = pd.DataFrame(asset_classes)
asset_classes = asset_classes.sort_values(by=['Asset'])

############################################################
# Building a loop that estimate optimal portfolios on
# rebalancing dates
############################################################

rms = ['MV']

rng_start = pd.date_range(start, periods=months_between, freq='MS')
ret = returns

############################################################
# Setting up empty DFs
############################################################

def next_month(i):
    next_i = i + pd.Timedelta(days=31)
    next_b = pd.date_range(start=next_i, periods=1, freq='M')
    next_b = next_b[0]
    return next_i,next_b

#setting up the empty DFs

weights = pd.DataFrame([])
x = pd.DataFrame([])
asset_pr = pd.DataFrame([])
sum_returns = pd.DataFrame([])
we_df = pd.DataFrame([])
y_next = pd.DataFrame([])
weight_df = pd.DataFrame([])
weighted_df = pd.DataFrame([])
time_df = pd.DataFrame([])
timed_df = pd.DataFrame([])
wght = pd.DataFrame([])

for i in rng_start:
    rng_end = pd.date_range(i, periods=1, freq='M')
    for b in rng_end:
        Z = ret
        Y = ret[i:b]
        Ycov = Y.cov()
        optimized_weights = optimize_risk_parity(Y, Ycov, counter)
        w = optimized_weights.round(6)

        next_i,next_b = next_month(i)
        y_next = Z[next_i:next_b]
        weight_printer = pd.DataFrame(w).T
        weight_printer.columns = Y.columns.T
        wgt = pd.concat([weight_printer.T, pd.DataFrame(rng_end, index={"Date"})]).T
        wgt = wgt.set_index(wgt["Date"])
        wght = pd.concat([wght, wgt], axis = 0)
        #Convert the returns and weightings to numpy.
        myreturns = np.dot(w, y_next.T)
        myret = pd.DataFrame(myreturns.T, index = y_next.index)
        x = pd.concat([x, myret], axis = 0)
        portfolio_price = pd.DataFrame(myret, index = y_next.index)
        asset_pr = pd.concat([asset_pr, portfolio_price], axis = 0)


wght.drop(columns=['Date'], axis = 1, inplace = True)
wght.drop(wght.columns[wght.sum() == 0], axis=1, inplace=True)

print(wght.to_string())
#qgrid_widget = qgrid.show_grid(wght)
#qgrid_widget
############################################################
# Portfolio returns
############################################################

cumret = (1 + x).cumprod() * 10000

cumret.rename(columns={'0': 'Returns Total'}, inplace = True)

cumret = pd.DataFrame(cumret)

############################################################
# Spy returns
############################################################
SPY = prices['SPY'].dropna()
SPY = SPY/SPY.iloc[0]*10000
############################################################
# Plot
############################################################
cumret.plot(label = 'Portfolio Returns')
SPY.plot(label='SPY')

