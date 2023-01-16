import pandas as pd
import datetime
import yfinance as yf
import backtrader as bt
import numpy as np 
import warnings
import riskfolio as rp
import requests
import plotly.graph_objects as go
from calendar import monthrange
from dateutil.relativedelta import relativedelta
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

# Date range
Start = '2022-01-01'
End = '2022-06-30'
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

def optimize_risk_parity(Y, Ycov):
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
            {'type': 'ineq', 'fun': lambda w: np.sum(risk_contribution(w)) - (1/n)}]
    bounds = [(0, 1) for i in range(n)]
    # Call the optimization solver
    res = minimize(objective, np.ones(n)/n, constraints=cons, bounds=bounds, method='SLSQP',
                   options={'disp': False, 'eps': 1e-8})
    return res.x

# Call the optimize function


#pd.options.display.float_format = '{:.4%}'.format

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

weights = pd.DataFrame([])
x = pd.DataFrame([])
asset_pr = pd.DataFrame([])
sum_returns = pd.DataFrame([])
we_df = pd.DataFrame([])
y_next = pd.DataFrame([])
for i in rng_start:
    rng_end = pd.date_range(i, periods=1, freq='M')
    for b in rng_end:
        Z = ret
        Y = ret[i:b]
        Ycov = Y.cov()
        optimized_weights = optimize_risk_parity(Y, Ycov)
        print("Optimized Weights: ", optimized_weights)
        w = optimized_weights
        next_i,next_b = next_month(i)
        y_next = Z[next_i:next_b]
        #Convert the returns and weightings to numpy.
        myreturns = np.dot(w, y_next.T)
        myret = pd.DataFrame(myreturns.T)
        print(myret)
        myret.columns = Y.columns[0]
        if w is None:
            w = weights.tail(1).T
        weights = pd.concat([weights, w.T], axis = 0)
        x = pd.concat([x, myret], axis = 0)

        adj_close = prices['Adj Close'][i:b].to_numpy()

        price = adj_close.T * we * 10000

        portfolio_price = pd.DataFrame(price.T, columns = Y.columns, index = prices[i:b].index)

        asset_pr = pd.concat([asset_pr, portfolio_price], axis = 0)


###
# Setup backtesting #
###
############################################################
# Portfolio returns
############################################################

f = prices['Adj Close']
sum_ret = f.sum(axis=1)
sum_ret = sum_ret/sum_ret.iloc[0]


sum_ret = sum_ret * 10000

############################################################
# Spy returns
############################################################
SPY = prices['Adj Close']
SPY = SPY['VTI'].dropna()
SPY = SPY/SPY.iloc[0]*10000
############################################################
# Plot
############################################################

fig = go.Figure()

print("PRINTING FIG")

fig.add_trace(go.Scatter(x=sum_ret.index, y=sum_ret, name='Portfolio Total'))

fig.add_trace(go.Scatter(x=SPY.index , y=SPY, name='VTI'))
