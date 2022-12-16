import pandas as pd
import datetime
import yfinance as yf
import backtrader as bt
import numpy as np 
import warnings
import requests
import plotly.graph_objects as go
from calendar import monthrange
warnings.filterwarnings("ignore")

# Date range
Start = '2019-01-01'
End = '2022-12-11'
start = Start
end = End
# Tickers of assets

Model='Classic' # Could be Classic (historical), BL (Black Litterman) or FM (Factor Model)
Rm = 'MV' # Risk measure used, this time will be variance
Obj = 'MaxRet' # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
Hist = True # Use historical scenarios for risk measures that depend on scenarios
Rf = 0.04 # Risk free rate
L = 1 # Risk aversion factor, only useful when obj is 'Utility'
Points = 50 # Number of points of the frontier

def constraints_weightings(constraints,asset_classes):
    asset_classes = pd.DataFrame(asset_classes)
    constraints = pd.DataFrame(constraints)
    data = constraints.fillna("")
    data = data.values.tolist()
    A, B = rp.assets_constraints(constraints, asset_classes)
    return A, B

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
    constraint_url = "https://github.com/ra6it/RiskParity/blob/main/RiskParity_Holdings_Constraints.xlsx?raw=true"
    constraint_url = requests.get(constraint_url).content
    constraints = pd.read_excel(holdings_url,'Constraints',usecols="B:K", engine='openpyxl')
    constraints=pd.DataFrame(constraints)
    return asset_classes, constraints, asset

def runner(asset_classes, constraints, returns):
    method_mu = method_cov = 'hist'
    Port = portfolio_object(asset_classes,method_mu, method_cov, returns)
    A,B = constraints_weightings(constraints,asset_classes)
    w, returns = ainequality(A,B,Port)
    return(Port, w, returns)

def portfolio_object(assets,method_mu, method_cov,returns):
    Port = rp.Portfolio(returns)
    Port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)
    return Port

def ainequality(A,B,Port):
    Port.ainequality = A
    Port.binequality = B
    w = Port.optimization(model=Model, rm=Rm, obj=Obj, rf=Rf, l=L, hist=Hist)
    returns = frontier_create(Port,w)
    return(w, returns)

def frontier_create(Port,w):
    frontier = Port.efficient_frontier(model=Model, rm=Rm, points=Points, rf=Rf, hist=Hist)
    label = 'Max Risk Adjusted Return Portfolio' # Title of point
    mu = Port.mu # Expected returns
    cov = Port.cov # Covariance matrix
    returns = Port.returns # Returns of the assets
    #ax = rp.plot_frontier(w_frontier=frontier, mu=mu, cov=cov, returns=returns, rm=Rm,
                      #rf=Rf, alpha=0.05, cmap='viridis', w=w, label=label,
                      #marker='*', s=16, c='r', height=6, width=10, ax=None)
    return(returns)

asset_classes, constraints, asset = excel_download()

assets = asset
# Downloading data
prices = yf.download(assets, start=start, end=end)
prices = prices.dropna()

prices_2 = prices


############################################################
# Create objects that contain the prices of assets
############################################################

# Creating Assets bt.feeds
assets_prices = []
for i in assets:
    if i != 'SPY':
        prices_ = prices.drop(columns='Adj Close').loc[:, (slice(None), i)].dropna()
        prices_.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
        assets_prices.append(bt.feeds.PandasData(dataname=prices_, plot=False))

# Creating Benchmark bt.feeds        
prices_ = prices.drop(columns='Adj Close').loc[:, (slice(None), 'SPY')].dropna()
prices_.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
benchmark = bt.feeds.PandasData(dataname=prices_, plot=False)

############################################################
# Calculate assets returns
############################################################

#pd.options.display.float_format = '{:.4%}'.format

data = prices.loc[:, ('Adj Close', slice(None))]
data.columns = assets
#data = data.drop(columns=['SPY']).dropna()
returns = data.pct_change().dropna()

############################################################
# Building Constraints
############################################################

asset_classes = pd.DataFrame(asset_classes)
asset_classes = asset_classes.sort_values(by=['Asset'])


constraints = pd.DataFrame(constraints)

############################################################
# Building constraint matrixes for Riskfolio Lib
############################################################

import riskfolio as rp

A,B = constraints_weightings(constraints,asset_classes)

############################################################
# Building a loop that estimate optimal portfolios on
# rebalancing dates
############################################################

rms = ['MV']

rng_start = pd.date_range(start, periods=12, freq='MS')

ret = returns

############################################################
# Setting up empty DFs
############################################################

weights = pd.DataFrame([])
x = pd.DataFrame([])
asset_pr = pd.DataFrame([])
sum_returns = pd.DataFrame([])
we_df = pd.DataFrame([])

for i in rng_start:
    rng_end = pd.date_range(i, periods=1, freq='M')
    for b in rng_end:
        Y = ret[i:b]
        Port, w, returns = runner(asset_classes, constraints, Y)
        re = returns.to_numpy()
        we = w.to_numpy()
        myreturns = re.T * we
        
        myret = pd.DataFrame(myreturns.T,columns = asset, index = Y.index)

        if w is None:
            w = weights.tail(1).T
        weights = pd.concat([weights, w.T], axis = 0)
        x = pd.concat([x, myret], axis = 0)

        adj_close = prices['Adj Close'][i:b].to_numpy()

        price = adj_close.T * we * 10000

        portfolio_price = pd.DataFrame(price.T, columns = asset, index = prices[i:b].index)

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
SPY = SPY['SPY']
SPY = SPY/SPY.iloc[0]*10000

############################################################
# Plot
############################################################

fig = go.Figure()

print("PRINTING FIG")

fig.add_trace(go.Scatter(x=sum_ret.index, y=sum_ret, name='Portfolio Total'))

fig.add_trace(go.Scatter(x=SPY.index , y=SPY, name='Benchmark_Returns'))
