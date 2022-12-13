import pandas as pd
import datetime
import yfinance as yf
import backtrader as bt
import numpy as np
import warnings
import requests
warnings.filterwarnings("ignore")

# Date range
Start = '2020-01-01'
End = '2020-12-31'
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
    method_mu, method_cov = method()
    Port = portfolio_object(asset_classes,method_mu, method_cov, returns)
    A,B = constraints_weightings(constraints,asset_classes)
    w, returns = ainequality(A,B,Port)
    return(Port, w, returns)

def method():
    method_mu='hist' # Method to estimate expected returns based on historical data.
    method_cov='hist' # Method to estimate covariance matrix based on historical data.
    return method_mu, method_cov

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
    ax = rp.plot_frontier(w_frontier=frontier, mu=mu, cov=cov, returns=returns, rm=Rm,
                      rf=Rf, alpha=0.05, cmap='viridis', w=w, label=label,
                      marker='*', s=16, c='r', height=6, width=10, ax=None)
    return(returns)

asset_classes, constraints, asset = excel_download()

assets = asset
# Downloading data
prices = yf.download(assets, start=start, end=end)
prices = prices.dropna()

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
# Building the Buy and Hold strategy
############################################################

class BuyAndHold(bt.Strategy):

    def __init__(self):
        self.counter = 0

    def next(self):
        if self.counter >= 1004:
            if self.getposition(self.data).size == 0:
                self.order_target_percent(self.data, target=0.99)
        self.counter += 1 

############################################################
# Calculate assets returns
############################################################

pd.options.display.float_format = '{:.4%}'.format

data = prices.loc[:, ('Adj Close', slice(None))]
data.columns = assets
#data = data.drop(columns=['SPY']).dropna()
returns = data.pct_change().dropna()

############################################################
# Selecting Dates for Rebalancing
############################################################

# Selecting last day of month of available data
index = returns.groupby([returns.index.year, returns.index.month]).tail(1).index

index_2 = returns.index

# Monthly Dates
index = [x for x in index if float(x.month) % 12.0 == 0 ] 

# Dates where the strategy will be backtested
index_ = [index_2.get_loc(x) for x in index if index_2.get_loc(x) > 0]

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

models = {}

# rms = ['MV', 'MAD', 'MSV', 'FLPM', 'SLPM',
#        'CVaR', 'WR', 'MDD', 'ADD', 'CDaR']

rms = ['MV']

print(index_)

increment = 21

c = 0
for j in rms:
    returns = returns
    weights = pd.DataFrame([])
    for i in index_:
        b = increment + i
        while c < 265:
            Y = returns.iloc[c-b:c,:]
            print(constraints)
            if c >= increment:
                Port, w, returns = runner(asset_classes, constraints, Y)
                if w is None:
                    w = weights.tail(1).T
                weights = pd.concat([weights, w.T], axis = 0)
            c = c + increment

    models[j] = weights.copy()

print(weights)