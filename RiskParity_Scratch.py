import pandas as pd
import pandas_datareader.data as web
import numpy as np
import datetime
import yfinance as yf
import requests
from pypfopt.efficient_frontier import EfficientFrontier
from scipy.optimize import minimize
TOLERANCE = 1e-10
#To start, we will calculate the risk allocation





################################
#Select the start and end dates:
################################
start = '2022-12-01'
end = '2022-12-31'
################################

def download_data(start, end):
    holdings_url = "https://github.com/ra6it/RiskParity/blob/main/RiskParity_Holdings_Constraints.xlsx?raw=true"
    holdings_url = requests.get(holdings_url).content
    assets = pd.read_excel(holdings_url,'Holdings',usecols="A:B", engine='openpyxl')
    asset_selection = assets['Asset'].tolist()
    print(asset_selection)
    prices = yf.download(asset_selection, start=start, end=end).loc[:, 'Adj Close']
    prices = download_data(asset_selection, start, end)
    prices_pct = prices.pct_change()
    return prices, prices_pct

def covariance_matrix(prices_pct):
    covariances = prices_pct.cov()
    print(covariances)

download_data(start, end)
