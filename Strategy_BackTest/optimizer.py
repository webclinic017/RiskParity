import numpy as np
import pandas as pd
from BackTest import months_between, portfolio_returns, asset_trimmer, next_month
from scipy.optimize import minimize
from Trend_Following import Start, End, ret, dummy_L_df
counter = 4

rng_start = pd.date_range(Start, periods=months_between, freq='MS')



Model='Classic' # Could be Classic (historical), BL (Black Litterman) or FM (Factor Model)
Rm = 'MV' # Risk measure used, this time will be variance
Obj = 'Sharpe' # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
Hist = True # Use historical scenarios for risk measures that depend on scenarios
Rf = 0.04 # Risk free rate
L = 1 # Risk aversion factor, only useful when obj is 'Utility'
Points = 50 # Number of points of the frontier
method_mu ='hist' # Method to estimate expected returns based on historical data.
method_cov ='hist' # Method to estimate covariance matrix based on historical data.

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
            ]
    bounds = [(0, 1) for i in range(n)]
    # Call the optimization solver
    res = minimize(objective, np.ones(n)/n, constraints=cons, bounds=bounds, method='SLSQP',
                   options={'disp': False, 'eps': 1e-12, 'maxiter': 10000})
    print(res.message)
    print(res.success)
    return res.x

def optimizer_backtest():
    portfolio_weight_concat = pd.DataFrame([])
    ret_pct = ret.pct_change()
    for i in rng_start:
        rng_end = pd.date_range(i, periods=1, freq='M')
        for b in rng_end:
            next_i,next_b = next_month(i)
            Y = ret_pct[i:b]
            Ycov = Y.cov()
            w = optimize_risk_parity(Y, Ycov, counter, i)
            w_df = pd.DataFrame(data=w.T.reshape(1, -1), columns=Y.columns)
            w_df['date'] = i
            y_next = ret_pct[next_i:next_b]
            print(y_next)
            w_df.set_index('date', inplace=True) #This is the weight for i+1 month, using i month data. 
            Y_adjusted_next_L = asset_trimmer(b, dummy_L_df, y_next) #Long
            print(Y_adjusted_next_L)
            portfolio_return = portfolio_returns(w, Y_adjusted_next_L) #Long
            portfolio_weight_concat = pd.concat([portfolio_weight_concat, w_df], axis=0) #Long
    return portfolio_weight_concat
portfolio_weight_concat = optimizer_backtest()
print(portfolio_weight_concat)
