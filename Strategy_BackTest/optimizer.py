import numpy as np
import pandas as pd
from BackTest import Y, YCov, i
from scipy.optimize import minimize

counter = 4

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