#Start with a MA.
import pandas as pd
from datamanagement import *
import numpy as np
import math
from datetime import date
import matplotlib.pyplot as plt
Start = '2010-01-01'
End = date.today().strftime("%Y-%m-%d")
number_of_iter = 1
long    = 200
medium  = 100
short   = 30

prices, asset_classes, asset = datamanagement_1(Start, End)
ret = data_management_2(prices, asset_classes, asset)

def calculate_rolling_average(ret, window):
    ret = ret.dropna()
    rolling_df = pd.DataFrame()
    for column in ret.columns:
        rolling_df[column] = ret[column].rolling(window=200).mean()
    rolling_df = dummy_sma(rolling_df, ret, window)
    return rolling_df

# Now we need to use this to determine what assets to hold. So for each month, we need to know if the asset is trending.
# To do this, I think for each day, we can have a dummy, 1 for above sma and 2 for below sma.

def dummy_sma(rolling_df, ret, days):
    dummy_L_df = pd.DataFrame(index=rolling_df.index)
    for asset_name in rolling_df.columns:
    # Skip non-numeric columns
        if not np.issubdtype(rolling_df[asset_name].dtype, np.number):
            continue
        # Compare the prices of the asset for each date
        dummy_L_df[asset_name] = (rolling_df[asset_name] < ret[asset_name]).astype(int)
    #dummy_L_df  = dummy_L_df.resample('M').mean()

    return dummy_L_df

rolling_short_df   = calculate_rolling_average(ret, min(short, len(ret)))
rolling_medium_df  = calculate_rolling_average(ret, min(medium, len(ret)))
rolling_long_df    = calculate_rolling_average(ret, min(long, len(ret)))

df_Long_short = pd.DataFrame([])

for asset_name in rolling_long_df.columns:
    # If P> 200ma, and P < 30ma, then 0 ,1 

    df_Long_short[asset_name] = ((rolling_short_df[asset_name] ==1) & (rolling_long_df[asset_name]==1)).astype(int)

df_Long_short  = df_Long_short.resample('M').mean()

print(df_Long_short)


'''
Do I need a shorter  timeframe?
So if the long term trend is up, and say short term trend is down, then the market has pivoted and we don't want to invest in that asset.

'''

def calculate_monthly_rsi(df):
    # Calculate monthly RSI for each column (i.e., asset)
    rsi_dfs = pd.DataFrame([])
    for col in df.columns:
        # Calculate monthly returns for this asset
        data_monthly = df[[col]].resample('M').last()
        data_monthly['returns'] = data_monthly[col].pct_change()

        # Calculate RSI for each month
        rsis = []
        for i in range(1, len(data_monthly)):
            month_data = data_monthly.iloc[i-1:i+1]
            gain = month_data[month_data['returns'] > 0]['returns'].mean()
            loss = -month_data[month_data['returns'] < 0]['returns'].mean()
            if loss == 0:
                rs = 100
            else:
                rs = gain / loss
            if math.isnan(gain):
                rsi=0
            elif math.isnan(loss):
                rsi=100
            else:
                rsi = 100 - (100 / (1 + rs))
            if math.isnan(rsi):
                print(gain, '/', loss)
            rsis.append(rsi)

        # Create DataFrame with RSI values for each month for this asset
        dates = data_monthly.index[1:]
        rsi_df = pd.DataFrame({'Date': dates, 'RSI': rsis})
        rsi_df.set_index('Date', inplace=True)
        rsi_df = rsi_df.rename(columns={'RSI': col})
        rsi_dfs = pd.concat([rsi_dfs, rsi_df], axis=1, join='outer')
    # Combine RSI DataFrames for all assets into one DataFrame
    return rsi_dfs

def get_market_trend(rsi_df):
    """
    Determines whether the asset is in a bull or bear market based on its RSI values.
    
    Parameters:
    rsi_df (pandas.DataFrame): The RSI DataFrame for an asset.
    
    Returns:
    int: 1 if the asset is in a bull market, 0 if it is in a bear market.
    """
    last_rsi = pd.DataFrame([])
    for asset_name in rsi_df.columns:
        last_rsi[asset_name] = (rsi_df[asset_name] >= 0.5).astype(int)
        
    return last_rsi

rsi_df = calculate_monthly_rsi(ret)
rsi_df_trend = get_market_trend(rsi_df)
# Now, if the row for a specific contract is <0, then we can exclude it from our sample set, and it is not needed. This is part of the asset selection component.

'''
Here, I am going to calculate the derivative of the dummy_long_dfs to see if we can track trends well there.
Is it the derivative of my dummy df, or my trend df?
'''
def deriv(ret):
    # Convert the index to a pandas Timestamp index
    ret.index = pd.to_datetime(ret.index)
    # Calculate derivatives for 1-month, 3-month, and 6-month returns
    returns_derivative_1m = ret.pct_change(1)
    returns_derivative_3m = ret.pct_change(3)
    returns_derivative_6m = ret.pct_change(6)

    # Calculate rolling derivatives for 3-month and 6-month returns
    returns_derivative_3m = returns_derivative_3m.rolling(window=3, min_periods=3).apply(lambda x: x[2]/x[0]-1)
    returns_derivative_6m = returns_derivative_6m.rolling(window=6, min_periods=6).apply(lambda x: x[5]/x[0]-1)
    # Create a new DataFrame to store trend data
    df_trend = pd.DataFrame(index=ret.index, columns=ret.columns)
    # Loop over the index of the df_trend DataFrame
    for idx in df_trend.index:
        # Check if the index is present in both DataFrames
        if idx in returns_derivative_1m.index and idx in returns_derivative_3m.index and idx in returns_derivative_6m.index:
            print("Here")
            for col in returns_derivative_1m.columns:
                print("There")
                print()
                is_increasing = (
                    (returns_derivative_1m[col].loc[idx] > returns_derivative_1m[col].loc[idx-pd.Timedelta(days=30)]) #and 
                    #(returns_derivative_3m[col].loc[idx] > returns_derivative_3m[col].loc[idx-pd.Timedelta(days=90)]) and
                    #(returns_derivative_6m[col].loc[idx] > returns_derivative_6m[col].loc[idx-pd.Timedelta(days=180)]) and
                    #(returns_derivative_1m[col].loc[idx] > returns_derivative_1m[col].loc[idx-pd.Timedelta(days=60)])
                )
                print(returns_derivative_1m[col].loc[idx])
                print(returns_derivative_1m[col].loc[idx-pd.Timedelta(days=30)])
                # If all derivatives are increasing, set the value to 1, otherwise 0
                df_trend.loc[idx, col] = 1 if is_increasing else 0
    print(df_trend)