#Start with a MA.
import pandas as pd
from datamanagement import *
import numpy as np
import math
from datetime import date

Start = '2019-01-01'
End = date.today().strftime("%Y-%m-%d")
number_of_iter = 50

prices, asset_classes, asset = datamanagement_1(Start, End)
ret = data_management_2(prices, asset_classes, asset)

def calculate_rolling_average(ret, window):
    rolling_df = pd.DataFrame()
    for column in ret.columns:
        rolling_df[column] = ret[column].rolling(window=window).mean()
    return rolling_df



# Now we need to use this to determine what assets to hold. So for each month, we need to know if the asset is trending.
# To do this, I think for each day, we can have a dummy, 1 for above sma and 2 for below sma.

def dummy_sma(rolling_df, ret):
    dummy_L_df = pd.DataFrame(index=rolling_df.index)
    dummy_LS_df = pd.DataFrame(index=rolling_df.index)
    for asset_name in rolling_df.columns:
    # Skip non-numeric columns
        if not np.issubdtype(rolling_df[asset_name].dtype, np.number):
            continue
            
        # Compare the prices of the asset for each date
        dummy_L_df[asset_name] = (rolling_df[asset_name] < ret[asset_name]).astype(int)
        dummy_LS_df[asset_name] = np.where(rolling_df[asset_name] < ret[asset_name], 0, -1)
    dummy_L_df  = dummy_L_df.resample('M').mean()
    dummy_LS_df = dummy_LS_df.resample('M').mean()
    return dummy_L_df, dummy_LS_df

rolling_df = calculate_rolling_average(ret, min(200, len(ret)))
dummy_L_df, dummy_LS_df = dummy_sma(rolling_df, ret)
 
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
    print(rsi_dfs)

    return rsi_dfs

rsi_df = calculate_monthly_rsi(ret)


# Now, if the row for a specific contract is <0, then we can exclude it from our sample set, and it is not needed. This is part of the asset selection component.