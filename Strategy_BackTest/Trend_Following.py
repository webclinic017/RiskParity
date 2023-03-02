#Start with a MA.
import pandas as pd
from datamanagement import *
import numpy as np

start = '2019-01-01'
end = '2022-12-31'

prices, asset_classes, asset = datamanagement_1(start, end)
ret = data_management_2(prices, asset_classes, asset)

def calculate_rolling_average(ret, window):
    """
    Calculates the rolling average of a DataFrame with a given window size.

    Args:
        df (pandas.DataFrame): The DataFrame to calculate the rolling average for.
        window (int): The size of the window to use for the rolling average calculation.

    Returns:
        pandas.DataFrame: A new DataFrame with the rolling averages for each column.
    """
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
        dummy_LS_df[asset_name] = np.where(rolling_df[asset_name] < ret[asset_name], 1, -1)
    dummy_L_df  = dummy_L_df.resample('M').mean()
    dummy_LS_df = dummy_LS_df.resample('M').mean()
    print(dummy_L_df)
    print(dummy_LS_df)
    return dummy_L_df, dummy_LS_df

rolling_df = calculate_rolling_average(ret, min(200, len(ret)))
dummy_L_df, dummy_LS_df = dummy_sma(rolling_df, ret)
 
# Now, if the row for a specific contract is <0, then we can exclude it from our sample set, and it is not needed. This is part of the asset selection component.