#Start with a MA.
import pandas as pd
from Strategy_BackTest.datamanagement import *

start = '2015-08-01'
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

rolling_df = calculate_rolling_average(ret, min(200, len(ret)))

print(rolling_df)