'''
This is the file where I rate my selections based on next months allocations.
Essentially, I want to work our an error variable if the new weighting is within say +/-1% next month, then we are ok, 
but if the weighting variable is >5% then we are toodef
but if the weighting variable is <5% then we are tooagg.

could it be a correlation between asset returns and weightings? I think so?

As a result of this we should be able to build a df of each asset and how we rate, adjusting it each month
'''
import pandas as pd
from Trend_Following import ret
from BackTest import weight_concat

ret_pct = ret.pct_change()

ret_pct = ret_pct.resample('M').sum()
ratings_df = pd.DataFrame([])
ratings_df = weight_concat.diff()



column_sum = ratings_df.sum()*100

# Print the result
corr_matrix = column_sum.corr(ret_pct)

print(corr_matrix)

print("stop here")
