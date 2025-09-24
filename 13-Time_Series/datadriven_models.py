# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 16:09:22 2025

@author: Ashlesha
"""

# Importing necessary libraries
import pandas as pd                # For data handling
import numpy as np                 # For numerical operations
import matplotlib.pyplot as plt    # For plotting graphs
from statsmodels.tsa.seasonal import seasonal_decompose   # For time series decomposition
from statsmodels.tsa.holtwinters import Holt, ExponentialSmoothing   # For trend/seasonality

# Load the dataset
cocacola = pd.read_excel("c:/Data-Science/10-Time/CocaCola_Sales_Rawdata.xlsx")

# Plot original sales data to visualize trend/seasonality
cocacola.Sales.plot()
'''
Upward Trend

The overall direction of the line is moving upwards.

This means CocaCola sales are increasing over time.

For example, in the early quarters (left side), sales are around 1500-2500,
while by the last quarters (right side), sales are reaching 5000+.

Seasonality / Fluctuations

Even though the general trend is upward, there are regular ups and downs.

This indicates quarterly seasonal patterns (likely linked to consumer demand cycles).

Volatility

The size of fluctuations increases as sales grow.

For example, the dips and peaks in the later quarters are much
larger than in the first 10 quartes

This suggests that while the company grows,it also eperiences
stronger seasonal spikes.

Business Insight

cocacola's sales are consistently growing, which is a good sign of brand
strength and market demand.

The repeated seasonal effect means the company can predict demand better
(e.g., stocking more in high-demand quarters).

Forecasting models like Holt-Winters (which you applied) are appropriate here,
'''
# Splitting dataset into Train and Test
# Data is quarterly → 4 quarters in a year
# Training on first 38 quarters, testing on last 4
Train = cocacola.head(38)
Test = cocacola.tail(4)

#Define mean absolute percentages  Error (MAPE as perform metric

def MAPE(pred,org):
    temp=np.abs((pred-org)/org)*100
    return np.mean(temp)

#-----Exploratory Data Analysis EDA------

# Calculate 4-quarter moving average (helps smooth seasonal fluctuations)
mv_pred = cocacola["Sales"].rolling(4).mean()
mv_pred.tail(4)

# Evaluate moving average using last 4 predicted values vs Test set
MAPE(mv_pred.tail(4), Test.Sales)

# Plot original data
cocacola.Sales.plot(label='org')

# Compare moving averages with different window sizes (2, 4, 6, 8)
for i in range(2, 9, 2):
    cocacola["Sales"].rolling(i).mean().plot(label=str(i))
plt.legend(loc=3)
# Window=4 and 8 give smoother (deseasonalized) plots

# Decompose time series into trend, seasonal, residual components (Additive model)
decompose_ts_add = seasonal_decompose(cocacola.Sales, model="additive", period=4)
'''
seasonal_decompose: A function for breaking down a time series into:

Trend - the long-term upward or downward movement.

Seasonality - repeating short-term cycles (quarterly in this case).

Residual (Noise) - random irregular fluctuations that cannot be explained by trend/seasonality.

Observed - the original series (Sales).

model="additive": Assumes

Sales = Trend + Seasonality + Residual

This is appropriate when seasonal variations are roughly constant in size (not proportional).

period=4: Because it's quarterly data → 4 quarters = 1 year.
'''

print(decompose_ts_add.trend)       # Trend component
print(decompose_ts_add.seasonal)    # Seasonal component
print(decompose_ts_add.resid)       # Residual/noise
print(decompose_ts_add.observed)    # Original observed data
decompose_ts_add.plot()

# Same decomposition with Multiplicative model
decompose_ts_mul = seasonal_decompose(cocacola.Sales, model="multiplicative", period=4)
print(decompose_ts_mul.trend)
print(decompose_ts_mul.seasonal)
print(decompose_ts_mul.resid)
print(decompose_ts_mul.observed)
decompose_ts_mul.plot()
'''
Additive Model
= Trend + Seasonality + Residual

When to use:
Seasonal variations are constant in size (do not depend on sales level).

Example: Sales increase by +500 units every summer, regardless of whether base sales are high or low.

Visual clue: Seasonal ups/downs look like they have fixed height (parallel oscillations).

Multiplicative Model
Formula:
= Trend × Seasonality × Residual

When to use:
Seasonal variations are proportional to the trend (bigger when sales are high, smaller when sales are low).

Example: Sales increase by +20% every summer, so the absolute seasonal effect grows as sales increase.

Visual clue: Seasonal ups/downs widen as the trend increases (oscillations become larger).

In CocaCola Sales Graph
The fluctuations in the later quarters are bigger in size compared to the earlier quarters.

That suggests multiplicative seasonality might fit better since seasonal effect grows with sales.

Rule of thumb:
    use additive if seasonality looks flat/constant.
    use multiplication if seasonality grows with the trend (proportional effect)
    
'''
#plot Auto- Correlation function(ACF)to check correlation with lags
import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(cocacola.Sales, lags=4)
# Here, lags 1-4 show high correlation - confirms quarterly seasonality

# ---------------- Data-Driven Forecasting Models ----------------

# Simple Exponential Smoothing (captures only LEVEL)
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
ses_model = SimpleExpSmoothing(Train['Sales']).fit()
pred_ses = ses_model.predict(start=Test.index[0], end=Test.index[-1])
MAPE(pred_ses, Test.Sales)       # ~8.36%

# Holt’s Method (captures LEVEL + TREND, no seasonality)
hw_model = Holt(Train["Sales"]).fit()
pred_hw = hw_model.predict(start=Test.index[0], end=Test.index[-1])
MAPE(pred_hw, Test.Sales)        # ~9.80%

# Holt-Winters Exponential Smoothing (Additive Trend + Additive Seasonality)
hwe_model_add_add = ExponentialSmoothing(
    Train["Sales"], seasonal="add", trend="add", seasonal_periods=4
).fit()
pred_hwe_model_add_add = hwe_model_add_add.predict(start=Test.index[0], end=Test.index[-1])
MAPE(pred_hwe_model_add_add, Test.Sales)   # best so far
#1.5023826355347967

from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Holt-Winters with Additive Trend + Multiplicative Seasonality
hwe_model_mul_add = ExponentialSmoothing(
    Train["Sales"], seasonal="mul", trend="add", seasonal_periods=4
).fit()

pred_hwe_model_mul_add = hwe_model_mul_add.predict(start=Test.index[0], end=Test.index[-1])
MAPE(pred_hwe_model_mul_add, Test.Sales)
   # ~2.88%

# --------------- Final Forecasting ---------------

# Since Additive Trend + Additive Seasonality gave lowest MAPE, we use it
hwe_model_add_add = ExponentialSmoothing(
    cocacola["Sales"], seasonal="add", trend="add", seasonal_periods=4
).fit()

# Load new dataset for future forecasting
new_data = pd.read_excel("c:/Data-Science/10-Time/Newdata_CocaCola_Sales.xlsx")

# Predict on new data indices
newdata_pred = hwe_model_add_add.predict(start=new_data.index[0], end=new_data.index[-1])

# Compare forecast with test set
MAPE(newdata_pred, Test.Sales)#2.26251702

# Final predictions
newdata_pred
