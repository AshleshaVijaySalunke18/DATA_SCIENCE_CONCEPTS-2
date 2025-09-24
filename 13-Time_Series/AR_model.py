# -*- coding: utf-8 -*-

"""
Created on Mon Sep  8 15:47:57 2025

@author: Ashlesha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")  # to avoid harmless warnings

# 1. Load dataset and preprocess
# -------------------------------
# Read CSV, parse DATE column as datetime, and set as index
df = pd.read_csv("C:/Data-Science/10-Time/uspopulation.csv", index_col="DATE", parse_dates=True)

'''
index_col="DATE"
This tells pandas:
"Use the column named DATE as the row index of the DataFrame."

Normally, pandas gives rows numeric indexes (0,1,2,...).

But here, we want time series indexing → rows indexed by dates
instead of numbers.

parse_dates=True

By default, pandas reads dates as plain strings ("2011-01-01").

With parse_dates=True, pandas converts them into datetime objects (Timestamp).

This allows us to do time series operations  like:
resampling  by  month/year

 Extracting .month, .year

 Plotting with time on the x-axis'''

# --------------------------------------------

# Tell pandas that data is monthly (MS = Month Start)
df.index.freq = "MS"

print(df.head())

# OUTPUT (first few rows should look like this):
#   DATE         PopEst
# 2011-01-01    311037

# Plot the data
df["PopEst"].plot(figsize=(12,5), title="U.S. Monthly Population Estimates")
plt.ylabel("Population Estimate")
plt.show()

# EXPECTED PLOT: Upward trend in population over time.

# ---------------------------------------------
# 2. Train-Test Split
# ---------------------------------------------
# First 84 months → training, last 12 months → testing
len(df)
train = df.iloc[:84]
test = df.iloc[84:]

print(len(train), len(test))
# EXPECTED: 84 training samples, 12 test samples

# ---------------------------------------------
# 3. Fit AR models with different lags
# ---------------------------------------------

# AR(1) model → uses 1 previous observation to predict the next
model = AutoReg(train["PopEst"], lags=1).fit()
pred1 = model.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)

'''
len(df) = 96

|<-------- TRAIN (84 months) -------->|<------ TEST (12 months) ----->|
2011 ................................. 2017        2018
Index: 0 ---------------------------- 83 | 84 ----------------------- 95

Dynamic Parameter Intuition

dynamic=False:
At each test step, it uses the true value from the past.
more accurate(good evalution)
dynamic=true
at each step , it uses its own  previous prediction (not real past)
Mimics "real forecasting" (when actuals are unknown).
Errors may accumulate.
'''
# AR(2) model → uses 2 previous observations
model2 = AutoReg(train["PopEst"], lags=2).fit()
pred2 = model2.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
 
# AR(11) model → uses 11 previous observations (chosen based on AIC in original code)
model_auto = AutoReg(train["PopEst"], lags=11).fit()
pred_auto = model_auto.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)

# ---------------------------------------------
# 4. Compare predictions with actual values
# -----------------------------------------
plt.figure(figsize=(12,6))

# Plot training data
plt.plot(train.index, train["PopEst"], label="Train")

# Plot actual test values
plt.plot(test.index, test["PopEst"], label="Test", color="black")

# Plot predictions from different AR models
plt.plot(test.index, pred1, label="AR(1) Prediction")
plt.plot(test.index, pred2, label="AR(2) Prediction")
plt.plot(test.index, pred_auto, label="AR(11) Prediction")

plt.legend()
plt.title("AR Model Forecasts")
plt.show()

# EXPECTED PLOT: AR(1) line should closely follow Test values,
# AR(2) and AR(11) might lag slightly behind.

# -----------------------------------------
# 5. Evaluation (MSE)
# -----------------------------------------

# Calculate Mean Squared Error (lower is better)
for label, pred in zip(["AR(1)", "AR(2)", "AR(11)"], [pred1, pred2, pred_auto]):
    error = mean_squared_error(test["PopEst"], pred)
    print(f"{label} MSE: {error:.2f}")

# EXPECTED OUTPUT
#AR(1) MSE: 17449.71 --THE BEST MODEL
#AR(2) MSE: 2713.26
#AR(11) MSE: 3206.19

# -----------------------------------------
# 6. Forecast future population
# -----------------------------------------

# Retrain AR(1) model on the full dataset for final forecasting
final_model = AutoReg(df["PopEst"], lags=2).fit()

# Forecast next 12 months
forecast = final_model.predict(start=len(df), end=len(df)+12, dynamic=False)

print(forecast.head())
# EXPECTED: First few future predictions (values slightly higher than last PopEst)

# Plot forecast vs historical data
plt.figure(figsize=(12,6))
plt.plot(df.index, df["PopEst"], label="Historical Data")
plt.plot(forecast.index, forecast, label="12-Month Forecast", color="red")
plt.legend()
plt.title("US Population Forecast using AR(1)")
plt.show()

# EXPECTED PLOT: Red forecast line continues the upward trend of population.






