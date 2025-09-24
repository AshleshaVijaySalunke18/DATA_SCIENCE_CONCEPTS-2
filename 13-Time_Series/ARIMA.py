# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 16:25:14 2025

@author: Ashlesha
"""
'''
What is d in ARIMA(p,d,q)?

d = degree of differencing.

It tells the model how many times the time series has been differenced.

Differencing means: subtract the previous value from the current value.

Example with d=1:
If series is [100, 105, 111, 120],
then first difference = [105-100, 111-105, 120-111] = [5, 6, 9].

◆ Why is d important?

Stationarity requirement
ARIMA models assume the time series is stationary (constant mean, variance, autocovariance over time).

If data has a trend (upward/downward), differencing removes it.

If variance grows with time, log + differencing stabilizes it.

Controls model complexity

d=0: model raw stationary data (like stock returns).
d=1: remove a linear trend.
d=2: remove quadratic trends (rare).
Too much differencing can add noise → poor forecasts.

Works with AR and MA terms
After differencing, AR (p) and MA (q) terms work on the “stationarized” data.

How to decide d?

Visual check: if data shows clear trend → likely d=1.

Statistical test: Augmented Dickey-Fuller (ADF) or KPSS.

If ADF p-value > 0.05 → series is non-stationary → need differencing.

Keep differencing until p-value < 0.05.

ACF shape:
If ACF decays very slowly → need differencing.

◆ Practical guideline

Usually, d = 0, 1, or 2.

Start with d=1, check residuals:
If still non-stationary → increase d.

If over-differenced (series oscillates around with no structure) → reduce d.
'''

import pandas as pd
import statsmodels.graphics.tsaplots as tsa_plots
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller   # For stationarity test
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot

# -----------------------------
# Load Dataset
# -----------------------------
Walmart = pd.read_csv("C:/Data-Science/10-Time/Walmart Footfalls Raw.csv")

# Data Partition (Train = first 147, Test = last 12)
Train = Walmart.head(147)
Test = Walmart.tail(12)

# -----------------------------
# Step 1: Check Stationarity with ADF Test
# -----------------------------
print("===== Stationarity Check (ADF Test) =====")
result = adfuller(Train.Footfalls)
print("ADF Statistic:", result[0])
#ADF Statistic: -0.8725058977579943
print("p-value:", result[1])
#0.7968917502773996

# Rule of thumb:
# If p-value > 0.05 -> series is NOT stationary -> apply differencing
# If p-value < 0.05 -> series IS stationary -> d=0

# Try 1st order differencing
diff1 = Train.Footfalls.diff().dropna()
result1 = adfuller(diff1)
print("\nAfter 1st difference -> p-value:", result1[1])
# p-value: 0.01703376145592958

# Try 2nd order differencing if needed
diff2 = diff1.diff().dropna()
result2=adfuller(diff2)
print("After 2nd  difference p - value", result2[1])
#1.243854967433062e-18

# Based on the results, you can decide whether d=0, d=1, or d=2
# In our ARIMA, we will tentatively set d=1 (most common)

# ---------------------------------------------------------
# Step 2: Plot ACF and PACF (to decide p and q)
tsa_plots.plot_acf(Walmart.Footfalls, lags=12)   # Suggests q
tsa_plots.plot_pacf(Walmart.Footfalls, lags=12)  # Suggests p

# ---------------------------------------------------------
# Step 3: Fit ARIMA Model

# Example: ARIMA with AR=4, d=1, MA=6
# AR (p) = 4 → taken from PACF
# d = 1 → from ADF test differencing
# MA (q) = 6 → taken from ACF
model1 = ARIMA(Train.Footfalls, order=(4,1,6))
res1 = model1.fit()

print("\n===== ARIMA Model Summary =====")
print(res1.summary())

'''
Model Selection Criteria

AIC  = 1813.093
BIC  = 1845.913
HQIC = 1826.428

These are penalized fit measures:

AIC (Akaike Information Criterion):
Balances model fit vs complexity.

Lower is better.
BIC (Bayesian Information Criterion):
Similar to AIC but penalizes complex models more strongly.

HQIC (Hannan-Quinn Criterion):
Lies between AIC and BIC in terms of penalty.

If you try multiple ARIMA models (e.g., ARIMA(2,1,2), ARIMA(3,1,4)), compare AIC, BIC, HQIC.
the  lowest AIC/BIC indicates the best trade-off between accuracy and  simplicity

P>|z|: p-value.
If < 0.05 → statistically significant.

If > 0.05 → may not be meaningful.


Interpretation of Each Coefficient
----------------------------------

AR Terms
ar.L1 = -0.7199, p=0.211 → not significant
ar.L2 = -0.7579, p=0.043 → significant
ar.L3 = -0.7094, p=0.220 → not significant
ar.L4 =  0.2441, p=0.515 → not significant

Only AR(2) is contributing meaningfully.

MA Terms
---------

ma.L1 =  0.2439, p=0.625 → not significant
ma.L2 =  0.7290, p=0.093 → borderline (weak evidence)
ma.L3 =  0.1224, p=0.843 → not significant
ma.L4 = -0.7716, p=0.215 → not significant
ma.L5 = -0.1027, p=0.694 → not significant
ma.L6 = -0.5691, p=0.042 → significant

Only MA(6) is clearly significant, while MA(2) is borderline.

ma.L6 = -0.5691, p=0.042 → significant

Only MA(6) is clearly significant, while MA(2) is borderline.

Final Takeaways

Out of 10 parameters (AR4 + MA6), only AR(2) and MA(6) are 
statistically significant.

The rest may be noise → your model might be over-parameterized.

A simpler model like ARIMA(2,1,1) or ARIMA(2,1,2) 
may perform equally well (with lower AIC/BIC).

In plain words:
Your ARIMA(4,1,6) model fits, but most coefficients are not statistical.
You should retest with smaller models (like ARIMA(2,1,2)) 
and compare AIC/BIC + RMSE.

'''

# -------------------------------------------------
# Step 4: Forecast for Test Data
# -------------------------------------------------
start_index = len(Train)
end_index = start_index + len(Test) - 1
forecast_test = res1.predict(start=start_index, end=end_index)

print("\nForecasted Values:")
print(forecast_test)

# -------------------------------------------------
# Step 5: Evaluate Forecast Accuracy
# -------------------------------------------------
rmse_test = sqrt(mean_squared_error(Test.Footfalls, forecast_test))
print("\nTest RMSE: %.3f" % rmse_test)
#Test RMSE: 172.699
# -------------------------------------------------
# Step 6: Plot Actual vs Predicted
# -------------------------------------------------
pyplot.plot(Test.Footfalls, label="Actual")
pyplot.plot(forecast_test, color='red', label="Forecast")
pyplot.legend()
pyplot.show()


