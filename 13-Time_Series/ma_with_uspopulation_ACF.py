# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 16:04:16 2025

@author: Ashlesha
"""

# AutoRegressive (AR) and Moving Average (MA) Order Selection
# Example on US Population Dataset
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# 1. Load dataset and preprocess
# ------------------------------
df = pd.read_csv("c:/Data-Science/10-Time/uspopulation.csv", index_col="DATE", parse_dates=True)
df.index.freq = "MS"  # monthly

# 2. Train-Test Split
# -------------------
train = df.iloc[:84]
test = df.iloc[84:]

print(len(train), len(test))   # EXPECTED: 84 training, 12 testing

# 3. ACF (for MA order selection)
# -------------------------------
plt.figure(figsize=(8,4))
plot_acf(train["PopEst"], lags=20)
plt.title("ACF Plot for MA Order Selection")
plt.show()

'''
Interpretation:
- In ACF, if only Lag 1 is significant → choose MA(1).
- If lags 1 and 2 are significant → choose MA(2).
- If up to lag q are significant → choose MA(q).
- After lag q, the correlations should cut off (fall inside blue band).
'''

# 4. Fit MA models using ARIMA(0,q,0)
# -----------------------------------

ma1 = ARIMA(train["PopEst"], order=(0,1,0)).fit()
pred_ma1 = ma1.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)

ma2 = ARIMA(train["PopEst"], order=(0,2,0)).fit()
pred_ma2 = ma2.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)

ma3 = ARIMA(train["PopEst"], order=(0,3,0)).fit()
pred_ma3 = ma3.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)

# 5. Compare MA predictions
# --------------------------
plt.figure(figsize=(12,6))
plt.plot(train.index, train["PopEst"], label="Train")
plt.plot(test.index, test["PopEst"], label="Test", color="black")
plt.plot(test.index, pred_ma1, label="MA(1) Prediction")
plt.plot(test.index, pred_ma2, label="MA(2) Prediction")
plt.plot(test.index, pred_ma3, label="MA(3) Prediction")
plt.legend()
plt.title("MA Model Forecasts")
plt.show()

# 6. Evaluation
# -------------
for label, pred in zip(["MA(1)", "MA(2)", "MA(3)"], [pred_ma1, pred_ma2, pred_ma3]):
    error = mean_squared_error(test["PopEst"], pred)
    print(f"{label} MSE: {error:.2f}")
'''
MA(1) MSE: 1483818.83
MA(2) MSE: 6940.83
MA(3) MSE: 299460.83
'''
# 7. Final Forecast using best MA model
# -------------------------------------
final_ma = ARIMA(df["PopEst"], order=(0,2,0)).fit()   # Example: MA(2)
forecast = final_ma.predict(start=len(df), end=len(df)+12, dynamic=False)

plt.figure(figsize=(12,6))
plt.plot(df.index, df["PopEst"], label="Historical Data")
plt.plot(forecast.index, forecast, label="12-Month Forecast (MA(2))", color="red")
plt.legend()
plt.title("US Population Forecast using MA Model")
plt.show()






