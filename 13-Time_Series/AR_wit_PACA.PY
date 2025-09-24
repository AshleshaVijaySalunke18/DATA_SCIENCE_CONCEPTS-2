# --------------------------------------------------------
# AutoRegressive (AR) Model Example on US Population Dataset
# --------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_pacf
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")  # to avoid harmless warnings

# ------------------------------
# 1. Load dataset and preprocess
# ------------------------------
df = pd.read_csv("c:/Data_Science/10-Time /uspopulation.csv", index_col="DATE", parse_dates=True)
df.index.freq = "MS"  # Monthly Start frequency

print(df.head())
# Plot the data
df["PopEst"].plot(figsize=(12,5), title="U.S. Monthly Population Estimates")
plt.ylabel("Population Estimate")
plt.show()

# ------------------------------
# 2. Train-Test Split
# ------------------------------
train = df.iloc[:84]
test = df.iloc[84:]
print(len(train), len(test))

# ------------------------------
# 3a. Use PACF to decide AR order
# ------------------------------
plt.figure(figsize=(8,4))
plot_pacf(train["PopEst"], lags=20, method="ywm")

train["PopEst"]
'''
The time series data you're analyzing (US population estimates in your dataset)
Lags=20
Plot PACF values up to Lag 20 (i.e., correlation with values from 1 step back)
You can change this number depending on how far back you want to check dependence
method="ywm"
Specifies the calculation method.
"ywm" stands for Yule-Walker Modified, a common and stable way to estimate PACF.
Other options: "ols" (ordinary least squares), "Ld" (Levinson-Durbin), but "ywm" is preferred.
'''

plt.title("PACF Plot for AR Order Selection")
plt.show()

'''
How to decide AR order from PACF:

- Look at the vertical bars (partial autocorrelations).
- The shaded area represents the 95% confidence interval.
- Bars that extend beyond the shaded area are considered significant.

Decision Rule:
- If only Lag 1 is significant (outside shaded area) → choose AR(1).
- If Lags 1 and 2 are significant → choose AR(2).
- If Lags keep being significant up to Lag k, then cut off
→ choose AR(k).
- In the population dataset, PACF often shows significance up to Lag ~11,
which is why AR(11) was chosen in the original code.
'''

# ---------------------------------
# 3. Fit AR models with different lags
# ---------------------------------
model = AutoReg(train["PopEst"], lags=1).fit()
pred1 = model.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)

model2 = AutoReg(train["PopEst"], lags=2).fit()
pred2 = model2.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
# ---------------------------------
# 4. Compare predictions with actual values
# ---------------------------------
plt.figure(figsize=(12,6))
plt.plot(train.index, train["PopEst"], label="Train")
plt.plot(test.index, test["PopEst"], label="Test", color="black")
plt.plot(test.index, pred1, label="AR(1) Prediction")
plt.plot(test.index, pred2, label="AR(2) Prediction")
plt.legend()
plt.title("AR Model Forecasts")
plt.show()

# ---------------------------------
# 5. Evaluation (MSE)
# ---------------------------------
for label, pred in zip(["AR(1)", "AR(2)"], [pred1, pred2]):
    error = mean_squared_error(test["PopEst"], pred)
    print(f"{label} MSE: {error:.2f}")
# ---------------------------------
# 6. Forecast future population
# ---------------------------------
final_model = AutoReg(df["PopEst"], lags=2).fit()
forecast = final_model.predict(start=len(df), end=len(df)+12, dynamic=False)

print(forecast.head())

plt.figure(figsize=(12,6))
plt.plot(df.index, df["PopEst"], label="Historical Data")
plt.plot(forecast.index, forecast, label="12-Month Forecast", color="red")
plt.legend()
plt.title("US Population Forecast using AR(2)")
plt.show()
