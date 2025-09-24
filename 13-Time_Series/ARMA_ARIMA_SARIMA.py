# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 16:07:13 2025

@author: Ashlesha
"""

# ---------------------------------------------------------
# ARMA, ARIMA, and SARIMA Models on Walmart Footfalls Dataset
# ---------------------------------------------------------

import pandas as pd
import statsmodels.graphics.tsaplots as tsa_plots
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot as plt

# ---------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------
Walmart = pd.read_csv("C:/Data-Science/10-Time/Walmart Footfalls Raw.csv")

# Train = first 147, Test = last 12
Train = Walmart.head(147)
Test = Walmart.tail(12)

# ---------------------------------------------------------
# Step 1: Stationarity Test (ADF)
# ---------------------------------------------------------
print("===== Stationarity Check (ADF Test) =====")
result = adfuller(Train.Footfalls)
print("ADF Statistic:", result[0])
#ADF Statistic: -0.8725058977579943
print("p-value:", result[1])
#p-value: 0.7968917502773996

# Try differencing if needed
diff1 = Train.Footfalls.diff().dropna()
print("\nADF p-value after 1st difference:", adfuller(diff1)[1])
#0.01703376145592958

# ---------------------------------------------------------
# Step 2: ACF and PACF plots
# ---------------------------------------------------------
tsa_plots.plot_acf(Train.Footfalls, lags=20)
plt.title("ACF Plot (Suggests MA order)")
plt.show()

tsa_plots.plot_pacf(Train.Footfalls, lags=20)
plt.title("PACF Plot (Suggests AR order)")
plt.show()

# ---------------------------------------------------------
# Step 3a: Fit ARMA Model (when d=0, stationary)
# ---------------------------------------------------------
# Example: ARMA(2,1)
arma_model = ARIMA(Train.Footfalls, order=(2,0,1)).fit()
print("\n===== ARMA(2,1) Summary =====")
print(arma_model.summary())

'''
AIC = 1892.587
BIC = 1907.539
HQIC = 1898.662
AIC (Akaike Information Criterion) and BIC 
(Bayesian Information Criterion) are used to compare models.

Lower is better (indicates a better fit with fewer parameters).

If you try another model (say ARMA(1,1) or ARMA(2,2)), 
you should compare these values.
The model with lowest AIC/BIC is usually preferred.

Your ARMA(2,1) model is now fitted.
To decide if it's a good  model:
    
compare AIC/BIC  with alternative
 model(like ARMA(1,1),ARMA(3,1),ARMA(2,2))
'''
# ---------------------------------------------------------
# Step 3b: Fit ARIMA Model (when d>0)
# ---------------------------------------------------------
# Example: ARIMA(2,1,2)
arima_model = ARIMA(Train.Footfalls, order=(2,1,2)).fit()
print("\n===== ARIMA(2,1,2) Summary =====")
print(arima_model.summary())

# ---------------------------------------------------------
# Step 3c: Fit SARIMA Model (with seasonality)
# ---------------------------------------------------------
# Assume monthly seasonality (s=12)
sarima_model = SARIMAX(Train.Footfalls,
                       order=(2,1,2),        # Non-seasonal part
                       seasonal_order=(1,1,1,12)).fit()

'''
How to decide seasonal_order=(P,D,Q,s)
---------------------------------------
- Decide s (seasonal period)
- Look at your data’s natural seasonality.
Look at your data's natural seasonality.

Example:
Monthly data with yearly seasonality → s=12
Quarterly data with yearly seasonality → s=4
Daily data with weekly seasonality → s=7

So here, since "Footfalls" is monthly → s=12.

Decide D (seasonal differencing order)

Check if seasonal patterns repeat (cycle every 12 months).

If seasonal cycle not stationary (trend across years), apply
seasonal differencing (D=1).

Use:

Seasonal plots,  
Augmented Dickey-Fuller (ADF) test after seasonal differencing.

Often we start with D=1.

Decide P (seasonal AR order)  
Look at the seasonal PACF (partial autocorrelation at lags that are multiples of s)

Example: PACF spikes at lag 12 → P=1.

Decide Q (seasonal MA order)  
Look at the seasonal ACF (autocorrelation at lags that are multiples of s)

Example: ACF spikes at lag 12 → Q=1.

In your case (1,1,1,12) means:
s=12 → yearly cycle in monthly data.
D=1 → one seasonal differencing was enough to make series stationary.
P=1 → PACF showed spike at lag 12.
Q=1 → ACF showed spike at lag 12.

'''
print("\n===== SARIMA(2,1,2)(1,1,1,12) Summary =====")
print(sarima_model.summary())

'''
Log Likelihood = -746.054
-> A measure of fit. Higher (closer to 0) is better.

AIC = 1506.108
Akaike Information Criterion → balances fit vs complexity.
Lower AIC = better model.

Compare with other models (e.g., ARIMA, SARIMA with different orders).
BIC = 1526.393
Bayesian Information Criterion.
Penalizes complexity more than AIC.
Use for comparing models; lower is better.

HQIC = 1514.351
Hannan-Quinn Information Criterion.
Lies between AIC and BIC in strictness.
'''
# ---------------------------------------------------------
# Step 4: Forecast on Test Data
# ---------------------------------------------------------
start = len(Train)
end = start + len(Test) - 1

forecast_arma = arma_model.predict(start=start, end=end)
forecast_arima = arima_model.predict(start=start, end=end)
forecast_sarima = sarima_model.predict(start=start, end=end)

# ---------------------------------------------------------
# Step 5: Evaluate RMSE
# ---------------------------------------------------------
print("\nForecast RMSEs:")
print("ARMA(2,1): %.3f" % sqrt(mean_squared_error(Test.Footfalls, forecast_arma)))
print("ARIMA(2,1,2): %.3f" % sqrt(mean_squared_error(Test.Footfalls, forecast_arima)))
print("SARIMA(2,1,2)(1,1,1,12): %.3f" % sqrt(mean_squared_error(Test.Footfalls, forecast_sarima)))

# ---------------------------------------------------------
# Step 6: Plot Comparison
# ---------------------------------------------------------
plt.figure(figsize=(10,6))
plt.plot(Test.Footfalls.values, label="Actual", color="black")
plt.plot(forecast_arma.values, label="ARMA Forecast")
plt.plot(forecast_arima.values, label="ARIMA Forecast")
plt.plot(forecast_sarima.values, label="SARIMA Forecast")
plt.legend()
plt.title("Forecast Comparison")
plt.show()

'''
Interpretation


Black line → Actual values (Test set)
The true Walmart Footfalls for the last 12 months.

Blue line → ARMA Forecast
Almost flat and downward-sloping.
Fails to capture either trend or seasonal fluctuations.
Underestimates the actual data consistently.

Orange line → ARIMA Forecast
Better than ARMA, follows ups and downs somewhat.
Still lags behind peaks and valleys (doesn’t catch sharp changes).

Green line → SARIMA Forecast
Closest to the black line (Actual).
Captures seasonal spikes and drops much better.
Tracks turning points (e.g., at index 4, 5, 10) more realistically than ARIMA.

ARMA (Blue)
Clearly underfits → assumes too simple a structure.
Not suitable for data with trend/seasonality.

ARIMA (Orange)
Handles trend fairly well (due to differencing).
But no seasonal component → misses repeated yearly ups/downs.


SARIMA (Green)

Best fit → accounts for both trend and yearly seasonality.

Forecasts align more closely with actuals, especially around peaks/troughs.

Some mismatch still exists (forecasts slightly smoother than real data),
but overall much more realistic.

Takeaway

SARIMA is the most appropriate model for this dataset.

ARMA fails → ignores both trend and seasonality.

ARIMA is better but incomplete → misses the seasonal effect.

In business terms:

If Walmart used ARMA/ARIMA, they’d underestimate footfalls during peak months.

SARIMA provides closer-to-reality forecasts, helping in planning inventory,
staffing, and promotions.
'''



















