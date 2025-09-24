# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 15:38:54 2025

@author: Ashlesha
"""

# =============================================
# Import Libraries
# =============================================
import pandas as pd                    # For handling data (CSV, DataFrame, etc.)
import numpy as np                     # For numerical computations
import matplotlib.pyplot as plt        # For plotting graphs
from sklearn.preprocessing import MinMaxScaler     # For scaling data between 0 and 1
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator  # To create sequences
from tensorflow.keras.models import Sequential     # Sequential model API
from tensorflow.keras.layers import Dense, LSTM    # Neural network layers
from sklearn.metrics import mean_squared_error     # To evaluate prediction performance
from math import sqrt                              # For RMSE calculation

# =============================================
# Load Data
# =============================================
# Load the CSV file, set "Month" as index, and parse dates properly
df = pd.read_csv("C:/Data-Science/10-Time/monthly-milk-production.csv", index_col="Month", parse_dates=True)

# Ensure the frequency of the index is Monthly Start ("MS")
df.index.freq = "MS"

# Rename the production column to a shorter name for convenience
df.rename(columns={"Monthly milk production (pounds per cow)": "Production"}, inplace=True)

print(df.head())  # Display first 5 rows

# Plot the original milk production time series
df.plot(figsize=(12, 6), title="Monthly Milk Production")
plt.show()

# =============================================
# Train-Test Split
# =============================================
# Use first 156 months (about 13 years) as training set
train = df.iloc[:156]
# Remaining months are used as testing set
test = df.iloc[156:]

#scale the data between 0 to  1 (LSTMs work  better with scaled input)

scaler=MinMaxScaler()
scaler.fit(train)
scaled_train= scaler.transform(train)
scaled_test=scaler.transform(test)   

#==============================
# Create Time Series Generator
# ============================
# We will use the past 12 months to predict the next month
n_input = 12         # lookback window (12 months)
n_features = 1       # univariate time series (only Production column)

# TimeseriesGenerator automatically creates input-output pairs for LSTM

generator = TimeseriesGenerator(scaled_train, scaled_train,
                                length=n_input, batch_size=1)

'''
generator is the key part that prepares the training data
for the LSTM model.
What it does:

TimeseriesGenerator automatically creates input-output pairs from a time series for supervised learning.

Inputs (X) = sequences of the last n_input months.
Outputs (y) = the next month’s value after those n_input months.

Here:
scaled_train → the training data (already scaled between 0 and 1).

length = n_input → Lookback window = 12 months.
batch_size = 1 → one sequence per batch (processed one by one).

Suppose scaled_train has 15 values:

[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

and n_input = 3.

The generator will automatically create:


and n_input = 3.

The generator will automatically create:

Input (X) → Output (y)

[1, 2, 3] → 4  
[2, 3, 4] → 5  
[3, 4, 5] → 6  
...  
[12, 13, 14] → 15

So the LSTM Learns:  
Given the last 3 numbers, predict the next one.

In our milk production dataset:  
Given the last 12 months of milk production, predict the next month.


'''
# ================================
# Build LSTM Model
# ================================
model = Sequential()
# LSTM layer with 100 units, relu activation, input is (12 time steps, 1 feature)
model.add(LSTM(100, activation="relu", input_shape=(n_input, n_features)))
# Dense layer with 1 neuron (output = next month's production)
model.add(Dense(1))
# Compile model with Adam optimizer and Mean Squared Error loss
model.compile(optimizer="adam", loss="mse")
# Train the model for 50 epochs using the training generator
model.fit(generator, epochs=50, verbose=1)

# ==============================
# Forecasting
# ==============================
test_predictions = []

# Take the last 12 months from training data as the first prediction input
first_eval_batch = scaled_train[-n_input:]
# scaled_train[-n_input:] → grabs the last 12 scaled values from the training data.
# These are used as the starting point to begin forecasting.

current_batch = first_eval_batch.reshape((1, n_input, n_features))

'''
LSTM expects input in 3D shape:
(batch_size, timesteps, features)
Here:
batch size = 1 (one sequence at a time)
timestep=n_input=12
feature=1(just one column:milk production)
so current_batch is  shaped as(1,12,1)

'''
#Predict step by step for each month in test set
for i in range(len(test)):
    #Predict the next month
    current_pred = model.predict(current_batch, verbose=0)[0][0]   # ✅ extract scalar
    test_predictions.append([current_pred])                       # ✅ keep 2D shape
    #update the batch(drop oldest month,append newest prediciton)
    current_batch = np.append(current_batch[:, 1:, :], [[[current_pred]]], axis=1)  # ✅ match (1,12,1)
    
'''
Loop runs for each month in the test set.
At every step:
The model predicts the next month (current_pred).
Prediction is stored in test_predictions.

current_batch[:, 1:, :] → drops the oldest month (sliding window).
[[current_pred]] → adds the new prediction as the most recent month.

So the batch always contains the last 12 months (some real, some predicted).

This is called recursive forecasting: predictions are fed back as input to predict.
'''

# Inverse transform predictions back to original scale
true_predictions = scaler.inverse_transform(test_predictions)

'''
Since training data was scaled (0-1), predictions are also in that range.

inverse_transform maps them back to the original milk production units (pounds per cow).
'''
# ==============================
# Evaluation
# ==============================

# Add predictions to test DataFrame
test["Predictions"] = true_predictions

# Calculate Root Mean Squared Error (RMSE)
rmse = sqrt(mean_squared_error(test["Production"], test["Predictions"]))
print(f"Test RMSE: {rmse:.4f}")
# Plot actual vs predicted values
test.plot(figsize=(14, 5), title="Milk Production Forecast")
plt.show()