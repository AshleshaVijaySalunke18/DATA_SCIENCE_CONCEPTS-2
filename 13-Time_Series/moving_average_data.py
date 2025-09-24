# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 15:44:35 2025

@author: Ashlesha
"""

import pandas as pd
import matplotlib.pyplot as plt

# Same time series data (daily sales for 20 days)
data = {
    "day": pd.date_range(start="2025-01-01", periods=20, freq="D"),  # ✅ Fixed typo: data_range -> date_range
    "sales": [20, 30, 34, 56, 54, 33, 56, 35, 76, 46, 48, 34, 64, 75, 43, 24, 64, 45, 67, 89]  # ✅ Added missing values to make 20 entries
}

df = pd.DataFrame(data)

df.set_index("day", inplace=True)  # ✅ Fixed syntax: inplace==True -> inplace=True

# Calculate moving averages
df["MA_3"] = df["sales"].rolling(window=3).mean()  # 3-day moving average
df["MA_5"] = df["sales"].rolling(window=5).mean()  # 5-day moving average

print(df)

# Plot original data with moving averages
plt.figure(figsize=(10, 5))
plt.plot(df.index, df["sales"], label="Original Sales", marker="o")
plt.plot(df.index, df["MA_3"], label="3-Day Moving Avg", linestyle="--")
plt.plot(df.index, df["MA_5"], label="5-Day Moving Avg", linestyle="--")
plt.legend()
plt.title("Moving Averages in Time Series")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
