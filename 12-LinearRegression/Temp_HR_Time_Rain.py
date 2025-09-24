# -*- coding: utf-8 -*-
"""
Created on July 31, 2025
@author: Ashlesha
"""
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Weather data including rain
data = {
    "Hour": ["12 PM", "1 PM", "2 PM", "3 PM", "4 PM", "5 PM", "6 PM", "7 PM", "8 PM", "9 PM", "10 PM", "11 PM", "12 AM"],
    "Temperature (°C)": [33, 34, 34, 30, 30, 29, 28, 27, 27, 26, 26, 25, 25],
    "RealFeel (°C)": [35, 36, 35, 32, 31, 30, 29, 28, 28, 27, 27, 26, 26],
    "Rain": ["No", "No", "Rain", "Rain", "Rain", "No", "No", "No", "Rain", "No", "No", "No", "No"]
}

# Step 2: Create DataFrame
df = pd.DataFrame(data)

# Step 3: Plot temperature and real feel
plt.figure(figsize=(12, 6))
plt.plot(df["Hour"], df["Temperature (°C)"], marker='o', label="Temperature (°C)", color='orange', linewidth=2)
plt.plot(df["Hour"], df["RealFeel (°C)"], marker='s', label="RealFeel (°C)", color='red', linewidth=2)

# Step 4: Add rain bars or symbols
for i, rain in enumerate(df["Rain"]):
    if rain == "Rain":
        # Draw a blue bar under x-axis to indicate rain hour
        plt.bar(df["Hour"][i], 1, bottom=min(df["Temperature (°C)"]) - 3, color='skyblue', width=0.5, label="Rain" if i == 2 else "")
        # Optional: rain emoji label above point
        plt.text(df["Hour"][i], df["Temperature (°C)"][i] + 1, "", ha='center', fontsize=12)

# Final plot settings
plt.title("Kopargaon Weather (12 PM to 12 AM) with Rain Hours")
plt.xlabel("Hour")
plt.ylabel("Temperature (°C)")
plt.xticks(rotation=45)
plt.ylim(min(df["Temperature (°C)"]) - 4, max(df["RealFeel (°C)"]) + 3)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
