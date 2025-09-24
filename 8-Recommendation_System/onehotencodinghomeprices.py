# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 05:32:11 2025

@author: user
"""

# Step 1: Import required libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Step 2: Load the dataset
# Assumes 'homeprices.csv' contains columns: 'town', 'area', 'price'
df = pd.read_csv("c:/Data-Science/8-RecommendationSystem/homeprices.csv")
print("Original Data:")
print(df)

# Step 3: Separate features and target variable
X = df[['town', 'area']]   # Features: town and area
y = df['price']            # Target: price

# Step 4: Convert 'town' column into numbers using OneHotEncoder
This will create extra columns like:
 - robbinsville: 1 or 0
 - east windsor: 1 or 0
 (west windsor is automatically dropped to avoid duplication)

# If your data was like this:
 |town         |  area |
 |-----------  | ----- |
 |west windsor |  2600 |
 |robbinsville |  2800 |
 |east windsor |  3000 |

 If you have a column like town with 3 categories:
 Then One-Hot Encoding gives you 3 new columns:
 |west windsor | robbinsville | east windsor
      1       |      0       |      0
      0       |      1       |      0
      0       |      0       |      1

 Notice something?
 The value of one column is always predictable if you know the other two.
 Example:
 If robbinsville = 0 and east windsor = 0, it must be west windsor = 1.
 This creates multicollinearity.
 This means one column is a perfect combination of the others.
 This causes problems in linear regression 
– the model can’t tell which feature is actually causing the change.

 Solution: Drop one column
OneHotEncoder(drop='first')
| robbinsville | east windsor |   Meaning
|--------------|---------------|-----------------------
|      0       |       0       | → means west windsor
|      1       |       0       | → means robbinsville
|      0       |       1       | → means east windsor


'''
'town_encoder' is just a name you give (can be anything).
OneHotEncoder(drop='first') encodes the town column (column 0).
[0] means apply it to the first column of X (which is 'town').
remainder='passthrough' tells it to leave the other columns 
(like 'area') unchanged.
'''

encoder = ColumnTransformer(
    [('town_encoder', OneHotEncoder(drop='first'), [0])],  # town is at index 0
    remainder='passthrough'  # keep 'area' as-is
)

#Apply transformation
X_encoded = encoder.fit_transform(X)

# Step 5: Train the model
model = LinearRegression()
model.fit(X_encoded, y)

# Step 6: Predict prices for new homes
#predict for 3400 sq.ft in west windsor
print("West Windsor (3400 sq.ft):", model.predict([[0, 0, 3400]]))
#predict for 2800 sq.ft in  robbinsville
print("Robbinsville (2800 sq.ft):", model.predict([[1, 0, 2800]]))
#predict for 3000 sq.ft in east windsor
print("East Windsor (3000 sq.ft):", model.predict([[0, 1, 3000]]))
# Step 7: Show model accuracy (R² score)
print("Model Accuracy (R² score):", model.score(X_encoded, y))
