# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 15:45:31 2025

@author: Ashlesha
"""
#overfitting

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load dataset
cars = pd.read_csv("c:/Data-Science/9-Classification/LinearRegression/cars.csv")

# STEP 1: Standardize the features + target
scaler = StandardScaler()
cars_scaled = pd.DataFrame(
    scaler.fit_transform(cars[['HP', 'SP', 'VOL', 'MPG']]),
    columns=['HP', 'SP', 'VOL', 'MPG']
)

# STEP 2: Train-test split
cars_train, cars_test = train_test_split(cars_scaled, test_size=0.2, random_state=42)

# STEP 3: Define X and y
X_train = cars_train[['HP', 'SP', 'VOL']]
y_train = cars_train['MPG']
X_test = cars_test[['HP', 'SP', 'VOL']]
y_test = cars_test['MPG']

# linear Regression

lr=LinearRegression()
lr.fit(X_train, y_train)

train_pred_lr =lr.predict(X_train)
test_pred_lr=lr.predict(X_test)


train_rmse_lr = np.sqrt(mean_squared_error(y_train, train_pred_lr))
test_rmse_lr = np.sqrt(mean_squared_error(y_test, test_pred_lr))

print("----- Linear Regression -----")
print("Train RMSE:", round(train_rmse_lr, 4))
# Train RMSE: 0.4608
print("Test RMSE :", round(test_rmse_lr, 4))
# 0.6214
# Train RMSE < Test RMSE overfitted

# =============================
# Ridge Regression
# =============================
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

train_pred_ridge = ridge.predict(X_train)
test_pred_ridge = ridge.predict(X_test)

train_rmse_ridge = np.sqrt(mean_squared_error(y_train, train_pred_ridge))
test_rmse_ridge = np.sqrt(mean_squared_error(y_test, test_pred_ridge))


print("----- Ridge Regression -----")
print("Train RMSE:", round(train_rmse_ridge, 4))
# Train RMSE: 0.4711
print("Test RMSE :", round(test_rmse_ridge, 4))
#Test RMSE : 0.591
print("Ridge Coefficients:", ridge.coef_)
# here train RMSE is slightly less than test RMSE

# =============================
# Lasso Regression
# =============================
lasso = Lasso(alpha=0.01)
lasso.fit(X_train, y_train)

train_pred_lasso = lasso.predict(X_train)
test_pred_lasso = lasso.predict(X_test)

train_rmse_lasso = np.sqrt(mean_squared_error(y_train, train_pred_lasso))
test_rmse_lasso = np.sqrt(mean_squared_error(y_test, test_pred_lasso))

print("\n----- Lasso Regression -----")
print("Train RMSE:", round(train_rmse_lasso, 4))
# Train RMSE: 0.4696
print("Test RMSE :", round(test_rmse_lasso, 4))
# Test RMSE : 0.588
print("Lasso Coefficients:" ,lasso.coef_)
############
|Model             |Train RMSE|Test RMSE|Comments
|Linear Regression | 0.4608   |0.6214   | Overfit slightly
|Ridge Regression   |  0.4711    |  0.5910    |  Marginal improvement in generalization  
|Lasso Regression   |  0.4696    |  0.5880    |  Very slight improvement

Why There’s No Big Change?
Few features:

You're using only 3 features: 'HP', 'SP', 'VOL'.

Regularization helps only when many correlated or irrelevant features exist.
In your case, the model is already simple – no need for strong regularization.

No multicollinearity problem:
If these features are not highly correlated, regularization won’t change much.

Check using df.corr().

Low Variance:
Your test RMSE is only slightly worse than train RMSE – model is not overfitting severely.
