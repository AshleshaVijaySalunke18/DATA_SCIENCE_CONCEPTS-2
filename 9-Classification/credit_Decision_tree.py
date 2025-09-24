# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 15:23:37 2025

@author: Ashlesha
"""

# Import necessary libraries
import pandas as pd                 # For data manipulation
import numpy as np                  # For numerical operations
from sklearn.preprocessing import LabelEncoder  # For encoding categorical variables

# Load the dataset from a local path
data = pd.read_csv("C:/Data-Science/9-Classification/credit.csv")

# Check for missing values in the dataset
data.isnull().sum()

# Drop rows with missing values(not assigned back, so it won't) affects)
data.dropna()

# Display column names
data.columns

#Drop the 'phone' column (maybe considered unimportant or too uniqe)
data = data.drop(["phone"], axis=1)

# Convert categorical columns into numeric values using Label Encoding
# LabelEncoder assigns a numeric value to each category (e.g., 'low' -> 1, 'high' -> 2)
lb = LabelEncoder()

# Encoding multiple categorical columns
data["checking_balance"] = lb.fit_transform(data["checking_balance"])
data["credit_history"] = lb.fit_transform(data["credit_history"])
data["purpose"] = lb.fit_transform(data["purpose"])
data["savings_balance"] = lb.fit_transform(data["savings_balance"])
data["employment_duration"] = lb.fit_transform(data["employment_duration"])
data["other_credit"] = lb.fit_transform(data["other_credit"])
data["housing"] = lb.fit_transform(data["housing"])
data["job"] = lb.fit_transform(data["job"])

#Encoding the  target column "default" is alreadt done; so this line
#Data["Default"]=lb.fit_transform(data["default"])

# View unique values in the target column
data['default'].unique()

# Count the frequency of each class (0 or 1) in the target column
data['default'].value_counts()

# Get all column names and split them into predictors and target variable
colnames = list(data.columns)

# Select the first 15 columns as predictors
predictors = colnames[:15]

# The 16th column (index 15) is the target
target = colnames[15]

# Split the data into training and testing sets
# 70% training, 30% testing
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size=0.3)
from imblearn.over_sampling import SMOTE
from collections import Counter

# SMOTE
print("Before SMOTE:", Counter(train[target]))
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(train[predictors], train[target])
print("After SMOTE:", Counter(y_train_sm))

# Import Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier as DT

# View the documentation/help for DecisionTreeClassifier
help(DT)

# Create a Decision Tree model with entropy criterion (i.e., Information Gain)
model = DT(criterion='entropy')

#train the model on the  training data
model.fit(train[predictors],train[target])

#predict on the test data
preds=model.predict(test[predictors])

#create a confusion matrix to compare actual vs prediced values
pd.crosstab(test[target],preds,rownames=['Actual'],colnames=['Predictions'])

#calculate the accuracy on the test data
np.mean(preds==test[target])


##############################
# Now check accuracy on the training dataset (to see if it's overfitting)

# Predict on the training data
preds = model.predict(train[predictors])

# Confusion matrix for training data
pd.crosstab(train[target], preds, rownames=['Actual'], colnames=['Predicted'])

# Accuracy on training data
np.mean(preds == train[target])

'''
Interpretation: Model is Overfitting
The model achieves perfect accuracy on the training data,
which means it has learned the training examples too well,
possibly memorizing them.

But it has significantly lower accuracy on test data
(e,g 72%), which means it fails to generalize to new, unseen cases.

'''

# Suggestions to Prevent Overfitting
# Limit Tree Depth

model = DT(criterion='entropy', max_depth=4)

# Check accuracy on test data
# Train the model on the training data
model.fit(train[predictors], train[target])

# Predict on the test data
preds = model.predict(test[predictors])

# Create a confusion matrix to compare actual vs predicted values
pd.crosstab(test[target], preds, rownames=['Actual'], colnames=['Predicted'])

# Calculate the accuracy on the test data
test_accuracy = np.mean(preds == test[target])
test_accuracy

#now  check  accuracy on the  training dataset(to see if it's overfitting)

# Predict on the training data
preds = model.predict(train[predictors])

# Confusion matrix for training data
pd.crosstab(train[target], preds, rownames=['Actual'], colnames=['Predicted'])

# Accuracy on training data
train_accuracy = np.mean(preds == train[target])
train_accuracy

# If train_accuracy >> test_accuracy, model is overfitting
if train_accuracy - test_accuracy > 0.2:
    print("The model is likely overfitting — too good on training but poor on test data.")

# Prune Using Min Samples
model = DT(criterion='entropy', min_samples_split=10, min_samples_leaf=5)
#a node must have  at least 10rows to try  split
# Limit Tree Depth
model = DT(criterion='entropy', max_depth=4)

# Check accuracy on test data
# Train the model on the training data
model.fit(train[predictors], train[target])

# Predict on the test data
preds = model.predict(test[predictors])

# Create a confusion matrix to compare actual vs predicted values
pd.crosstab(test[target], preds, rownames=['Actual'], colnames=['Predicted'])

# Calculate the accuracy on the test data
test_accuracy = np.mean(preds == test[target])
test_accuracy

# Now check accuracy on the training dataset (to see if it's overfitting)

# Predict on the training data
preds = model.predict(train[predictors])

# Confusion matrix for training data
pd.crosstab(train[target], preds, rownames=['Actual'], colnames=['Predicted'])

# Accuracy on training data
train_accuracy = np.mean(preds == train[target])
train_accuracy

# If train_accuracy >> test_accuracy, model is overfitting
if train_accuracy - test_accuracy > 0.2:
    print("The model is likely overfitting – too good on training but poor on test data.")

# Try Ensemble Methods
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100)

rf_model.fit(train[predictors], train[target])

# Predict on the test data
preds = rf_model.predict(test[predictors])

# Create a confusion matrix to compare actual vs predicted values
pd.crosstab(test[target], preds, rownames=['Actual'], colnames=['Predicted'])

# Calculate the accuracy on the test data
test_accuracy = np.mean(preds == test[target])
test_accuracy

#now check accuracy on the  training dataset (to see if it's overfitting)

#predict on the trainging  data

preds = rf_model.predict(train[predictors])

#confusion  matrix fro training data
pd.crosstab(train[target],preds,rownames=['Actual'],colnames=['prediction'])

#accuracy on training data
train_accuracy = np.mean(preds == train[target])
train_accuracy

#If train_accuracy >> test_accuracy , model is overfitting
if train_accuracy - test_accuracy >0.2:
    print("The model is likely overfitting -too good on training but poor on train data")


############################################################






