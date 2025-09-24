# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 15:18:34 2025

@author: Ashlesha
"""

import pandas as pd
df = pd.read_csv("c:/Data-Science/9-Classification/movies_classification.csv")
df.head()
df.columns
df.dtypes

# There are two columns of object type
df = pd.get_dummies(df, columns=["3D_available", "Genre"], drop_first=True)

# Assign input and output
predictors = df.loc[:, df.columns != "Start_Tech_Oscar"]
target = df["Start_Tech_Oscar"]

# train_test_split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size=0.3, random_state=42)
from sklearn.ensemble import GradientBoostingClassifier
grand_boost = GradientBoostingClassifier()

grand_boost.fit(x_train,y_train)
#Evalution of model
pred1=grand_boost.predict(x_test)
from sklearn.metrics import accuracy_score,confusion_matrix
accuracy_score(y_test,pred1)
confusion_matrix(y_test,pred1)

#Evalutation on training data
pred2=grand_boost.predict(x_train)
accuracy_score(y_train,pred2)

#let us change the hyper parameters
grand_boost1=GradientBoostingClassifier(learning_rate=0.02,n_estimators=5000,max_depth=1)

'''
learning_rate=0.02
Controls how much each new tree contributes to the overall prediction.
Lower value → slower learning, but potentially better accuracy.

Works well with larger n_estimators.
n_estimators = 5000
Number of boosting rounds (or trees).
Since learning_rate is small, more trees are needed to fit the data.

max_depth = 1
Each tree is a shallow tree (decision stump).
Learns only simple rules (1 split per tree).
This helps the model learn gradually and avoid overfitting.

'''

grand_boost1.fit(x_train,y_train)

'''
Trains the GradientBoostingClassifier on the features and target.
Each tree is built sequentially to correct
the errors of the previous model.
After 5000 trees, you get the final boosted model.
'''

############################################

from sklearn.metrics import accuracy_score, confusion_matrix
pred3 = grand_boost1.predict(x_test)
accuracy_score(y_test, pred3)
confusion_matrix(y_test, pred3)

############################################

## Evaluation on training data
pred4 = grand_boost1.predict(x_train)
accuracy_score(y_train, pred4)

############################################








