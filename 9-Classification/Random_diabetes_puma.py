# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 16:21:43 2025

@author: Ashlesha
"""
'''
how to train a classification model (diabetes prediction) using:
A Decision Tree
A Bagging Classifier
A Random Forest
And compares the performance using Cross-Validation 
and Out-of-Bag score

'''

import pandas as pd

df = pd.read_csv("c:/Data-Science/9-Classification/diabetes_puma.csv")
df.head()
df.isnull().sum()
df.describe()
df.Outcome.value_counts()
#outcome

#0    500
#1    268

#There is slight imbalance in our dataset but since
#it is not major we will not worry about it!
#Train test split

X = df.drop("Outcome", axis="columns")
y = df.Outcome

'''
X contains all features except Outcome.
y contains the target variable (0 or 1).
'''
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#Scales features to have mean = 0 and std = 1.

X_scaled[:3]
#in  order to make your data balanced while splitting you can
#use stratify

from sklearn.model_selection import train_test_split
X_train, X_test ,y_train,y_test=train_test_split(X_scaled,y,stratify=y, random_state=10)
X_train.shape
X_test.shape
y_train.value_counts()
'''
0    375
1    201
#fairly balanced
'''

201/375
#0.536

y_test.value_counts()
'''
0    125
1     67
'''

67/125
#train using stand along model
