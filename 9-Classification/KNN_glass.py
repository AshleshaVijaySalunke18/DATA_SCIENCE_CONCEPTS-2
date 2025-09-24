# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 16:32:12 2025

@author: Ashlesha
"""

"""
A glass manufacturing plant uses different earth
elements to design new glass materials based on
customer requirements. For that, they would like
to automate the process of classification as it's
a tedious job to manually classify them.

Help the company achieve its objective by correctly
classifying the glass type based on the other features 
using KNN algorithm

1. Business Problem
1.1. What is the business objective?
    1.1.1 Glass production still faces the challenges of finding optimum combination 
          for reducing atmospheric emission.
    1.1.2 Identifying, reducing and replacing the hazardous substances in the 
          material that end up with end product.

1.2. Are there any constraints?
    1.2.1 Issue of climate change and energy consumptions 
          are the major constraints
"""
#Data Description
#Data Set Characteristics: Multivariate
#Number of Instances: 214

#1.  Id number: 1 to 214
#2.  RI: refractive index
#3.  Na: Sodium (unit measurement: weight percent in corresponding oxide, as are all the others)
#4.  Mg: Magnesium
#5.  Al: Aluminum
#6.  Si: Silicon
#7.  K: Potassium
#8.  Ca: Calcium
#9.  Ba: Barium
#10. Fe: Iron
#11. Type of glass: (class attribute)

#Glass Type 1 building_windows_float_processed
#Glass Type 2 building_windows_non_float_processed
#Glass Type 3 vehicle_windows_float_processed
#Glass Type 4 vehicle_windows_non_float_processed (none in this database)
#Glass Type 5 containers
#Glass Type 6 tableware
#Glass Type 7 headlamps

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

glass=pd.read_csv("c:/Data-Science/9-Classification/glass.csv")

############

#Exploratory Data Analysis(EDA)
glass.dtypes

#All the input feature are of float type and output 
glass.columns
glass.describe()

#the minimum value of RI 1.511 and max 1.53
#the avarage value RT 1.51

# Select only feature columns (RI to Fe) for boxplot
feature_cols = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']

# Boxplot for RI
sns.boxplot(x='Type', y='RI', data=glass, palette='Set2')
sns.boxplot(x='Type', y='Na', data=glass, palette='Set2')
sns.boxplot(x='Type', y='Mg', data=glass, palette='Set2')
sns.boxplot(x='Type', y='Al', data=glass, palette='Set2')
sns.boxplot(x='Type', y='Si', data=glass, palette='Set2')
sns.boxplot(x='Type', y='K', data=glass, palette='Set2')
sns.boxplot(x='Type', y='Ca', data=glass, palette='Set2')
sns.boxplot(x='Type', y='Ba', data=glass, palette='Set2')
sns.boxplot(x='Type', y='Fe', data=glass, palette='Set2')

#Histograms 

sns.histplot(glass['RI'])
sns.histplot(glass['Mg'])
sns.histplot(glass['Al'])
sns.histplot(glass['Si'])
sns.histplot(glass['k'])
sns.histplot(glass['Ca'])
sns.histplot(glass['Ba'])
sns.histplot(glass['Fe'])




# Pairplot 
sns.pairplot(glass, hue='Type', palette='Set2', diag_kind='hist', corner=True)
sns.pairplot(glass[['RI', 'Type']], hue='Type', palette='Set2', diag_kind='kde')
sns.pairplot(glass[['Na', 'Type']], hue='Type', diag_kind='kde', palette='Set2')
sns.pairplot(glass[['Mg', 'Type']], hue='Type', diag_kind='kde', palette='Set2')
sns.pairplot(glass[['Al', 'Type']], hue='Type', diag_kind='kde', palette='Set2')
plt.suptitle("Pairplot of Aluminum (Al) by Glass Type", y=1.02)
sns.pairplot(glass[['Si', 'Type']], hue='Type', diag_kind='kde', palette='Set2')
sns.pairplot(glass[['K', 'Type']], hue='Type', diag_kind='kde', palette='Set2')
sns.pairplot(glass[['Ca', 'Type']], hue='Type', diag_kind='kde', palette='Set2')
sns.pairplot(glass[['Ba', 'Type']], hue='Type', diag_kind='kde', palette='Set2')
sns.pairplot(glass[['Fe', 'Type']], hue='Type', diag_kind='kde', palette='Set2')
sns.pairplot(glass[['Type']], hue='Type', diag_kind='kde', palette='Set2')

# Scatter plot 
plt.scatter(glass['Type'], glass['RI'], color='green')
plt.scatter(glass['Type'], glass['Na'], color='darkpink')
plt.scatter(glass['Type'], glass['Mg'], color='purple')
plt.scatter(glass['Type'], glass['Al'], color='brown')
plt.scatter(glass['Type'], glass['Si'], color='skyblue')
plt.scatter(glass['Type'], glass['K'], color='black')
plt.scatter(glass['Type'], glass['Ca'], color='yellow')
plt.scatter(glass['Type'], glass['Ba'], color='red')
plt.scatter(glass['Type'], glass['Fe'], color='pink')

#Drop the ID column as it's not useful for prediction
glass = glass.iloc[:, 1:]

# Step 3: Normalize the features (excluding the target column 'Type')

scaler = MinMaxScaler()
X = scaler.fit_transform(glass.iloc[:, :-1])  
y = glass.iloc[:, -1]  

#Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Apply KNN classifier

knn = KNeighborsClassifier(n_neighbors=5) 
knn.fit(X_train, y_train)  

#Make predictions

y_pred = knn.predict(X_test)

#Evaluate the model

accuracy_score(y_test, y_pred)
confusion_matrix(y_test, y_pred)

#Try different values of k and plot accuracy

acc = []

for k in range(3, 50, 2):  # Trying odd values of k
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    acc.append(model.score(X_test, y_test))

## Plotting accuracy for different k values

plt.plot(range(3, 50, 2), acc, marker='o')
plt.title('Accuracy vs K value')
plt.xlabel('K value')
plt.ylabel('Accuracy')
plt.grid()
plt.show()


