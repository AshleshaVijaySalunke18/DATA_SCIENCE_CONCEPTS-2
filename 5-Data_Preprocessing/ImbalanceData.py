# -*- coding: utf-8 -*-
"""
Created on Wed May 21 15:27:48 2025

@author: Ashlesha
"""

#Imbalance Data Set
from sklearn.datasets import make_classification
from collections import Counter
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#step1 Create imbalance  data
x, y = make_classification(n_classes=2, class_sep=2,
                           weights=[0.95, 0.05],
                           n_informative=3, n_redundant=1, n_clusters_per_class=1,
                           n_samples=1000, random_state=42)

"""
This line uses `make_classification` from scikit-learn,
which generates a synthetic dataset suitable for classification tasks.
It's particularly useful for testing and demonstrating
machine learning techniques like SMOTE.

Parameter              | Description
-----------------------|------------------------------------------------------
`n_classes=2`          | We are simulating a **binary classification** problem.
`class_sep=2`          | Controls **how separable** the classes are. Higher = more separation.
`weights=[0.95, 0.05]` | Controls the **imbalance**. 95% of samples will belong to class 0.
`n_informative=3`      | Number of features that are actually **useful** for classification.
`n_redundant=1`        | Number of features that are **linear combinations** of informative ones.
`n_clusters_per_class=1` | Each class will form **1 cluster** in feature space.
`n_samples=1000`       | Total number of **rows (samples)** in the dataset.
`random_state=42`      | Fixes the randomness for **reproducibility**.
"""
print("Original class distribution:",Counter(y))
#optional1:Visualize original class distribution
sns.countplot(x=y)
plt.title("original class Distributtion")
plt.show()

#step 2 apply SMOTE
smote=SMOTE(randam_state=42)
x_resampled,y_resampled=smote.fit_resample(x,y)
print("Resampled class Distribution:",Counter(y_resampled))

#optinal : Visulize resampled class distribution
sns.countplot(x=y_resampled)
plt.title("After SMOTE:Resample class Doistribution")
plt.show()





###########################################
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 17:51:16 2025

@author: user
"""

# ---------------------------------------------------------
#  Feature             |  Normalization                   |  Standardization
# ---------------------|----------------------------------|----------------------------
#  Also Known As       |  Min-Max Scaling                 |  Z-score Scaling
#  Formula             |  (X - X_min) / (X_max - X_min)   |  (X - μ) / σ
#  Output Range        |  [0, 1] (or custom range)        |  Mean = 0, Std Dev = 1
#  Sensitive to Outliers | Yes                            |  Less sensitive
#  Use Case            |  When features have different    |  When data follows
#                      |  scales and are bounded          |  Gaussian distribution or
#                      |                                  |  algorithm assumes normality
#  Examples            |  KNN, Neural Networks            |  Logistic Regression,
#                      |                                  |  SVM, PCA, Linear Regression
# ---------------------------------------------------------

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Sample data...
data=pd.DataFrame({
    'income':[25000,48000,5500,72000,10000],
    'age':[21,25,30,37,34]})
print("Original Data:\n", data)

#---------------------------
#1 .  Normalization(min=max scaling)
#--------------------------------
min_max_scaler=MinMaxScaler()
normalized_data=min_max_scaler.fit_transform(data)
normalized_df=pd.DataFrame(normalized_data,columns=data.columns)
print("\nNormalizatized Datda(0-1 range):\n", normalized_df)

#-------------------------------
# 2.  Standarlization (z-score)
#--------------------------------
standard_scaler = StandardScaler()
standardized_data = standard_scaler.fit_transform(data)
# Create DataFrame from the standardized data
standardized_df = pd.DataFrame(standardized_data, columns=data.columns)
print(standardized_df.head())

############################################################################################
#-------------------------------------------------------------------------------------------
#SAMPLING TECH

import pandas as pd
import numpy as np
#create sample dataset

data = pd.DataFrame({
    'ID': range(1, 101),
    'Age': np.random.randint(20, 60, 100),
    'Gender': np.random.choice(['Male', 'Female'], 100)
})
# Simple Random Sampling
# Every record has an equal chance of being selected.
# Randomly select 10% of the data
simple_random_sample = data.sample(frac=0.1, random_state=42)
print("Simple Random Sample:\n", simple_random_sample.head())
# Stratified Sampling
# Data is split into strata (subgroups), and samples are drawn proportionally
# Stratified sampling based on 'Gender'
from sklearn.model_selection import train_test_split
stratified_sample, _ = train_test_split(
    data,
    test_size=0.9,                  # Keep 10% as sample
    stratify=data['Gender'],        # Stratify by 'Gender'
    random_state=42                 # Fix randomness for reproducibility
)
print("Stratified sample (by Gender):\n", stratified_sample['Gender'].value_counts())

#systematic Sampling
#selscted every k-th element from a list
#choose  every 10th item

k=10
systematic_sample=data.iloc[::k]
print("systematic sample :\n",systematic_sample.head())



















































