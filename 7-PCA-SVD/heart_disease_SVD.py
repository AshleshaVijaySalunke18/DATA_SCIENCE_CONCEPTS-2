# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 15:29:56 2025

@author: Ashlesha
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from numpy.linalg import svd
import matplotlib.pyplot as plt

# Step 1: Load data
df = pd.read_csv("C:/Data-Science/7-PCA-SVD/heart_disease.csv")

#step2 sepreate feature and target
x=df.drop(columns=['target'])#all feature
y=df['target'] #target:0,1

#step 3 standerdize the feature
scaler=StandardScaler()
x_scaled=scaler.fit_transform(x )

#step4 apply singular value decomposition
U,S,VT=svd(x_scaled,full_matrices=False)

#step 5 project data onto first 2svd componests
x_svd_2d=U[:,:2]*S[:2]#2D projrction for plotting

#step 6 plotting
plt.figure(figsize=(8,6))
for lable in sorted(y.unique()):
    plt.scatter(x_svd_2d[y==lable,0],x_svd_2d[y==lable,1],
                lable=f'Class{lable}',alpha=0.6)
plt.title("SVD:Projection onto 1st  2 components(Heart Dataset")
plt.xlable("component:1")
plt.ylable("component:2")
plt.show()



























































































