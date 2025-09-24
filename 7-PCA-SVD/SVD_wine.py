# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 15:47:01 2025

@author: Ashlesha
"""
import pandas as pd
#load the dataset
df = pd.read_csv("c:/Data-Science/7-PCA-SVD/wine.csv")
print(df.head())
#drop the target column (assumed to be 'Type')
x = df.drop(columns=['Type'])  # feature
y = df['Type']

#step3 Standardize the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)   # ✅ FIXED (x_scaled was missing)

#perform svd
from numpy.linalg import svd
#step 4 preform SVD
U, S, VT = svd(x_scaled, full_matrices=False)   # ✅ FIXED (== → =)

#U=left singular vector (sample*component)
#S: singular value
#VT: right singular  vector

#step 5: print shapes of decomposed mattrices
#step6 explained variance calcluation
import numpy as np
print(x_scaled.shape)

#compute variance explained by each component
explained_variance = (S**2) / (x_scaled.shape[0] - 1)   # ✅ FIXED
total_variance = explained_variance.sum()
explained_variance_ratio = explained_variance / total_variance

#display variance explained by each compont
for i, ev in enumerate(explained_variance_ratio):
    print(f"Componet {i+1}:{ev:.4f} ({ev*100:.2f}%)")

#step7 project data onto 1st 2 component (for visualization)
#1st 2 component projection
x_svd_2d = U[:, :2] * S[:2]   #  FIXED indexing

import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
for label in sorted(y.unique()):
    plt.scatter(
        x_svd_2d[y==label,0],
        x_svd_2d[y==label,1],
        label=f"Class {label}"   #  FIXED (lable → label)
    )

plt.xlabel("First SVD component")
plt.ylabel("Second SVD component")
plt.title("Wine dataset - SVD 2D Projection")
plt.legend()
plt.show()
