# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 15:43:57 2025

@author: Ashlesha
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Load dataset
letters = pd.read_csv("c:/Data-Science/9-Classification/SVM/Letterdata.csv")

# Split train/test
train, test = train_test_split(letters, test_size=0.2, random_state=42)
train_X, train_y = train.iloc[:, 1:], train.iloc[:, 0]
test_X, test_y = test.iloc[:, 1:], test.iloc[:, 0]

# ---------------------------------
# Helper function for training and testing

def evaluate_svc(**kwargs):
    """Train SVC with  given parameters and return test accurac"""
    model=SVC(**kwargs)
    model.fit(train_X, train_y)
    preds = model.predict(test_X)
    acc = np.mean(preds == test_y)
    print(f"Params: {kwargs} --> Accuracy: {acc:.4f}")
    return acc


# ==============================================================
# 1. Effect of Kernel
# ==============================================================
print("\n--- Kernel Comparison ---")
for k in ["linear", "poly", "rbf", "sigmoid"]:
    evaluate_svc(kernel=k)

'''
--- Kernel Comparison ---
Params: {'kernel': 'linear'} --> Accuracy: 0.8545
Params: {'kernel': 'poly'} --> Accuracy: 0.9507
Params: {'kernel': 'rbf'} --> Accuracy: 0.9305
'''

"""
Kernel choice
linear: Works well when features are linearly separable. Accuracy ~85%.
rbf: Flexible, handles non-linear data → usually best for this dataset (~90%+).
poly: Can perform well but depends on degree. Risk of overfitting with high degree.
sigmoid: Often unstable; not widely used.
"""

# ==============================================================
# 2. Effect of Regularization Parameter C
# (C controls margin width vs classification error)
# ==============================================================
print("\n--- C Parameter (Regularization) ---")
for c in [0.1, 1, 10, 100]:
    evaluate_svc(kernel="rbf", C=c)

'''
--- C Parameter (Regularization) ---
Params: {'kernel': 'rbf', 'C': 0.1} --> Accuracy: 0.8237
Params: {'kernel': 'rbf', 'C': 1} --> Accuracy: 0.9305
Params: {'kernel': 'rbf', 'C': 10} --> Accuracy: 0.9675
Params: {'kernel': 'rbf', 'C': 100} --> Accuracy: 0.9742
'''
'''
C (Regularization strength)
Low C (e.g., 0.1): Allows wider margin → simpler model, more bias, lower accuracy.
High C (e.g., 100): Forces classifier to fit more points → higher accuracy but risk of overfitting.
'''
# ==============================================================
# 3. Effect of Gamma (for RBF, Poly, Sigmoid)
# (Gamma controls influence of a single training sample)
# ==============================================================
print("\n--- Parameter (Gamma) ---")
for g in [0.01, 0.1, 1,10]:
    evaluate_svc(kernel="rbf", gamma=g)

'''
--- Parameter (Gamma) ---
Params: {'kernel': 'rbf', 'gamma': 0.01} --> Accuracy: 0.9430
Params: {'kernel': 'rbf', 'gamma': 0.1} --> Accuracy: 0.9715
Params: {'kernel': 'rbf', 'gamma': 1} --> Accuracy: 0.5142
Params: {'kernel': 'rbf', 'gamma': 10} --> Accuracy: 0.1293'''
    
'''Gamma (Influence of support vectors)
Low gamma(0.01): Decision boundary smoother, may underfit.
High Gamma(10): very tight bounddaries = overfits training data, reduces generalization.
'''
# ==============================================================
# 4. Effect of Degree (only for polynomial kernel)
# (Higher degree = more complex decision boundary)
# ==============================================================
print("\n--- Polynomial Degree ---")
for d in [2, 3, 4, 5]:
    evaluate_svc(kernel="poly", degree=d)
'''
--- Polynomial Degree ---
Params: {'kernel': 'poly', 'degree': 2} --> Accuracy: 0.9058
Params: {'kernel': 'poly', 'degree': 3} --> Accuracy: 0.9507
Params: {'kernel': 'poly', 'degree': 4} --> Accuracy: 0.9617
Params: {'kernel': 'poly', 'degree': 5} --> Accuracy: 0.9585

'''
'''
Degree (for Polynomial kernel)
Degree=2: Simple curves, may underfit.
Degree=3 or 4: Captures more complex boundaries.
Degree≥5: Often too complex → overfits.
'''

# ==============================================================
# 5. Effect of coef0 (for poly/sigmoid kernels)
# (Controls trade-off between high-order and low-order terms)
# ==============================================================
print("\n--- coef0 Parameter ---")
for c0 in [0.0, 0.5, 1, 2]:
    evaluate_svc(kernel="poly", degree=3, coef0=c0)

'''
--- coef0 Parameter ---
Params: {'kernel': 'poly', 'degree': 3, 'coef0': 0.0} --> Accuracy: 0.9507
Params: {'kernel': 'poly', 'degree': 3, 'coef0': 0.5} --> Accuracy: 0.9537
Params: {'kernel': 'poly', 'degree': 3, 'coef0': 1} --> Accuracy: 0.9553
Params: {'kernel': 'poly', 'degree': 3, 'coef0': 2} --> Accuracy: 0.9557
'''

'''
coef0 (for poly/sigmoid kernals)
Balance high-order vs low-order polynomial terms.
Increasing coef0 makes rely more on higher degree feauters
'''








