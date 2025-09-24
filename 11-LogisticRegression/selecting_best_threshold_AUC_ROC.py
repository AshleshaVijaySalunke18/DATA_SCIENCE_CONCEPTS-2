# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 16:19:10 2025

@author: Ashlesha
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# 1. Example "medical" data
data = pd.DataFrame({
    "Patient": ["A", "B", "C", "D", "E", "F", "G", "H"],
    "Actual": [1, 0, 1, 0, 1, 0, 0, 1],  # 1 = has disease, 0 = healthy
    "Predicted_Prob": [0.92, 0.72, 0.51, 0.12, 0.88, 0.45, 0.30, 0.65]
})

print("Medical Test Predictions:")
print(data)
'''
     Patient  Actual  Predicted_Prob
0       A       1            0.92
1       B       0            0.72
2       C       1            0.51
3       D       0            0.12
4       E       1            0.88
5       F       0            0.45
6       G       0            0.30
7       H       1            0.65
'''

# 2. Calculate ROC curve
fpr, tpr, thresholds = roc_curve(data["Actual"], data["Predicted_Prob"])
auc_score = roc_auc_score(data["Actual"], data["Predicted_Prob"])

# 3. Find best threshold (Youden's J statistic = TPR - FPR)
j_scores = tpr - fpr
j_scores
#array([0.  , 0.25, 0.5 , 0.25, 0.75, 0.  ])

best_idx = np.argmax(j_scores)
best_idx
#4

best_threshold = thresholds[best_idx]
best_threshold
#0.51

best_tpr, best_fpr = tpr[best_idx], fpr[best_idx]
best_tpr, best_fpr
#(1.0, 0.25)

print(f"\nBest Threshold = {best_threshold:.2f}")
print(f"True Positive Rate (Sensitivity) = {best_tpr:.2f}")
print(f"False Positive Rate = {best_fpr:.2f}")
# Best Threshold = 0.51
# True Positive Rate (Sensitivity) = 1.00
# False Positive Rate = 0.25

# 4. Classify using best threshold
data["Predicted_Label"] = (data["Predicted_Prob"] >= best_threshold).astype(int)
print("\nClassification using Best Threshold:")
print(data[["Patient", "Actual", "Predicted_Prob", "Predicted_Label"]])
'''
Best Threshold:
     Patient  Actual  Predicted_Prob  Predicted_Label
0       A       1            0.92                1
1       B       0            0.72                1
2       C       1            0.51                1
3       D       0            0.12                0
4       E       1            0.88                1
5       F       0            0.45                0
6       G       0            0.30                0
7       H       1            0.65                1

'''
# 5. Plot ROC curve with best  point highlighted
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, marker='o', label=f"Model (AUC = {auc_score:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess (AUC = 0.5)")

# Highlight best threshold point
plt.scatter(best_fpr, best_tpr, color='red', s=100,
            label=f"Best Threshold = {best_threshold:.2f}")
plt.text(best_fpr + 0.02, best_tpr - 0.05, f"T={best_threshold:.2f}", color='red')

plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.title("ROC Curve - Medical Test Example")
plt.legend()
plt.grid(True)
plt.show()