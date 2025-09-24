# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 16:02:51 2025

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

# 2. Try classification at different thresholds
def classify_and_show(threshold):
    data["Predicted_Label"] = (data["Predicted_Prob"] >= threshold).astype(int)
    print(f"\nThreshold = {threshold}")
    print(data[["Patient", "Actual", "Predicted_Prob", "Predicted_Label"]])

#try different thresholds
for t in [0.5,0.7,0.3]:
    classify_and_show(t)
'''

Threshold = 0.5
     Patient  Actual  Predicted_Prob  Predicted_Label
0       A       1            0.92                1
1       B       0            0.72                1
2       C       1            0.51                1
3       D       0            0.12                0
4       E       1            0.88                1
5       F       0            0.45                0
6       G       0            0.30                0
7       H       1            0.65                1

Threshold = 0.7
     Patient  Actual  Predicted_Prob  Predicted_Label
0       A       1            0.92                1
1       B       0            0.72                1
2       C       1            0.51                0
3       D       0            0.12                0
4       E       1            0.88                1
5       F       0            0.45                0
6       G       0            0.30                0
7       H       1            0.65                0

Threshold = 0.3
    Patient  Actual  Predicted_Prob  Predicted_Label
0       A       1            0.92                1
1       B       0            0.72                1
2       C       1            0.51                1
3       D       0            0.12                0
4       E       1            0.88                1
5       F       0            0.45                1
6       G       0            0.30                1
7       H       1            0.65                1

'''

# 3. Calculate ROC curve
fpr, tpr, thresholds = roc_curve(data["Actual"], data["Predicted_Prob"])
auc_score = roc_auc_score(data["Actual"], data["Predicted_Prob"])


# 4. Plot ROC curve
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, marker='o', label=f"Model (AUC = {auc_score:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess (AUC = 0.5)")
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.title("ROC Curve - Medical Test Example")
plt.legend()
plt.grid(True)
plt.show()