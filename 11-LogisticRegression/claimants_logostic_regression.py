# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 16:09:40 2025

@author: Ashlesha
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report

claimants = pd.read_csv("c:/Data-Science/9-Classification/LogisticRegression/claimants.csv")

# The 0th column is CASENUM which is not useful, hence drop the column
c1 = claimants.drop('CASENUM', axis=1)

c1.head()
'''
      ATTORNEY  CLMSEX  CLMINSUR  SEATBELT  CLMAGE    LOSS
0         0     0.0       1.0       0.0    50.0  34.940
1         1     1.0       0.0       0.0    18.0   0.891
2         1     0.0       1.0       0.0     5.0   0.330
3         0     0.0       1.0       1.0    31.0   0.037
4         1     0.0       1.0       0.0    30.0   0.038
'''
c1.describe()
'''
          ATTORNEY       CLMSEX  ...       CLMAGE         LOSS
count  1340.000000  1328.000000  ...  1151.000000  1340.000000
mean      0.488806     0.558735  ...    28.414422     3.806307
std       0.500061     0.496725  ...    20.304451    10.636903
min       0.000000     0.000000  ...     0.000000     0.000000
25%       0.000000     0.000000  ...     9.000000     0.400000
50%       0.000000     1.000000  ...    30.000000     1.069500
75%       1.000000     1.000000  ...    43.000000     3.781500
max       1.000000     1.000000  ...    95.000000   173.604000
'''

# Let us check the null values
c1.isna().sum()
'''
ATTORNEY      0
CLMSEX       12
CLMINSUR     41
SEATBELT     48
CLMAGE      189
LOSS          0
dtype: int64
'''
# There are several null values around 290
# Let us use mean imputation for continuous data and mode imputation for discrete data
# For discrete data
mean_value = c1.CLMAGE.mean()
mean_value
#28.414422241529106

c1.CLMAGE = c1.CLMAGE.fillna(mean_value)
c1.CLMAGE.isna().sum()
#0

# For discrete value like CLMSEX we need to use mode imputation
mode_CLMSEX = c1.CLMSEX.mode()
mode_CLMSEX
#0    1.0
#Name: CLMSEX, dtype: float64


# Here if you will observe the output it is 0 1 i.e.
# mode_CLMSEX[0] = 0, mode_CLMSEX[1] = 1, we are passing mode_CLMSEX[0]
c1.CLMSEX = c1.CLMSEX.fillna(mode_CLMSEX[0])
c1.CLMSEX.isna().sum()
# 0

# CLMINSUR
mode_INSUR = c1['CLMINSUR'].mode()
mode_INSUR
# 0    1.0
# Name: CLMINSUR, dtype: float64

c1.CLMINSUR = c1.CLMINSUR.fillna(mode_INSUR[0])
c1.CLMINSUR.isna().sum()
# 0

#seat Belt

mode_SB=c1['SEATBELT'].mode()
mode_SB
#0    0.0
#Name: SEATBELT, dtype: float64

c1.SEATBELT = c1.SEATBELT.fillna((mode_SB)[0])
c1.SEATBELT.isna().sum()
#  0
c1.isna().sum()
'''ATTORNEY    0
    CLMSEX      0
    CLMINSUR    0
    SEATBELT    0
    CLMAGE      0
    LOSS        0'''

############################
# model building
logit_model = sm.logit('ATTORNEY ~ CLMAGE + LOSS + CLMINSUR + CLMSEX + SEATBELT', data=c1).fit()
'''ptimization terminated successfully.
         Current function value: 0.609131
         Iterations 7'''
logit_model.summary()
'''
logit_model.summary()
 
<class 'statsmodels.iolib.summary.Summary'>
"""
                           Logit Regression Results                           
==============================================================================
Dep. Variable:               ATTORNEY   No. Observations:                 1340
Model:                          Logit   Df Residuals:                     1334
Method:                           MLE   Df Model:                            5
Date:                Thu, 14 Aug 2025   Pseudo R-squ.:                  0.1209
Time:                        16:26:58   Log-Likelihood:                -816.24
converged:                       True   LL-Null:                       -928.48
Covariance Type:            nonrobust   LLR p-value:                 1.620e-46
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -0.1493      0.226     -0.660      0.509      -0.592       0.294
CLMAGE         0.0066      0.003      2.058      0.040       0.000       0.013
LOSS          -0.3228      0.029    -10.962      0.000      -0.381      -0.265
CLMINSUR       0.5284      0.210      2.516      0.012       0.117       0.940
CLMSEX         0.3204      0.120      2.674      0.008       0.086       0.555
SEATBELT      -0.6718      0.522     -1.286      0.198      -1.696       0.352
==============================================================================
"""

'''
logit_model.summary2()
'''
<class 'statsmodels.iolib.summary2.Summary'>
"""
                         Results: Logit
=================================================================
Model:              Logit            Method:           MLE       
Dependent Variable: ATTORNEY         Pseudo R-squared: 0.121     
Date:               2025-08-14 16:27 AIC:              1644.4709 
No. Observations:   1340             BIC:              1675.6734 
Df Model:           5                Log-Likelihood:   -816.24   
Df Residuals:       1334             LL-Null:          -928.48   
Converged:          1.0000           LLR p-value:      1.6204e-46
No. Iterations:     7.0000           Scale:            1.0000    
------------------------------------------------------------------
              Coef.   Std.Err.     z      P>|z|    [0.025   0.975]
------------------------------------------------------------------
Intercept    -0.1493    0.2260   -0.6604  0.5090  -0.5922   0.2937
CLMAGE        0.0066    0.0032    2.0583  0.0396   0.0003   0.0128
LOSS         -0.3228    0.0294  -10.9615  0.0000  -0.3805  -0.2651
CLMINSUR      0.5284    0.2100    2.5159  0.0119   0.1168   0.9400
CLMSEX        0.3204    0.1198    2.6736  0.0075   0.0855   0.5552
SEATBELT     -0.6718    0.5224   -1.2860  0.1984  -1.6958   0.3521
=================================================================
'''
# let us go for prediction
pred = logit_model.predict(c1.iloc[:, 1:])
############################
# To derive ROC curve
# ROC curve has tpr on y-axis and fpr on x-axis, ideally tpr must be high
# fpr must be low
fpr, tpr, thresholds = roc_curve(c1.ATTORNEY, pred)

# To identify optimum threshold
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
optimal_threshold
#: 0.5294418043694739
#0.52944. by defualt you can take 0.5 value as a threshold
#now we want to identify if new value is given to the model, it will
#fall in which  region 0 or 1 , for that we need to derive ROC curve
#TO draw ROC Curve

import pylab as pl
i=np.arange(len(tpr))
roc=pd.DataFrame({
    'fpr':pd.Series(fpr, index=i),
    'tpr':pd.Series(tpr, index=i),
    '1-fpr':pd.Series(1-fpr,index=i),
    'tf':pd.Series(tpr - (1-fpr), index=i),
    'threshold':pd.Series(thresholds, index=i)})

#This code creates a Ddataframe called roc using pandas(pd)
#It  orgaizes various metrics related to the  receiver operating characterististc(ROC)
#into  columns. Each columns represents a specifi metric , and  the rows are  indexed by
#plot ROC curve
plt.plot(fpr,tpr)
plt.xlabel("False positive rate");plt.ylabel("True positive rate")
roc_auc = auc(fpr, tpr)
print("Area under the curve %f" % roc_auc)
# Area under the curve 0.760101
############################
# Now let us add prediction column in dataframe
c1["pred"] = np.zeros(1340)
c1.loc[pred > optimal_threshold, "pred"] = 1
# if predicted value is greater than optimal threshold then change pred column as 1

# Classification report
classification = classification_report(c1["pred"], c1["ATTORNEY"])
classification
'''
precision    recall  f1-score   support

         0.0       0.67      0.74      0.70       615
         1.0       0.76      0.69      0.72       725

    accuracy                           0.71      1340
   macro avg       0.71      0.72      0.71      1340
weighted avg       0.72      0.71      0.71      1340
'''
############################
# splitting the data into train and test data
train_data, test_data = train_test_split(c1, test_size=0.3)

# model building using train data
model = sm.logit('ATTORNEY ~ CLMAGE + LOSS + CLMINSUR + CLMSEX + SEATBELT', data=train_data).fit()
model.summary()
'''
   Logit Regression Results                           
==============================================================================
Dep. Variable:               ATTORNEY   No. Observations:                  938
Model:                          Logit   Df Residuals:                      932
Method:                           MLE   Df Model:                            5
Date:                Thu, 14 Aug 2025   Pseudo R-squ.:                  0.1617
Time:                        16:43:46   Log-Likelihood:                -545.04
converged:                       True   LL-Null:                       -650.14
Covariance Type:            nonrobust   LLR p-value:                 1.869e-43
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     -0.0419      0.270     -0.155      0.877      -0.571       0.487
CLMAGE         0.0088      0.004      2.235      0.025       0.001       0.016
LOSS          -0.4283      0.040    -10.706      0.000      -0.507      -0.350
CLMINSUR       0.4933      0.250      1.975      0.048       0.004       0.983
CLMSEX         0.4615      0.148      3.124      0.002       0.172       0.751
SEATBELT      -0.5448      0.670     -0.813      0.416      -1.859       0.769
==============================================================================
'''
model.summary2()
'''
       Results: Logit
=================================================================
Model:              Logit            Method:           MLE       
Dependent Variable: ATTORNEY         Pseudo R-squared: 0.162     
Date:               2025-08-14 16:43 AIC:              1102.0807 
No. Observations:   938              BIC:              1131.1432 
Df Model:           5                Log-Likelihood:   -545.04   
Df Residuals:       932              LL-Null:          -650.14   
Converged:          1.0000           LLR p-value:      1.8691e-43
No. Iterations:     8.0000           Scale:            1.0000    
------------------------------------------------------------------
              Coef.   Std.Err.     z      P>|z|    [0.025   0.975]
------------------------------------------------------------------
Intercept    -0.0419    0.2697   -0.1554  0.8765  -0.5706   0.4868
CLMAGE        0.0088    0.0039    2.2346  0.0254   0.0011   0.0165
LOSS         -0.4283    0.0400  -10.7057  0.0000  -0.5068  -0.3499
CLMINSUR      0.4933    0.2498    1.9749  0.0483   0.0037   0.9830
CLMSEX        0.4615    0.1477    3.1242  0.0018   0.1720   0.7510
SEATBELT     -0.5448    0.6705   -0.8125  0.4165  -1.8588   0.7693
=================================================================

"""
'''
# AIC is 1157
# prediction on test data
test_pred = model.predict(test_data)
test_data["test_pred"] = np.zeros(402)

# taking threshold value as optimal threshold value
test_data.loc[test_pred > optimal_threshold, "test_pred"] = 1

# Confusion matrix
confusion_matrix = pd.crosstab(test_data.test_pred, test_data.ATTORNEY)
confusion_matrix
'''
ATTORNEY     0    1
test_pred          
0.0        138   47
1.0         74  143
'''
# Accuracy calculation
accuracy_test = (130 + 146) / 402
accuracy_test
# 0.6865671641791045

# Classification report
classification_test = classification_report(test_data["test_pred"], test_data["ATTORNEY"])
classification_test
'''
  precision    recall  f1-score   support

         0.0       0.65      0.75      0.70       185
         1.0       0.75      0.66      0.70       217

    accuracy                           0.70       402
   macro avg       0.70      0.70      0.70       402
weighted avg       0.71      0.70      0.70       402

'''
# ROC curve and AUC
fpr, tpr, threshold = metrics.roc_curve(test_data["ATTORNEY"], test_pred)

# plot of ROC
plt.plot(fpr, tpr)
plt.xlabel("False positive rate");plt.ylabel("True positive rate")
roc_auc_test=metrics.auc(fpr,tpr)
roc_auc_test
#0.7333416087388283
# prediction on train data
train_pred = model.predict(train_data.iloc[:, 1:])

# creating new column
train_data["train_pred"] = np.zeros(938)
train_data.loc[train_pred > optimal_threshold, "train_pred"] = 1

# confusion matrix
confusion_matrix = pd.crosstab(train_data.train_pred, train_data.ATTORNEY)
confusion_matrix
'''
ATTORNEY      0    1
train_pred          
0.0         313  103
1.0         160  362
'''
# Accuracy test
accuracy_train = (334 + 335) / 938
accuracy_train
# 0.7132196162046909

# classification report
classification_train = classification_report(train_data.train_pred, train_data.ATTORNEY)
classification_train
'''
              precision    recall  f1-score   support

         0.0       0.66      0.75      0.70       416
         1.0       0.78      0.69      0.73       522

    accuracy                           0.72       938
   macro avg       0.72      0.72      0.72       938
weighted avg       0.73      0.72      0.72       938
'''
# ROC_AUC curve
roc_auc_train = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
