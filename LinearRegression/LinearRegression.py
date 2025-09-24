# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 16:45:17 2025

@author: Ashlesha
"""

import pandas as pd
import numpy as np
import seaborn as sns

wcat = pd.read_csv("C:/Data-Science/9-Classification/LinearRegression/wc-at.csv")

# EDA
wcat.info()
wcat.describe()

# Average waist is 91.90 and min is 63.50 and max is 121
# Average AT is 101.89 and min is 11.44 and max is 253

import matplotlib.pyplot as plt

sns.displot(wcat.AT)
# Data is normal but right skewed
plt.boxplot(wcat.AT)
# No outliers but right skewed

sns.displot(wcat.Waist)
# Data is normal bimodal
plt.boxplot(wcat.Waist)
# No outliers but right skewed

##############
# Bivariant analysis
plt.scatter(x=wcat.Waist, y=wcat.AT)
# Data is linearly scattered, direction positive, strength: poor

# Now let us check the correlation coefficient:
np.corrcoef(wcat.Waist, wcat.AT)
# The correlation coefficient is 0.8185 < 0.85 hence the correlation

# Let us check the direction of correlation
cov_output = np.cov(wcat.Waist, wcat.AT)[0, 1]
cov_output
# 635.91, it is positive means correlation will be positive

####################
# Let us apply to various models and check the feasibility
import statsmodels.formula.api as smf

# First simple linear model
model = smf.ols('AT ~ Waist', data=wcat).fit()
# Y is AT and X is Waist
model.summary()
# R-squared = 0.67 < 0.85, there is scope of improvement
# p = 0.00 < 0.05 hence acceptable
# beta_0 = -215.98
# beta_1 = 3.45

'''
Goal of the Model  
We are trying to predict AT (Dependent Variable) using Waist (Independent Variable)

1. Model Fit (Goodness of fit)  
R-squared = 0.670  
→ About 67% of the variation in AT is explained by Waist.  
   This means the model fits the data fairly well.

Adjusted R-squared = 0.667  
→ Slightly adjusted for number of predictors; still close to 67%.

F-statistic: This value measures how well your overall regression model fits the data.  
A higher F-value means your model explains much more variance than a model with no predictors.

F-statistic = 217.3 and Prob(F) = 1.62e-27 (≈ 0)  
→ Model is statistically significant (better than a model with no predictors)

| **Intercept** | -215.98         | When Waist = 0, AT is -215.98 (not realistic but required for model)  
| **Waist**     | +3.4589         | For **every 1 unit increase in Waist, AT increases by 3.46 units**
P-values for both = 0.000 → ✔ Both are statistically significant.

3. Residual Analysis  
The Durbin-Watson statistic tests whether the residuals (errors) from the regression are autocorrelated.

| DW Value   | What it Means                            |  
| ---------- | ----------------------------------------- |  
| **≈ 2**     | ✅ No autocorrelation (Ideal)             |  
| **1.5**     | ⚠ Positive autocorrelation (bad)         |  
| **> 2.5**   | ⚠ Negative autocorrelation (also bad)    |  
| **1.5-2.5** | 👍 Acceptable range (generally safe)     |

Specifically, it checks for autocorrelation — a situation where errors follow a pattern.  
Durbin-Watson = 1.56
→ No strong autocorrelation. (2 is ideal; this is close enough.)

In One Sentence:  
“Waist has a strong and significant impact on AT, and this model explains it well.”
'''
pred1 = model.predict(pd.DataFrame(wcat.Waist))  
pred1

###############################  
# Regression line  
plt.scatter(wcat.Waist, wcat.AT)  
plt.plot(wcat.Waist, pred1, 'r')  
plt.legend(['Actual data', 'Predicted data'])  
plt.show()

###############################  
# Error calculations  
res1 = wcat.AT - pred1  
np.mean(res1)
res_sqr1 = res1 * res1  
mse1 = np.mean(res_sqr1)  
rmse1 = np.sqrt(mse1)  
rmse1  
# 32.76  
# Lower RMSE = better model performance

###############################  
# Let us try another model  
# x = log(Waist)  
plt.scatter(x = np.log(wcat.Waist), y = wcat.AT)  
# Data is linearly scattered, direction positive, strength: poor  
# Now let us check the correlation coefficient  
np.corrcoef(np.log(wcat.Waist), wcat.AT)  
# The correlation coefficient is 0.8217 < 0.85 hence the correlation is acceptable  
# r = 0.8217  

model2 = smf.ols('AT ~ np.log(Waist)', data = wcat).fit()  
# Y is AT and X = log(Waist)  
model2.summary()

# R-squared = 0.675 < 0.85, there is scope of improvement  
# p = 0 < 0.05 hence acceptable  
# beta0 = -1328.3420  
# beta1 = np.log(Waist)   317.1356
'''
please ref word file OLS_MODEL_COMPARISON_1 comparision of 2 models from
'''

# Prediction using model 2
pred2 = model.predict(pd.DataFrame(wcat.Waist))
pred2

###########################
# Regression line
plt.scatter(np.log(wcat.Waist), wcat.AT)
plt.plot(np.log(wcat.Waist), pred1, 'r')
plt.legend(['Actual data', 'predicted data_model2'])
plt.show()

###########################
# Error calculations
res2 = wcat.AT - pred2
np.mean(res1)
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2
#32.76
#there are no significant change as r=0.821,RSquare=0.675 and RMSE=32.76
#Hence let us try another model
###########################

# Now let us make logY and X as is
plt.scatter(x=(wcat.Waist), y=np.log(wcat.AT))
# Data is linearly scattered, direction positive, strength: poor

# Now let us check the correlation coefficient
np.corrcoef(wcat.Waist, np.log(wcat.AT))
# The correlation coefficient is 0.8409 < 0.85 hence the correlation is moderate
# r = 0.8409

model3 = smf.ols('np.log(AT) ~ Waist', data=wcat).fit()
# Y is log(AT) and X = Waist

model3.summary()
# R-squared = 0.707 < 0.85, there is scope of improvement
# p = 0.002 < 0.05 hence acceptable
# beta0 = 0.7410
# beta1 = 0.0403

pred3 = model3.predict(pd.DataFrame(wcat.Waist))
pred3_at = np.exp(pred3)
pred3_at

#############################
# Regression line
plt.scatter(wcat.Waist, np.log(wcat.AT))
plt.plot(wcat.Waist, pred3, 'r')
plt.legend(['Predicted line', 'Observed data_model3'])
plt.show()

#############################
# Error calculations
res3 = wcat.AT - pred3_at

res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3
# 38.52

# There are no significant change as r = 0.8409, R-squared = 0.707 and RMSE = 38.52
# Hence let us try another model

# Now let us make Y=log(AT) and X=Waist, X*X=Waist.Waist
# Polynomial model
# Here r can not be calculated
model4 = smf.ols('np.log(AT) ~ Waist + I(Waist * Waist)', data=wcat).fit()

model4.summary()
# R-squared = 0.779 < 0.85, there is scope of improvement
# p = 0.000 < 0.05 hence acceptable
# beta0 = -7.8241
# beta1 =  0.2289

pred4 = model4.predict(pd.DataFrame(wcat.Waist))
pred4
pred4_at = np.exp(pred4)
pred4_at

# Regression line
plt.scatter(wcat.Waist, np.log(wcat.AT))
plt.plot(wcat.Waist, pred4, 'r')
plt.legend(['Predicted Line', 'Observed data_model3'])
plt.show()

# Error calculations
res4 = wcat.AT - pred4_at
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4
# 32.24

# Among all the models, model4 is the best
data = {"model": pd.Series(["SLR", "Log_model", "Exp_model", "Poly_model"])}
data
table_rmse = pd.DataFrame(data)
table_rmse

# We have to generalize the best model
from sklearn.model_selection import train_test_split
train, test = train_test_split(wcat, test_size=0.2)
plt.scatter(train.Waist, np.log(train.AT))
plt.scatter(test.Waist, np.log(test.AT))
final_model=smf.ols('np.log(AT)~Waist+I(Waist*Waist)',data=wcat).fit()
#y is log(AT) and x=waist
final_model.summary()
#R-sqarred=0.779<0.85, there is scope of improvement
#p=0.000<0.05 heance acceptable
#bita-0=-7.8241
# beta1 = 0.2289
test_pred = final_model.predict(pd.DataFrame(test))
test_pred_at = np.exp(test_pred)
test_pred_at

#########################
train_pred = final_model.predict(pd.DataFrame(train))
train_pred_at = np.exp(train_pred)
train_pred_at

#########################
# Evaluation on test data
test_err = test.AT - test_pred_at
test_sqr = test_err * test_err
test_mse = np.mean(test_sqr)
test_rmse = np.sqrt(test_mse)
test_rmse

#########################
# Evaluation on train data
train_res = train.AT - train_pred_at
train_sqr = train_res * train_res
train_mse = np.mean(train_sqr)
train_rmse = np.sqrt(train_mse)
train_rmse


#########################
# test_rmse > train_rmse
# The model is overfit





