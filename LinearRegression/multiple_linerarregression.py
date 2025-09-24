# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 15:50:39 2025

@author: Ashlesha
"""

# multiple correlation regression analysis with preprocessing
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler

# Load data
cars = pd.read_csv("c:/Data-Science/9-Classification/LinearRegression/cars.csv")

# Exploratory data analysis
#1. Measure the central tendency
#2. Measure the dispersion
#3. Third moment business decision
#4. Fourth moment business decision
#5. probability distribution
#6. Graphical representation (Histogram, Boxplot)
cars.describe()
# Graphical representation
import matplotlib.pyplot as plt
import numpy as np

sns.distplot(cars.HP)
# data is right skewed
plt.boxplot(cars.HP)
# There are several outliers in HP column
# similar operations are expected for other three columns

sns.distplot(cars.MPG)
# data is slightly left distributed
'''
Right-skewed (positively skewed): The tail is longer on the right (higher values are stretched out).

Left-skewed (negatively skewed): The tail is longer on the left (lower values are stretched out).

Symmetric: Looks like a bell curve, balanced on both sides.


'''
plt.boxplot(cars.MPG)
# There are no outliers

sns.distplot(cars.VOL)
# data is slightly left distributed
plt.boxplot(cars.VOL)

sns.distplot(cars.SP)
# data is slightly right distributed
plt.boxplot(cars.SP)
###There are several outliers
sns.distplot(cars.WT)
plt.boxplot(cars.WT)
##There are several outliers
#Now let us plot joint plot,joint plot is to show scatter plot as we
# histogram
import seaborn as sns
sns.jointplot(x=cars['HP'],y=cars['MPG'])

# now let us plot count plot
plt.figure(1,figsize=(16,10))
sns.countplot(cars['HP'])
#count plot shows how many times the each value occured
#92 HP value occured 7 times

##QQ plot
from scipy import stats
import pylab
stats.probplot(cars.MPG,dist="norm",plot=pylab)
plt.show()
# MPG data is normally distributed
# MPG data is normally distributed
# There are 10 scatter plots need to be plotted,one by one is difficult
#to plot,so we can use pair plots
import seaborn as sns
sns.pairplot(cars.iloc[:,:])

# --------- DATA PREPROCESSING ---------

# 1. Winsorization (outlier treatment)
def winsorize_series(series, lower_quantile=0.05, upper_quantile=0.95):
    lower = series.quantile(lower_quantile)
    upper = series.quantile(upper_quantile)
    return np.clip(series, lower, upper)

for col in ['HP', 'SP', 'WT']:
    cars[col] = winsorize_series(cars[col])
# 2. Log Transformation (for right-skewed columns)
for col in ['HP', 'SP']:
    cars[col] = np.log1p(cars[col])

# 3. Box-Cox Transformation (for slightly skewed columns)
for col in ['VOL', 'WT']:
    if all(cars[col] > 0):  # Required for boxcox
        cars[col], _ = stats.boxcox(cars[col])
# 4. Standardization
scaler = StandardScaler()
cars_scaled = pd.DataFrame(scaler.fit_transform(cars), columns=cars.columns)

# --------- EXPLORATORY DATA ANALYSIS ---------

# Central tendency, dispersion, skewness, kurtosis
print(cars_scaled.describe())

# Distribution plots
sns.distplot(cars_scaled.MPG)
#data is symmetrical distributed
plt.boxplot(cars_scaled.MPG)
##There are no outliers
sns.distplot(cars_scaled.VOL)
#data is symmetrical distributed
plt.boxplot(cars_scaled.VOL)
sns.distplot(cars_scaled.SP)
#data is symmetrical distributed
plt.boxplot(cars_scaled.SP)
###There are still few outliers

#There are no outliers
# Joint plot
sns.jointplot(x=cars_scaled['HP'], y=cars_scaled['MPG'])

# Count plot
plt.figure(figsize=(16, 6))
sns.countplot(x='HP', data=cars_scaled)
plt.title('Count plot of HP')

# QQ plot for normality check
stats.probplot(cars_scaled['MPG'], dist="norm", plot=plt)
plt.title('QQ Plot of MPG')
plt.show()

# Pair plot
sns.pairplot(cars_scaled)
cars_scaled.corr()
# you can check SP and HP ,p value is 0.9627 and same way
# you can check WT and VOL ,it has got 0.9794
#which is greater than 0.85
# Now although we observed strogly correlated pairs,
#still we will go for linear regression
import statsmodels.formula.api as smf
ml1=smf.ols('MPG~WT+VOL+SP+HP',data=cars_scaled).fit()
ml1.summary()
#R square value observed is 0.920>0.85
#p-values of WT and VOL is 0.385 and 0.279 which is greater than 0.05
# it me, earlier we ignored WT and VOL columns

# or delete.Instead deleting 81 entries,
#let us check row wise outliers
# identifying is there any influential value.
#To check you can use influential index
import statsmodels.api as sm
sm.graphics.influence_plot(ml1)
# 76 is the value which has got outliers
# go to data frame and check 76 th entry
# let us delete that entry
cars_new=cars_scaled.drop(cars.index[[65,76,78,70]])


# again apply regression to cars_new
ml_new=smf.ols('MPG~WT+VOL+HP+SP',data=cars_new).fit()
ml_new.summary()
#R-square value is 0.931 and p values are 0.195 ,0.286 hence not solving the
# Now next option is delete the column but
# question is which column is to be deleted
# we have already checked correlation factor r
# VOL has got -0.529 and for WT=-0.526
# WT is less hence can be deleted

# another approach is to check the collinearity
# rsquare is giving that value
# we will have to apply regression w.r.t. x1 and input as x2, x3, and x4 so on

rsq_hp = smf.ols('HP~WT+VOL+SP', data=cars_scaled).fit().rsquared
vif_hp = 1 / (1 - rsq_hp)
vif_hp
# 14.35

# VIF is variance inflation factor, calculating VIF helps to find collinearity
# thumb rule is VIF should not be greater than 10

rsq_wt = smf.ols('WT~HP+VOL+SP', data=cars_scaled).fit().rsquared
vif_wt = 1 / (1 - rsq_wt)
vif_wt
# 26.26

rsq_vol = smf.ols('VOL~HP+WT+SP', data=cars_scaled).fit().rsquared
vif_vol = 1 / (1 - rsq_vol)
vif_vol
# 25.66

rsq_sp=smf.ols('SP~HP+WT+VOL',data=cars_scaled).fit().rsquared
vif_sp=1/(1-rsq_sp)
vif_sp

#14.041
#vif_wt=26.26 and vif_vol=25.66 heance vif_wt
#is greter , thumb is vif should not be  greater than 10
#storing the value data frame
d1 = {'Variables': ['HP', 'WT', 'VOL', 'SP'], 'VIF': [vif_hp, vif_wt, vif_vol, vif_sp]}
vif_frame = pd.DataFrame(d1)
vif_frame

### let us drop WT and apply correlation to remaining three
final_ml = smf.ols('MPG~VOL+SP+HP', data=cars_scaled).fit()
final_ml.summary()
# R square is 0.919 and p values = 0.00, 0.00, 0.00 < 0.05

# prediction
pred = final_ml.predict(cars_scaled)

## QQ plot
res = final_ml.resid
sm.qqplot(res)
plt.show()

# This QQ plot is on residual which is obtained on training data
# errors are obtained on test data
import pylab
stats.probplot(res, dist="norm", plot=pylab)
plt.show()

# let us plot the residual plot ,which takes the residuals values
# and the data
sns.residplot(x=pred, y=cars.MPG, lowess=True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted VS Residual')
plt.show()

# residual plots are used to check whether the errors are
# independent or not
'''
Good Signs
The residuals (errors) are centered around zero.
They appear to be randomly scattered without a clear pattern – a good sign that
There's no strong curvature or funnel shape, which supports the assumption of

Potential Issues Noted
Outliers: A few residuals go beyond ±7 or even ±10, which may indicate outliers.
Slight clustering: There might be a bit of horizontal clustering in the middle.
'''


# let us plot the influence plot  
sm.graphics.influence_plot(final_ml)  
# we have taken cars instead car_new data ,hence 76 is reflected  
# again in influencial data

#splitting the data into train and test data
from sklearn.model_selection import train_test_split
cars_train, cars_test = train_test_split(cars_scaled, test_size=0.2)
#preparing the model on train data
model_train = smf.ols('MPG~VOL+SP+HP', data=cars_train).fit()
model_train.summary()
test_pred = model_train.predict(cars_test)
train_pred = model_train.predict(cars_train)
##test_errors
test_error = test_pred - cars_test.MPG
test_rmse = np.sqrt(np.mean(test_error * test_error))
test_rmse
#0.5467
train_error = train_pred - cars_train.MPG
train_rmse = np.sqrt(np.mean(train_error * train_error))
train_rmse
#0.2066

#This is underfitted model
#############################################
#Add Polynomial Features (to capture non-linear relationships)

from sklearn.preprocessing import PolynomialFeatures

# Create polynomial features up to degree 2

# Generate polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(cars_scaled[['HP', 'SP', 'VOL']])
'''
When you use PolynomialFeatures from scikit-learn,
it generates feature names like:
['HP', 'SP', 'VOL', 'HP^2', 'HP SP', 'HP VOL', 'SP^2', 'SP VOL', 'VOL^2']
These names include:
Spaces (HP SP)
Carets (^2)
These are not valid Python variable names in formulas used by statsmodels.form
TO fix:
    poly feature names clean = [name.replace(' ', '_').replace('^', '') for n
'''

# Fix column names to be formula-safe (no ^, spaces, etc.)
poly_feature_names = poly.get_feature_names_out(['HP', 'SP', 'VOL'])
poly_feature_names_clean = [name.replace(' ', '_').replace('^', '') for name in poly_feature_names]

# Create DataFrame
poly_df = pd.DataFrame(poly_features, columns=poly_feature_names_clean)

# Add the target variable
poly_df['MPG'] = cars_scaled['MPG']

# Rebuild the Model with Polynomial Features
# Fit model on polynomial features
# Fit polynomial regression model
formula = 'MPG ~ ' + ' + '.join(poly_df.columns.difference(['MPG']))
poly_model = smf.ols(formula=formula, data=poly_df).fit()
print(poly_model.summary())

# R Square is 0.9 > 0.85 but p value of HP2=0.502,
# p value of HP_SP is 0.270
'''
You're building a polynomial regression model (degree=2), and the model shows
# R-squared = 0.90, which is very good
# But, some p-values are greater than 0.05:
# HP2 → p = 0.502
# HP_SP → p = 0.270
# Is it Acceptable?
# Short Answer:
# No - p-values > 0.05 generally mean the term is not statistically significant
# You have a domain-specific or theoretical reason to keep it.
# That is the reason understanding the business is very important

# Option 1: Remove Insignificant Terms
# Start by dropping features with high p-values one at a time, and refit
# Simpler model
# Similar or slightly lower R²
# More statistically valid coefficients
# Example: Drop HP2 and HP_SP from your formula.
# Option 2: Use stepwise feature selection

'''
# 3. Split into train and test sets
train_poly, test_poly = train_test_split(poly_df, test_size=0.2, random_state=1)
X_train = train_poly.drop(columns=['MPG'])
y_train = train_poly['MPG']
X_test = test_poly.drop(columns=['MPG'])
y_test = test_poly['MPG']

import statsmodels.api as sm
# Add constant for intercept
X_train_const = sm.add_constant(X_train)
X_test_const = sm.add_constant(X_test)

# Fit the model
poly_model = sm.OLS(y_train, X_train_const).fit()
# 5. Predict
train_pred = poly_model.predict(X_train_const)
test_pred = poly_model.predict(X_test_const)

# 6. Compute RMSE
train_rmse = np.sqrt(np.mean((train_pred - y_train) ** 2))
test_rmse = np.sqrt(np.mean((test_pred - y_test) ** 2))

# 7. Print Results
print("Train RMSE (Polynomial):", round(train_rmse, 4))
print("Test RMSE (Polynomial):", round(test_rmse, 4))
# Train RMSE (Polynomial):  0.217
# Test RMSE (Polynomial): 0.3028

# Train RMSE < Test RMSE | Normal case, well-trained
