# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 15:47:10 2025

@author: Ashlesha
"""

# multiple correlation regression analysis
import pandas as pd
import numpy as np
import seaborn as sns

cars = pd.read_csv("c:/Data-Science/9-Classification/LinearRegression/cars.csv")

# Exploratory data analysis
#1. Measure the central tendency
#2. Measure the dispersion
#3. Third moment business decision
#4. Fourth moment business decision
#5. Probability distribution
#6. Graphical representation (Histogram, Boxplot)

cars.describe()

# Graphical representation
import matplotlib.pyplot as plt
import numpy as np

sns.histplot(cars.HP, kde=True)
# data is right skewed
plt.boxplot(cars.HP)
plt.show()
# There are several outliers in HP columns
# similar operations are expected for other three columns

sns.histplot(cars.MPG, kde=True)
# data is slightly left distributed
plt.boxplot(cars.MPG)
plt.show()
# There are no outliers

sns.histplot(cars.VOL, kde=True)
# data is slightly left distributed
plt.boxplot(cars.VOL)
plt.show()

sns.histplot(cars.SP, kde=True)
# data is slightly right distributed
plt.boxplot(cars.SP)
plt.show()
# There are several outliers

sns.histplot(cars.WT, kde=True)
plt.boxplot(cars.WT)
plt.show()
# Now let us plot joint plot. Joint plot is to show scatter plot as well as distribution.
# histogram
import seaborn as sns
sns.jointplot(x=cars['HP'], y=cars['MPG'])

# now let us plot count plot
plt.figure(1, figsize=(16,10))
# Use barplot since HP is numeric
hp_counts = cars['HP'].value_counts().reset_index()
hp_counts.columns = ['HP', 'Count']
sns.barplot(data=hp_counts, x='HP', y='Count')
# count plot shows how many times the each value occurred
# 92 HP value occurred 7 times

# QQ plot
from scipy import stats
import pylab
stats.probplot(cars.MPG, dist="norm", plot=pylab)
plt.show()
# MPG data is normally distributed
# There are 10 scatter plots need to be plotted, one by one is diff
# to plot, so we can use pair plots
import seaborn as sns
sns.pairplot(cars.iloc[:,:])
# Linearity: direction : and strength:
# if you can check the collinearity problem between the input variables
# you can check plot between SP and HP, they are strongly correlated
# same way you can check WT and VOL, it is also strongly correlated

# now let us check r value between variables
cars.corr()
# if you can check SP and HP, r value is 0.97 and same way
# you can check WT and VOL, it has got 0.999
#which is greater
# Now although we observed strongly correlated pairs,
# still we will go for linear regression
import statsmodels.formula.api as smf
ml1 = smf.ols('MPG~WT+VOL+SP+HP', data=cars).fit()
ml1.summary()
# R square value observed is 0.7710≈85
# p-values of WT and VOL is 0.814 and 0.556 which is very high
# it means it is greater than 0.05, WT and VOL columns 
# we need to ignore 
# or delete. Instead deleting 81 entries,
# let us check row wise outliers
# identifying is there any influential value.
# To check you can use influential index
import statsmodels.api as sm
sm.graphics.influence_plot(ml1)
# 76 is the value which has got outliers
# go to data frame and check 76 th entry
# let us delete that entry
cars_new = cars.drop(cars.index[[76]])
# again apply regression to cars_new
ml_new = smf.ols('MPG~WT+VOL+HP+SP', data=cars_new).fit()
ml_new.summary()
# R-square value is 0.819 but p values are same, hence not solving the issue.
# Now next option is delete the column but 
# question is which column is to be deleted
# we have already checked correlation factor r
# VOL has got -0.529 and for WT=-0.526
# WT is less hence can be deleted

# another approach is to check the collinearity,
# rsquare is giving that value,
# we will have to apply regression w.r.t. x1 and input
# as x2, x3 and x4 so on so forth
rsq_hp = smf.ols('HP~WT+VOL+SP', data=cars).fit().rsquared
vif_hp = 1 / (1 - rsq_hp)
vif_hp
#19.92
#VIF is variance influential factor , calculating VIF helps ho final
#of X1 w.r.t X2,X3 and x4
rsq_wt = smf.ols('WT~HP+VOL+SP', data=cars).fit().rsquared
vif_wt = 1 / (1 - rsq_wt)
vif_wt
#639.53

rsq_vol = smf.ols('VOL~HP+WT+SP', data=cars).fit().rsquared
vif_vol = 1 / (1 - rsq_vol)
vif_vol
# 638.80

rsq_sp = smf.ols('SP~HP+WT+VOL', data=cars).fit().rsquared
vif_sp = 1 / (1 - rsq_sp)
vif_sp
# 20.00

## vif_wt=639.53 and vif_vol=638.80 hence vif_wt 
## is greater , thumb rule is vif should not be greater than 10

# storing the values in dataframe
d1 = {'Variables': ['HP', 'WT', 'VOL', 'SP'],
      'VIF': [vif_hp, vif_wt, vif_vol, vif_sp]}
vif_frame = pd.DataFrame(d1)
vif_frame


'''
 Variables         VIF
0        HP   19.926589
1        WT  639.533818
2       VOL  638.806084
3        SP   20.007639
'''

### let us drop WT and apply correlation to remaining three
final_ml = smf.ols('MPG~VOL+SP+HP', data=cars).fit()
final_ml.summary()
# R square is 0.770 and p values 0.00, 0.012 < 0.05

# prediction
pred = final_ml.predict(cars)
pred
# QQ plot
res = final_ml.resid
sm.qqplot(res)
plt.show()

# This QQ plot is on residual which is obtained on training data
# errors are obtained on test data
stats.probplot(res, dist="norm", plot=pylab)
plt.show()

# let us plot the residual plot, which takes the residuals values and the data
sns.residplot(x=pred, y=cars.MPG, lowess=True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted VS Residual')
plt.show()
# residual plots are used to check whether the errors are independent or not

# let us plot the influence plot
sm.graphics.influence_plot(final_ml)
# we have taken cars instead car_new data , hence 76 is reflected
# again in influencial data

# splitting the data into train and test data
from sklearn.model_selection import train_test_split
cars_train, cars_test = train_test_split(cars, test_size=0.2)

# preparing the model on train data
model_train = smf.ols('MPG~VOL+SP+HP', data=cars_train).fit()
model_train.summary()

test_pred = model_train.predict(cars_test)
train_pred = model_train.predict(cars_train)

# test errors
test_error = test_pred - cars_test.MPG
test_rmse = np.sqrt(np.mean(test_error * test_error))
test_rmse
# 4.0765
train_error = train_pred - cars_train.MPG
train_rmse = np.sqrt(np.mean(train_error * train_error))
train_rmse
#4.4762

'''
| Scenario                 | Interpretation                        |
|-------------------------|----------------------------------------|
| Train RMSE < Test RMSE  | Normal case, well-trained              |
| Train RMSE ≈ Test RMSE  | Ideal generalization                   |
| Train RMSE >> Test RMSE | Possible underfitting or odd test set  |
| Train RMSE << Test RMSE | Overfitting                            |
'''