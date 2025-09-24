# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 15:58:23 2025

@author: Ashlesha
"""

import pandas as pd
walmart = pd.read_csv("c:/Data-Science/10-Time/Walmart Footfalls Raw.CSV")
month = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# in wal mart data we have han-1991 in 0 th columns, we need only first
# 3 letter
# example-jan from each cell
p = walmart["Month"][0]
p[0:3]
#before we will extract, let us create new column called month to 
# store extracted values
walmart['Month']=0
#you can check the dataframe with month name  with all values 0
#the  total rocords are 159 in walmart
for i in range(159):
    p=walmart["Month"][i]
    walmart["months"][i]=p[0:3]
    #for all these months create dummy variables
    
month_dummies=pd.DataFrame(pd.get_dummies(walmart['Month']))
#now let us  concatente these dummy values to dataframe
walmart1=pd.concat([walmart,month_dummies],axis=1)
#you can check  the  dataframe walmart1

#similarly we need to creat column t

import numpy as np
walmart1['t']=np.arange(1,160 )
walmart1['t_squared']=walmart1['t']*walmart1['t']
walmart1['log_footfalls']=np.log(walmart1['Footfalls'])
walmart1.columns
#now let us check  the  visualsof the footfall

walmart1.Footfalls.plot()
#you will get  exponential trend  with  first decreasingand  then increasing
# wehave  to  forecast footfalls in next 12 months, hence  horizon=12, even
#season=12,so validating  data will be 12 and  training  will 159-12=147

Train=walmart1.head(147)
Test=walmart1.tail(12)
#now  let us apply lines regression
import statsmodels.formula.api as smf
# linear model

linear_model = smf.ols("Footfalls ~ t", data=Train).fit()
pred_linear = pd.Series(linear_model.predict(Test)) 
rmse_linear = np.sqrt(np.mean((np.array(Test['Footfalls']) - np.array(pred_linear))**2))
rmse_linear
# 209.92559265462572

##Quadratic model
Quad = smf.ols("Footfalls ~ t + t_squared", data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t", "t_squared"]]))  # Fixed column name here
rmse_Quad = np.sqrt(np.mean((np.array(Test['Footfalls']) - np.array(pred_Quad)) ** 2))
rmse_Quad
#137.15462741356146

## ADDITIVE SEASONALITY
add_sea = smf.ols('Footfalls ~ Jan + Feb + Mar + Apr + May + Jun + Jul + Aug + Sep + Oct + Nov', data=Train).fit()
add_sea.summary()
pred_add_sea = pd.Series(add_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Footfalls']) - np.array(pred_add_sea))**2))
rmse_add_sea

## Multiplicative seasonality model
mul_sea = smf.ols("log_Footfalls ~ Jan + Feb + Mar + Apr + May + Jun + Jul + Aug + Sep + Oct + Nov", data=Train).fit()
pred_mul_sea = pd.Series(mul_sea.predict(Test))
rmse_mul_sea = np.sqrt(np.mean((np.array(Test['Footfalls']) - np.array(np.exp(pred_mul_sea)))**2))
rmse_mul_sea



################ Additive seasonality with quadratic trend ################

add_sea_quad = smf.ols('Footfalls ~ t + t_squared + Jan + Feb + Mar + Apr + May + Jun + Jul + Aug + Sep + Oct + Nov', data=Train).fit()

pred_add_sea_quad = pd.Series(add_sea_quad.predict(Test[['Jan','Feb','Mar', 'Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_squared']]))

rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Footfalls']) - 
                                     np.array(pred_add_sea_quad))**2))

rmse_add_sea_quad

## Multiplicative seasonality linear model
mul_add_sea = smf.ols("log_Footfalls ~ t + Jan + Feb + Mar + Apr + May + Jun + Jul + Aug + Sep + Oct + Nov",
                      data=Train).fit()

pred_mul_add_sea = pd.Series(mul_add_sea.predict(Test))

rmse_mul_add_sea = np.sqrt(np.mean((np.array(Test['Footfalls']) - np.array(np.exp(pred_mul_add_sea)))**2))

rmse_mul_add_sea

### let us create a dataframe and add all these rmse_values
data = {"Model": pd.Series(['rmse_Linear','rmse_Exp','rmse_Quad','rmse_add_sea','rmse_add_sea_quad','rmse_mul_add_sea'])}
data


# Predict future data using the best model
predict_data = pd.read_excel("C:/Data-Science/10-Time/Predict_new.xlsx")

# Assuming best model is: additive seasonality with quadratic trend
model_full = smf.ols('Footfalls ~ t + t_squared + Jan + Feb + Mar + Apr + May + Jun + Jul + Aug + Sep + Oct + Nov', data=walmart1).fit()

pred_new = pd.Series(model_full.predict(predict_data))
print(pred_new)

# Add predictions to the DataFrame
predict_data["forecasted_Footfalls"] = pred_new
