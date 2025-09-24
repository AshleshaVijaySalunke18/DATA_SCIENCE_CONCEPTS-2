# -*- coding: utf-8 -*-
"""
Created on Tue May  6 08:20:49 2025

@author: Ashlesha
"""

import pandas as pd
import numpy as np
import seaborn as sns
df=pd.read_csv("c:/Data-Science/5-Data_Preprocessing/ethnic diversity.csv")
#check data types
df.dtypes
#salary data type is flot let us conver
df['Salaries'] = df['Salaries'].astype(int)
df.dtypes
#now the data type  salaries is int
#similarly age data type must be float
#preesenty it is int
df.age=df.age.astype(float)
df.dtypes
sns.histplot(df['Salaries'],kde=True)
df['Salaries']=np.log(df['log_Salaries'])
sns.histplot(df['log_Salaries'],kde=True)


###############################
#identify the duplication
#file name education
df_new=pd.read_csv("c:/Data-Science/5-Data_Preprocessing/education.csv")
duplicate=df_new.duplicated()
duplicate
sum(duplicate)
#sum(duplicate) 0
#mtcars_dup file
df_new1=pd.read_csv("c:/Data-Science/5-Data_Preprocessing/mtcars_dup.csv")
duplicate=df_new.duplicated()
duplicate
sum(duplicate)
#sum(duplicate) 3
#row 17 is duplicate of row 2 like wise you can 3 duplicate
#Record
#there is function called drop_duplicated
df_new2=df_new1.drop_duplicates()
duplicate2=df_new2.duplicated()
duplicate2
sum(duplicate2)
#IQR calculate
IQR=df.Salaries.quantile(0.75)-df.Salaries.quantile(0.25)
#iqr  varible explorer
IQR
#lower-upper limited
lower_limit=df.Salaries.quantile(0.25)-1.5*IQR
upper_limit=df.Salaries.quantile(0.75)+1.5*IQR
#now if you will check the lower limit of salary ,
#is -19445.
#trimming
import numpy as np
import seaborn as sns
outliers_df=np.where(df.Salaries>upper_limit,True,np.where(df.Salaries<lower_limit,True,False))
#you can check outlier _df column in varible explore
df_trimmed=df.loc[~outliers_df]
df.shape
df_trimmed.shape
#########################
#Replacement  Techniqes
#drowback id teimming  techniqe is we losing data
df=pd.read_csv("c:/Data-Science/5-Data_Preprocessing/ethnic diversity.csv")
df.describe()
#record no 23 has got outlier
#map all oulier value to upper limit
df_replaced=pd.DataFrame(np.where(df.Salaries>upper_limit,
     upper_limit,np.where(df.Salaries<lower_limit,lower_limit,df.Salaries)))
#if the value are greteer than uppe-limit
#map  upper and less than lower
#map lower within range
sns.boxplot(df_replaced[0])

######################################
20-5-25
#pip install feature_engine
#winsorizer
from feature_engine.outliers import winsorizer
# Create Winsorizer object
winsor = winsorizer(capping_method='iqr', tail='both', fold=1.5, variables=['Salaries'])
#copy winsorizer and paste is help tab of top right window,study the method
df_t=winsor.fit_tranform(df[['Salaries']])
sns.boxplot(df['Salaries'])
sns.boxplot(df_t['Salaries'])
###########################
#0 variance and near 0 variance
#if there is no variance in the feature

#features
import pandas as pd
df=pd.read_csv("c:/Data-Science/5-Data_Preprocessing/ethnic diversity.csv")
#df.var()
numeric_df=df.select_dtypes(include='number')
numeric_df.var()
#df.select_dtypes(include='number').var()
#here empid and zip is normal 
#1 select only numericcolumns
numeric_df=df.select_dtypes(include='number')
#2 find variance of each numeric column
variances=numeric_df.var()
#3identify columns with zero variance
zero_var_cols=variances[variances==0].index.tolist()
#drop those column from  original dataframe
df_cleaned =df.drop(columns=zero_var_cols)
#optinal:print the column dropped
print("Dropped columns with zero variance:",zero_var_cols)
############################
#install ==  pip install scikit-learn
#mean imputer
from  sklearn.impute import SimpleImputer
mean_imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
#check the dataframe
df['Salaries']=pd.DataFrame()(mean_imputer.fit_transform(df[['Salaries']]))
#check the dataframe
df['Salaries'].isna().sum()
#0
#21-5-25
#median Imputer
import numpy as np
median_imputer=SimpleImputer(missing_values=np.nan,strategy='median')
df['age']=pd.DataFrame(median_imputer.fit_tranform(df[['age']]))
df['age'].isna().sum()
#output =0
###################################
#mode imputer
import numpy as np
mode_imputer=SimpleImputer(missing_values=np.nan,strategy='most_frequent')
df['age']=pd.DataFrame(mode_imputer.fit_tranform(df[['age']]))
df['age'].isna().sum()
#0
df['MaritalDesc']=pd.DataFrame(mode_imputer.fit_tranform(df[['MaritalDesc']]))
df['MaritalDesc'].isna().sum()
################################################