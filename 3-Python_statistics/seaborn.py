# -*- coding: utf-8 -*-
"""
Created on Mon May  5 15:19:24 2025

@author: Ashlesha
"""
#5-5-25
import matplotlib.pyplot as plt
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
sns.displot(tips.total_bill,kde=True)
tips.shape
tips.size
tips.duplicated().sum()
tips.drop_duplicates(inplace=True)
#missing value
print("missing values:\n",tips.isnull().sum())
tips.info()
tips.describe()
#histrogram
sns.displot(tips.total_bill,kde=True)
sns.displot(tips.tip,kde=True)
#sns.displot(tips.size.kde=True)
sns.jointplot(x=tips.tip,y=tips.total_bill, kind='reg')
#scatter plot(center
'''each point represent one observation(a customer's
bill and the corresponding tip)
there a positive '''
sns.jointplot(x=tips.size,y=tips.total_bill,kind='reg')
sns.pairplot(tips,kind='reg')
sns.pairplot(tips,hue='day')

#Undestanding correlation coeffiction
sns.heatmap(tips.corr(numeric_only=True),annot=True)
sns.boxplot(tips.total_bill)
#there are outliers in total_bill
sns.boxplot(tips.tip)
#sun,fri,sat,thu days
sns.countplot(tips.day)
#male or female count
sns.countplot(tips.sex)
tips.sex.value_counts().plot(kind='pie')
tips.sex.value_counts().plot(kind='bar')
sns.boxplot(tips.total_bill)
sns.boxplot(tips.tip)
sns.countplot(tips.day)
sns.countplot(tips.sex)
tips.sex.value_counts().plot(kind='pie')
tips.sex.value_counts().plot(kind='bar')
sns.countplot(data=tips[tips.time=='Dinner'],x='day')
sns.countplot(data=tips[tips.time=='Lunch'],x='day')
#FaceGride
fg = sns.FacetGrid(tips, row='smoker', col='time')
fg.map(sns.histplot, 'total_bill')
