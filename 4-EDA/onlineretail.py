# -*- coding: utf-8 -*-
"""
Created on Mon May  5 16:36:39 2025

@author: Ashlesha
"""
# Mon May  5 2025
import pandas as pd
import seaborn as sns
import matplotlib.pypolt as plt
import numpy as np
retail = pd.read_csv("c:/Data-Science/4-EDA/onlineRetail.csv", encoding='ISO-8859-1')
retail['UnitPrice']=retail['UnitPrice']
retail['Descripation']=retail['Descripation']
retail.describe()
retail.quantity.mean()
retail.quantity.median()
#inference
#unitprice is change to float for precision in calculat
#description is convered to string avoid type
#kurt_values=retail.kurt()
num_duplicates = retail.duplicated().sum()
print(f"Number of duplicate rows: {num_duplicates}")
#step 2 removing duplicates
retail.drop_duplicates(inplace=True)
#inference:
#duplicated rows can bias analysis, especially count
#step3 check for missing value
print(f"Missing values:\n",retail.isnull().sum())
retail.describe()
#step4 center tendency
print(f"Mean:\n{retail.mean(numeric_only=True)}")
plt.figure(figsize=(8,5))
sns.histplot(retail['Quantity'],bins=30,kde=True)
plt.title('Histogram of Quantity')
plt.show()
