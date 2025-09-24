# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 16:37:47 2025

@author: Ashlesha
"""

import pandas as pd
df=pd.read_csv("c:/Data-Science/9-Classification/salaries.csv")
df.head()
inputs = df.drop('salary_more_than_100k',axis='columns')
target=df['salary_more_than_100k']
from sklearn.preprocessing import LabelEncoder
le_company=LabelEncoder()
le_job=LabelEncoder()
le_degree=LabelEncoder()
inputs['company_n']=le_company.fit_transform(inputs['company'])
inputs['job_n']=le_job.fit_transform(inputs['job'])
inputs['degree_n']=le_degree.fit_transform(inputs['degree'])
inputs_n=inputs.drop(['company','job','degree'],axis='columns')
target
from sklearn import tree
model=tree.DecisionTreeClassifier()
model.fit(inputs_n, target)
#is salary of google, computer engineer, bachelors degree>100 k?
model.predict([[2,1,0]])
#is salary of google, computer engineer,masters degree>100?
model.predict([[2,1,1]])
