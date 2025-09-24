# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 15:14:10 2025

@author: Ashlesha
"""

import numpy as np
from numpy import array
from scipy.linalg import svd
A=array([[1,0,0,0,2],[0,0,3,0,0],[0,0,0,0,0],[0,4,0,0,0]])
print(A)
#SVD
U,d,Vt=svd(A)
print(U)
print(d)
print(Vt)
print(np.diag(d))
#SVD Appling to dataset
import pandas as pd
data=pd.read_excel("c:/Data_Science/6-Data_Mining/University_clustring.xlsx")
data.head()
data=data.iloc[:,2:]#remove non numeric data
data
from sklearn.decompsition import TruncatedSVD
svd=TruncatedSVD(n_components=3)
svd.fit(data)
reslut=pd.DataFrame(svd.transform(data))
result.head()
result.columns="pc0","pc1","pc2"
result.head()
   






