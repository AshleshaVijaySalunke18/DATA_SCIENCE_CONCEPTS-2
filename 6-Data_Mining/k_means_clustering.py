# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 15:33:28 2025

@author: Ashlesha
"""
import os
os.environ["OMP_NUM_THREADS"]="1"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
#load unviersity dataset
Univ1 = pd.read_excel("C:/Data-Science/6-Data_Mining/University_Clustering.xlsx")
#drop non-numeric column
Univ=Univ1.drop(['State'],axis=1)
#apply standerdization
scaler=StandardScaler()
df_std=pd.DataFrame(scaler.fit_transform(Univ.iloc[:,1:]),columns=Univ.columns[1:])
#finding optioalk using elbow method (with standerdize data)
TWSS=[]
k_range=list(range(2,8))

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_std)
    TWSS.append(kmeans.inertia_)

#plot elbow curve
plt.plot(k_range, TWSS, 'ro-')
plt.title("Elbow Curve to Determine Optimal K")
plt.xlabel("Number of Clusters")
plt.ylabel("Total Within Sum of Squares (TWSS)")
plt.grid(True)
plt.show()
#apply kmean with optimal cluster (e.g,k=3)
model=KMeans(n_clusters=3,random_state=42)
model.fit(df_std)
#add cluster lables to the original dataset
Univ['Cluster'] = model.labels_
#rearranging columns  to bring cluster first
Univ=Univ[['Cluster']+list(Univ.columns[:-1])]
#now check the Univ1 dataframe
Univ.iloc[:,2:].groupby(Univ.Cluster).mean()



'''
🟩 Cluster 0
Feature	Value	Interpretation
SAT	1360	High SAT scores — academically strong students
Top10	87.5%	Majority are top performers in high school
Accept	34.5%	Low acceptance rate — selective college
SFRatio	6.5	Very low student-faculty ratio — personalized attention
Expenses	$61,133	Very high — likely private elite institutions
GradRate	84%	High graduation rate

📝 Summary:

This cluster represents elite, expensive, and selective institutions.

🟨 Cluster 1
Feature  	Value	Interpretation
SAT	       1114	Lower SAT scores
Top10	47%	Fewer top-performing students
Accept	67.8%	High acceptance — less selective
SFRatio	17.0	High ratio — large class sizes
Expenses	$13,385	Low cost — likely public or community colleges
GradRate	74%	Moderate graduation rate

Cluster 2
Feature	   Value	    Interpretation
SAT	       1309	      High SAT scores — strong academics
Top10	   85.6%	    High % of top-performing students
Accept	  29.6%	     Very selective
SFRatio	  11.94	     Moderate student-faculty ratio
Expenses  $28,360	 Mid-range cost
GradRate  91.5%   	 Very high graduation rate

'''





