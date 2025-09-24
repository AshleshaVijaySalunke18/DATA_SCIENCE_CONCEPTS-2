# -*- coding: utf-8 -*-
"""
Created on Mon May 26 16:35:27 2025

@author: Ashlesha
"""
import pandas as pd
import matplotlib.pylab as plt
airlines = pd.read_csv("c:/Data-Science/6-Data_Mining/airlines.csv")
a=airlines.describe()
airlines = airlines.drop(['state'], axis=1, errors='ignore')
def norm_func(i):
    x = (i - i.min()) / (i.max() - i.min())
    return x
airlines_data = airlines.iloc[:, 1:]
b = airlines.describe()
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
z = linkage(df_norm, method="complete", metric="euclidean")
plt.figure(figsize=(12,6));
plt.title("Airlines-Hierachical Clustring dendogram");
plt.xlabel("Index")
plt.ylabel("Distance") 
sch.dendrogram(z, leaf_rotation=0, leaf_font_size=10)
plt.show()
n_clusters = 3
h_complete = AgglomerativeClustering(n_clusters=n_clusters, linkage="complete", metric="euclidean")
cluster_labels = pd.Series(h_complete.fit_predict(airlines))
airlines['Cluster'] = cluster_labels
cols = ['Cluster'] + [col for col in airlines.columns if col != 'Cluster']
airlines = airlines[cols]
airlines.groupby('Cluster').mean()
