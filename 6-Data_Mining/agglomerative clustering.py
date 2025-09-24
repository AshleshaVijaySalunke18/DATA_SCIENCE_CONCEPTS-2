# -*- coding: utf-8 -*-
"""
Created on Mon May 26 15:38:42 2025

@author: Ashlesha
"""

import pandas as pd
import matplotlib.pylab as plt
# Load the dataset from the given Excel file path
Univ1 = pd.read_excel(r"C:\Data-Science\6-Data_Mining\University_Clustering.xlsx")
# Get summary statistics of the dataset
a = Univ1.describe()
# Drop the 'State' column as it's not useful for clustering
Univ = Univ1.drop(["State"], axis=1)
#we know that there is scale difference among the columns,
#which we have to remove
#either by  using  normalization or standardization
#whenever there is mixed data apply normalization
def norm_func(i):
    x = (i - i.min()) / (i.max() - i.min())
    return x
#now apply this normalization  function to univ datframe
#for all the row  and column from 1 until end
#since 0th column has university name hence skipped
df_norm = norm_func(Univ.iloc[:, 1:])
#you can check  the df_norm dataframe  which is scaled
#between value frome 0 to 1
#you can app decribe function to new data frame
b = df_norm.describe()
#before you apply clustering you need to plot dendogram 1st 
#now to create dendograme , we need to measure distance
#we have to import linkage
from scipy.cluster.hierarchy import linkage

import scipy.cluster.hierarchy as sch
#linkage function  gives us hierarchical or agglomerative clustering
#ref the help for linkage
z=linkage(df_norm,method="complete",metric="euclidean")
plt.figure(figsize=(15,8));
plt.title("Hierachical Clustring dendogram");
plt.xlable("Index")
plt.ylable("Distance")
#ref help of dendograme
#sch.dendogram(z)
sch.dendrogram(z,leaf_rotation=0,leaf_font_size=10)
plt.show()
#dendogram()
#applying  agglomertive clustring choosing 3 as cluster from dendrogram
#whateveer has been displyed in dendrogram is not clustring
#it is just showing number od possible clustring
from sklearn.cluster import AgglomerativeClustering
h_complete=AgglomerativeClustering(n_clusters=3,linkage="complete",metric="euclidean").fit(df_norm)
#apply lables to the cluster
cluster_labels = pd.Series(h_complete.labels_)
#assign this series to univ dataframe as column and name the column as
Univ['clust'] = cluster_labels
#we want to relocation the column  7 to 0th  position
Univ1 = Univ.iloc[:, [7, 1, 2, 3, 4, 5, 6]]
#now check the univ1  dataframe
Univ1.iloc[:, 2:].groupby(Univ1.clust).mean()
#from the output cluster 2 has got hight top10
#lowest accent ratin best faculty ratio and highest expenses
#highest graduates ratio
""" Cluster 0: Upper-Mid Tier Colleges
Strong academic profile (78.8% of students in Top 10%).
Moderate selectivity: 39% acceptance rate.
Balanced student-faculty ratio (12.8).
Mid-level expenses (~$21,447).
High graduation rate (87.6%).
This cluster likely includes competitive but affordable colleges 
with strong  academics and good outcomes

Cluster 1: Less Selective, Low-Cost Colleges
Weaker academic profile (38.8% Top 10% students).
High acceptance rate (70%).
High student-faculty ratio (19.3) – possibly larger classes.
Lowest expenses (~$9,953).
Moderate graduation rate (71.8%).
These might be public universities or regional colleges with
open access  policies , lower tuition , and fewer resources per student

Cluster 2: Elite, Expensive Colleges
Very strong academic profile (89% Top 10%).
Very selective (only ~27% acceptance).
Low student-faculty ratio (10.0) – small class sizes.
Very high expenses (~$49,897).
"""

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Clustering Performance
# =======================

'''
If we have external ground truth labels to compute
supervised performance metrics like Adjusted Rand Index (ARI) or Normalized Mutual Information (NMI).
However, since your dataset likely does not have true cluster labels (as this is an unsupervised task),
we’ll use internal evaluation metrics such as:

Silhouette Score – Measures how similar a data point is to its own cluster vs. other clusters.

Davies-Bouldin Index – Measures the average similarity ratio of each cluster with its most similar cluster (lower is better).

Calinski-Harabasz Index – Ratio of between-cluster dispersion to within-cluster dispersion (higher is better).
'''

# Calinski-Harabasz Index – Ratio of between-cluster
# dispersion to within-cluster dispersion (higher is better).

silhouette = silhouette_score(df_norm, h_complete.labels_)
db_index = davies_bouldin_score(df_norm, h_complete.labels_)
ch_index = calinski_harabasz_score(df_norm, h_complete.labels_)


print("\n--- Clustering Performance Metrics ---")
print(f"Silhouette Score                : {silhouette:.4f} (range: -1 to +1, higher is better)")

# Silhouette Score
'''
Range: -1 to +1

Interpretation:
A value near +1 indicates that the sample is far away from the neighboring clusters (i.e., good clustering).
A value near 0 means the sample is on or very close to the decision boundary between two clusters.
A value less than 0 means the sample might have been assigned to the wrong cluster.

score (0.2930):
Low-to-moderate clustering quality.
There is some structure, but the clusters overlap or are not well separated.
Possibly, the number of clusters (3) is not optimal.
'''
print(f"Davies-Bouldin Index            : {db_index:.4f} (Lower is better)")
#Davies-Bouldin Index            : 1.0286
'''

Interpretation:
Measures intra-cluster similarity vs. inter-cluster difference.
Lower values are better.
A value around 1 is acceptable, but:
< 1 is considered good
> 2 is usually poor (clusters are too similar or not well separated).
score (1.0286):
Average clustering performance.
Clusters are not very tight or well-separated, but they’re not completely bad either.
bad either
Could be improved by:
Trying different Linkage methods (average, ward, etc.)
Trying a different number of clusters.

'''
print(f"Calinski-Harabasz Index   : {ch_index:.4f} (higher is better)")
#Calinski-Harabasz Index   : 24.6202
'''
Range: 0 to ∞

Interpretation:
Measures the ratio of between-cluster dispersion to within-cluster dispersion.

Higher values are better – they indicate well-separated, compact clusters.

No universal scale, but values:

> 100 = strong separation  
10–50 = moderate separation
<10 =weak
your score (24.62)
moderate clustring structure
not great , but better than random or trivial grouping
'''