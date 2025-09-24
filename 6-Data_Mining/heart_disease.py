# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 19:24:04 2025

@author: Ashlesha
"""

"""
A pharmaceuticals manufacturing company is conducting a study on 
a new medicine to treat heart diseases. 
The company has gathered data from its secondary 
sources and would like you to provide high level analytical insight
on the data.  
Its aim is to segregate patients depending on their age group and other
factors given in the data.  
Perform PCA and clustering algorithms on the dataset and check if the 
clusters formed before and after PCA are the same and
provide a brief report on your model.  

You can also explore more ways to improve your model.  

1. Business Problem  
1.1 What is the business objective?  
Cardiovascular disease is the most common cause of mortality in
developed countries. Across the globe, the incidence of death
from cardiovascular and circulatory diseases has risen by one
third between 1990 and 2010, such that by 2015 one in three 
deaths worldwide will be due to cardiovascular diseases.  
Epidemiologic studies have played an important role in elucidating
the factors that predispose to cardiovascular disease and highlighting opp

1.2. Are there any constraints?
It is difficult to manually determine the odds of
getting heart disease based on risk factors.
However, machine learning techniques are useful to
predict the output from existing data.
"""
# age        int64: age  
# sex        int64: sex  
# cp         int64: chest pain type (4 values)  
# trestbps   int64: resting blood pressure  
# chol       int64: serum cholestoral in mg/dl  
# fbs        int64: fasting blood sugar > 120 mg/dl  
# restecg    int64: resting electrocardiographic results (values 0,1,2)  
# thalach    int64: maximum heart rate achieved  
# exang      int64: exercise induced angina  
# oldpeak    float64: ST depression induced by exercise relative to rest  
# slope      int64: the slope of the peak exercise ST segment  
# ca         int64: number of major vessels (0-3) colored by fluoroscopy  
# thal       int64: thal: 0 = normal; 1 = fixed defect; 2 = reversible defect  
# Let us apply K-means algorithm


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from sklearn.cluster import KMeans

# Load heart disease dataset
heart = pd.read_csv("C:/Data-Science/6-Data_Mining/heart_disease.csv")
heart.describe()
heart.info()
heart.dtypes
# List of features to plot histograms 
features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang']

plt.figure(figsize=(15, 10))

for i, feature in enumerate(features, 1):
    plt.subplot(3, 3, i)
    plt.hist(heart[feature], bins=20, color='skyblue', edgecolor='black')
    plt.title(f'Distribution of {feature}')
    plt.xlabel('feature')
    plt.ylabel('Count')
plt.show()
#unique age in the dataset
heart.age.value_counts()
#data Preprocessing
heart.isna().sum()
# we know that there is scale difference among the columns, which we have to eliminate
# either by using normalization or standardization
def norm_func(i):
    x = (i - i.min()) / (i.max() - i.min())
    return x

# Normalize the dataset (excluding non-numeric columns if needed)
heart_norm = norm_func(heart.iloc[:, :])  # or use specific columns if some are non-numeric
#---------------
#K-means
#-------------
# Elbow method to find optimal number of clusters
TWSS = []  # Total Within Sum of Squares
k_range = list(range(2, 8))

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(heart_norm)
    TWSS.append(kmeans.inertia_)
TWSS
# Plotting TWSS to determine optimal clusters
plt.plot(k_range, TWSS, 'ro-')
plt.xlabel("No_of_clusters")
plt.ylabel("Total Within Sum of Squares (TWSS)")
plt.show()
# From the plot, it is clear that the TWSS is reducing from k=2 to 3 and then slower,
# Hence k=3 is selected
model = KMeans(n_clusters=3)
model.fit(heart_norm)
model.labels_
mb = pd.Series(model.labels_)
heart['clust'] = mb
heart.head()
# Reordering columns for clarity
heart = heart.iloc[:, [16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]
heart.clust.value_counts()
from sklearn.preprocessing import scaler
from sklearn.preprocessing import MinMaxScaler

#considering  only numberical data
heart.data=heart.iloc[:,1:]

#normalizing  the numerical data
heart_norm=scaler(heart.data)
heart_norm
#--------------------
#pca
#--------------------
from sklearn.decomposition import PCA
# PCA with 6 components
pca = PCA(n_components=6)
pca_values = pca.fit_transform(heart_norm)

# Explained variance ratio
var = pca.explained_variance_ratio_
var
# Cumulative variance
var1 = np.cumsum(np.round(var, decimals=4) * 100)
print("Cumulative variance:", var1)
#PCA Score
pca_values
pca_value=pd.DataFrame(pca_values)
# Create DataFrame for PCA components (fixed pca_value → pca_values)
pca_data = pd.DataFrame(pca_values, columns=["comp0", "comp1", "comp2", "comp3", "comp4", "comp5"])

# Concatenate university names with first 3 PCA components
final = pd.concat([heart['clust'], pca_data.iloc[:, 0:3]], axis=1)
final
#-------------------------------------------
#Aglomerative clustring 
#----------------------------------------------

#either by  using  normalization or standardization
#whenever there is mixed data apply normalization
def norm_func(i):
    x = (i - i.min()) / (i.max() - i.min())
    return x
#now apply this normalization  function to univ datframe
#for all the row  and column from 1 until end
#since 0th column has university name hence skipped
heart_norm = norm_func(heart.iloc[:, 1:])
#you can check  the df_norm dataframe  which is scaled
#between value frome 0 to 1
#you can app decribe function to new data frame
b = heart_norm.describe()
#before you apply clustering you need to plot dendogram 1st 
#now to create dendograme , we need to measure distance
#we have to import linkage
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
#linkage function  gives us hierarchical or agglomerative clustering
#ref the help for linkage
z=linkage(heart_norm,method="complete",metric="euclidean")
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
h_complete=AgglomerativeClustering(n_clusters=3,linkage="complete",metric="euclidean").fit(heart_norm)
#apply lables to the cluster
cluster_labels = pd.Series(h_complete.labels_)
h_complete.labels_ 
#assign this series to heart dataframe as column and name the column as
heart['clust'] = cluster_labels
#we want to relocation the column  7 to 0th  position
heart = heart.iloc[:, [16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]
#now check the heart  dataframe
heart.iloc[:, 2:].groupby(heart.clust).mean()
