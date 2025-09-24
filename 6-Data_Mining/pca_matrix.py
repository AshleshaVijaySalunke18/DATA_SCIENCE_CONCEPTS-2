# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 15:58:51 2025

@author: Ashlesha
"""

import  numpy as np
import matplotlib.pylab as plt

#step 1: Create a data matrix  A(simulating 1700 sample with 7 feature)

np.random.seed(0)
A=np.random.rand(1700,7)
# A shape: 1700 rows(samples)* 7 columns (features/properties)
print("Step 1:Data Matrix  A shape:",A.shape)

#step 2:  Normalize the data - matrix X
#substract mean and divide by standard derivation(unit variance scaling)

mean_A=np. mean(A,axis=0)
std_A=np.std(A,axis=0)
X=(A-mean_A)/std_A

print("Step 2: Normalized matrix X created",X.shape)

#step 3: Compute Covariance Matrix S

#Covariance  show how much 2 feature vary together
S=np.cov(X, rowvar=False)# rowvar=false - columns are feature
print("Step 3:Covariance matrix S shape:", S.shape)

#step 4: Eigen Decomposition of Covariance matrix 
# Eigen value - importance of compontes
#EigenVecters- direction (principal components)

eig_value, eig_vectors=np.linalg.eigh(S)

#sort by descending eigenvalue(most imp 1)

sorted_idx=np.argsort(eig_value)[::-1]
eig_value=eig_value[sorted_idx]
eig_vectors=eig_vectors[:,sorted_idx]

print("Step 4: Eigen decomposition done")
print("Top 2 eigenvalues(important):", eig_value[:2])

#Step 5:  project the data  onto  principal components(score)
#multiply normalized data X with top eigenvectors

T=X@ eig_vectors #scores(new representation)
T2=T[:, :2]  # take only 1st 2 components

print("Step 5: Project data shape(1st PCs):", T2.shape)

#plot the result in 2D

plt.figure(figsize=(8,5))
plt.scatter(T2[:,0],T2[:,1], alpha=0.5,c='blue')

'''
T2[:,0]
T2[:the 2D PCA-transformed data]
T2[:,0]: selects the x-coordinates(principal component 1)for species
T2[:,1]selects the y-coordinates
alpha=0.5
sets transparency of the points(o=invisible, 1=fully opaque)

'''
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA:Data Projcted to 1st 2 components")
plt.grid(True)
plt.show()

###################################################

#_______ for IRIS Dataset______

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Step 1: Load the Iris dataset
iris = load_iris()
A = iris.data         # Feature matrix: 150 samples × 4 features
labels = iris.target  # Target labels: species as 0 (Setosa), 1 (Versicolor), 2 (Virginica)

print("Step 1: Data Matrix A shape:", A.shape)

# Step 2: Normalize the data (zero mean and unit variance)
# This helps ensure all features contribute equally
mean_A = np.mean(A, axis=0)      # Mean of each column (feature)
std_A = np.std(A, axis=0)        # Standard deviation of each feature
X = (A - mean_A) / std_A         # Standardization

print("Step 2: Normalized matrix X Shape",X.shape)

# Step 3: Compute Covariance Matrix
# Covariance shows how features vary together
S = np.cov(X, rowvar=False)      # Each column is a feature

print("Step 3: Covariance matrix S shape:", S.shape)

# Step 4: Perform Eigen Decomposition on the Covariance Matrix
# Get eigenvalues (variance explained) and eigenvectors (principal components)
eig_value, eig_vectors = np.linalg.eigh(S)   # eigh for symmetric matrices

# Sort in descending order
idx = np.argsort(eig_value)[::-1]   
eig_value = eig_value[sorted_idx]            # Reorder eigenvalues
eig_vectors = eig_vectors[:, sorted_idx]     # Reorder eigenvectors accordingly

print("Step 4: Eigenvalue decomposition done")
print("Step 4 eigenvalues:", eig_value[:2])   # Show most important components

# Step 5: Project the normalized data onto the top 2 principal components
T = X @ eig_vectors           # Projected data (scores)
T2 = T[:, :2]                 # Take only the first 2 components for 2D visualization

print("Step 5: Project data shape ( 2 D):", T2.shape)

# Optional Step: Visualize the PCA result with color-coded species
colors = ['red', 'green', 'blue']            # Color for each species
species = ['Setosa', 'Versicolor', 'Virginica']  # Species names

plt.figure(figsize=(8, 6))
for i in range(3): 
    plt.scatter(
        T2[labels == i, 0],        # x-axis: PC1 for species i
        T2[labels == i, 1],        # y-axis: PC2 for species i
        c=colors[i],               # Color of points
        label=species[i],          # Label for legend
        alpha=0.6                  # Transparency
    )

'''

T2 [lables=i,0]
T2: the 2D pca-transformed data (T2. shape=150*2)
lables==i:selects the rows where the class(species)lable is i(0,)
T2[lables==i,0]:select the x-coordinates(PC 1)
T2[lables=i,1]
select the y-coorident (PC2)for  species i
c=colors[i]
sets the color for this species point
color=[red,green,blue]-(one for each species)
spcies=[setosa,versicolor,virginica]
alpha=0.6
sets transparency of the point(0=invisible,1=full opque)
this helps ovelapping point be more visible

for i in range(3):# there are 3 species
    plt.scatter(T2)lables==i,0,t2[lables==i,1]
'''
#  plot
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA on Iris Dataset")
plt.legend()
plt.grid(True)
plt.show()


#-------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#-------------------------------------------------------------------------

#_______ for University_clustring.xlsx Dataset______

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import matplotlib.pylab as plt

# Load data
Uni1 = pd.read_excel("C:/Data-Science/6-Data_Mining/University_Clustering.xlsx")

# Optional: display summary info
print(Uni1.describe())
print(Uni1.info())

# Drop 'State' column
Uni = Uni1.drop(["State"], axis=1)

# Select numerical columns only (all except 'Univ')
Uni_data = Uni.iloc[:, 1:]

# Normalize numerical data
Uni_normal = scale(Uni_data)

# PCA with 6 components
pca = PCA(n_components=6)
pca_values = pca.fit_transform(Uni_normal)

# Explained variance ratio
var = pca.explained_variance_ratio_
print("Variance ratio per component:", var)

# Cumulative variance
var1 = np.cumsum(np.round(var, decimals=4) * 100)
print("Cumulative variance:", var1)

# Plot cumulative variance explained
plt.plot(var1, color="red")     # fixed typo plt.polt → plt.plot
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance (%)")
plt.title("PCA - Cumulative Variance Explained")
plt.grid()
plt.show()

# Create DataFrame for PCA components (fixed pca_value → pca_values)
pca_data = pd.DataFrame(pca_values, columns=["comp0", "comp1", "comp2", "comp3", "comp4", "comp5"])

# Concatenate university names with first 3 PCA components
final = pd.concat([Uni["Univ"], pca_data.iloc[:, 0:3]], axis=1)

# Scatter plot of first two PCA components (fixed 'com0' → 'comp0')
ax = final.plot(x='comp0', y='comp1', kind='scatter', figsize=(12, 8))

# Add university names as labels on the scatter plot points
# Fixed ax.test → ax.text and used apply correctly
final.apply(lambda x: ax.text(x['comp0'], x['comp1'], x['Univ'], fontsize=8), axis=1)

plt.show()















