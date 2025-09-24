# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 16:00:41 2025

@author: Ashlesha
"""

#SVD

import numpy as np

# Define a small 2x2 matrix
A = np.array([
    [4, 0],
    [3, -5]
])
print("Original Matrix A:\n", A)

# Step 2: Apply SVD
U, S, VT = np.linalg.svd(A)
'''
this given you:
    
    U: left singluar vectors(2*2)
S: Singluar values(only the diagonal,as a vector of length)
VT:Right singular vector transposed(2*2)'''

#step 3: Display the components
print("U(left singluar vectors):\n",U)
print("Singluar values(Sigma):\n",S)
print("V^T(Right singluar vectors):\n",VT)

#step 4: convert S(1D array) into a diagonal matrix
sigma = np.diag(S)
print("sigma as a diagonal matrix:\n", sigma)

#step 5: Reconstruct A  to check the decomposition
#multiply U*sigma* V^T to reconstruct A
A_reconstructed = U @ sigma @ VT
print("Reconstructed A (U * Sigma * V^T):\n", A_reconstructed)
'''
output(example value will look like):
    original matrix A:
[[4 0]
 [3 -5]]

U (Left Singular Vectors):
   [[ 0.62469505  0.78086881]
 [ 0.78086881 -0.62469505]]
S (Singular Values):
    [6.32455532 3.16227766]
VT (Right Singular Vectors Transposed):
plaintext
Copy code
[[ 0.9486833  -0.31622777]
 [ 0.31622777  0.9486833 ]]
'''
import numpy as np
#let's ssay we  have samples and 5 feature
A=np.array([
    [2,4,1,3,5],
    [1,3,0,4,4],
    [2,5,1,2,6],
    [3,6,2,3,7]])
print("Original Matrix A(4 samples*5 features):\n",A)
#apply SVD
U,S,VT=np.linalg.svd(A,full_matrices=False)

print("\nSingular values:\n",S)
#reduce dimension (keep top 2)
#keep top k=2 singular value
k=2
U_k=U[:,:k]#shape(4,2)
S_k=np.diag(S[:k])#2,2
VT_k=VT[:k,:]#2,5

#step 4: Reconstruct matrix with reduced rank
A_approx=U_k @S_k @ VT_k
print("\nReconstructed Matrix A(With k=20:\n",np.round(A_approx))
#compare size (optional)
original_size=A.size
reduced_size=U_k.size + S_k.size + VT_k.size

print(f"\nOriginal size: {original_size} values")
print(f"Reduced size (U_K + S_K+VT_K):{reduced_size} values")

'''
orignal data : 4*5 =20 value
SVD(rank-2):
  

'''














































