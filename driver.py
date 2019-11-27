import pca
import numpy as np


X = np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
Z = pca.compute_Z(X)
print(Z)
print()
COV = pca.compute_covariance_matrix(Z)
COV = np.array([[5,1],[4,5]])
print(COV)
print()
L, PCS = pca.find_pcs(COV)
print(L)
print(PCS)
print()
Z_star = pca.project_data(Z,PCS,L,1,0)