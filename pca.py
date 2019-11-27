import numpy as np

def compute_Z(X, centering=True, scaling=False):
    if(centering):
        mean = np.mean(X,axis=0)
        X = np.subtract(X,mean)
    if(scaling):
        std = np.std(X,axis=0)
        X = np.divide(X,std)
    return X

def compute_covariance_matrix(Z):
    return np.matmul(Z.transpose(),Z)

def find_pcs(COV):
    return np.linalg.eig(COV)

def project_data(Z, PCS, L, k, var):
    pass
