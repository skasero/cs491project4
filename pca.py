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
    # count = 0
    if(var != 0):
        pass
        # while cur_variance < var
        # calculate new variable with +1 eigenvalue 
        # count++ on how many eigenvectors to use
    #else
        # count = k

    #matmul the Z with how many K to use
