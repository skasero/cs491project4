import numpy as np

def compute_Z(X, centering=True, scaling=False):
    if(centering):
        mean = np.mean(X,axis=0)
        X = np.subtract(X,mean)
    if(scaling):
        std = np.std(X,axis=0,ddof=1)
        X = np.divide(X,std)
    return X

def compute_covariance_matrix(Z):
    return np.matmul(Z.T,Z)

def find_pcs(COV):
    return np.linalg.eig(COV)

def project_data(Z, PCS, L, k, var):
    index = -1 # Used b/c I increment in the while loop at the begin
    if(var != 0):
        total_sum = np.sum(L)
        pca_array = 0
        curr_var = 0
        while(curr_var < var):
            index +=1
            pca_array += L[index]
            curr_var = pca_array / total_sum
    else:
        index = k-1 # Subtract 1 b/c I +1 in the statement below. This is just used b/c if
                    # k was original set to 2, and the matrix only had 2 dimensions, this would decrease it
                    # to 1, but then 1+1 for the statement below. 

    PCS = PCS[:,:index+1] # Used to keep the first index+1 columns of PCS

    return np.matmul(Z,PCS)
