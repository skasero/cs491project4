import numpy as np
import matplotlib.pyplot as plt
import os

import pca

def compress_images(DATA,k):
    Z = pca.compute_Z(DATA,True,True)
    COV = pca.compute_covariance_matrix(Z)
    L, U = pca.find_pcs(COV)
    Z_star = pca.project_data(Z,U,L,k,0)
    X_compressed = np.dot(Z_star,U[:,:k].T)
    return X_compressed

def load_data(input_dir):
    output_array = []
    for images in os.listdir(input_dir):
        tmp_array = np.array(plt.imread(input_dir + images)).astype(np.float)
        output_array.append(tmp_array.flatten())
    return np.array(output_array).T

def save_data(X_compressed, output_dir = 'Data/myoutput/'):
    count = 0
    for images in X_compressed.T:
        images *= 255.0/images.max()
        plt.imsave(output_dir + 'output_' + str(count) + '.png',np.reshape(images,(-1,48)),cmap='gray')
        count +=1

    