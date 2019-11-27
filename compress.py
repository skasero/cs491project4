import numpy as np
import matplotlib.pyplot as plt
import os

def compress_images(DATA,k):
    pass

def load_data(input_dir):
    output_array = []
    for images in os.listdir(input_dir):
        tmp_array = np.array(plt.imread(input_dir + images)).astype(np.float)
        output_array.append(tmp_array.flatten())

    return np.array(output_array).T
    