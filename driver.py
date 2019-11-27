import pca
import compress
import numpy as np

img = compress.load_data('Data/Train/')
compress.compress_images(img,100)



exit()


#X = np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
X = np.array([[1,1],[1,0],[2,2],[2,1],[2,4],[3,4],[3,3],[3,2],[4,4],[4,5],[5,5],[5,7],[5,4]])
#X = np.array([[90,60,90],[90,90,30],[60,60,60],[60,60,90],[30,30,30]])
#X = np.array([[2.5,2.4],[.5,.7],[2.2,2.9],[1.9,2.2],[3.1,3],[2.3,2.7],[2,1.6],[1,1.1],[1.5,1.6],[1.1,.9]])
#X = np.array([[0,8],[8,9],[12,11],[20,12]])
Z = pca.compute_Z(X,True,True)
print("Z:")
print(Z)
print()
COV = pca.compute_covariance_matrix(Z)
print("COV:")
print(COV)
print()
L, PCS = pca.find_pcs(COV)
print("L:")
print(L)
print("PCS:")
print(PCS)
print()
Z_star = pca.project_data(Z,PCS,L,1,0)
print("Z_star:")
print(Z_star)