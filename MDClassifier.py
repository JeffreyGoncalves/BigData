import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time
import test
import preprocessing as pr
from sklearn.decomposition import PCA
import png

print(" ** MDClassifier.py : starting program")
print(" ** MDClassifier.py : loading data files")

train_data = loadmat("../Data/train_32x32.mat")
test_data = loadmat("../Data/test_32x32.mat")
pca_data = train_data.copy()

# ~ for i in range(len(train_data['y'])) :
for i in range(1) :
    picture = pca_data['X'][:,:,:,i]
    nsamples, x, y = pca_data['X'][:, :, :, i].shape
    X = picture.reshape(nsamples, x*y)
    for i in range(x) :
        print(X[i])
    # PCA 
    pca = PCA(n_components=32)
    pca.fit(X)
    X = pca.components_
    plt.imshow(X)
    plt.show()
    

print(" ** MDClassifier.py : applying pre-processing")

def calculate_barycenters(data):

    print(" ** MDClassifier.py : splitting vectors by classes")
    allVectors = []
    
    for i in range(10):
        allVectors.append([])
    
    for i in range(len(data['y'])):
                allVectors[data['y'][i][0]-1].append(data['X'][:, :, :, i])
    
    print(" ** MDClassifier.py : splitting done")

    barycenters = {}
    print(" ** MDClassifier.py : starting calculation of barycenters")
    for i in range(10):
        if (len(allVectors[i]) != 0):
            barycenters[i+1] = np.mean(allVectors[i], axis = 0)
    print(" ** MDClassfier.py : calculation done")
    return barycenters


def find_classes(picture, barycenters):
    idClass = 1
    mtxNorm = np.linalg.norm(picture - barycenters[idClass])
    for i in range(2,11):
        if(np.linalg.norm(picture - barycenters[i]) < mtxNorm):
            mtxNorm = np.linalg.norm(picture - barycenters[i])
            idClass = i
    return idClass

def calculate_success_rate(test, train):
    success = 0
    bc = calculate_barycenters(train)
    for i in range(len(test['y'])):
        idClass = find_classes(test['X'][:, :, :, i], bc)
        if (idClass == test['y'][i]):
            success += 1
    successRate = 100*success/len(test["y"])
    return success, successRate

if __name__ == "__main__":
    
    computeStart = time.time()
    div = False
    print(" ** MDClassifier.py : starting to compute learning vectors")
    print(" ** MDClassifier.py : computing with default data")
    [success, successRate] = calculate_success_rate(test_data, pca_data)
    print("\n ** MDClassifier.py : success = " + str(success))
    
    print(" ** MDClassifier.py : sucess rate = " + str(successRate) + "%")
    computeEnd = time.time()
    computeDuration = computeEnd - computeStart
    if computeDuration > 60 :
        computeDuration = computeDuration / 60
        div = True
    unit = " min" if div else " s"
    print(" ** MDClassifier.py : computing duration = " + str(computeDuration) + unit)
