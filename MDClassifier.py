import numpy as np
import matplotlib as plt
from scipy.io import loadmat
import time
import test

print(" ** MDClassifier.py : starting program")
print(" ** MDClassifier.py : loading data files")

train_data = loadmat("../Data/train_32x32.mat")
test_data = loadmat("../Data/test_32x32.mat")
train_eq_data = loadmat("../Data/train_32x32_equalized.mat")
test_eq_data = loadmat("../Data/test_32x32_equalized.mat")


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
    return success

if __name__ == "__main__":
    
    computeStart = time.time()
    div = False
    print(" ** MDClassifier.py : starting to compute learning vectors")
    print(" ** MDClassifier.py : computing with default data")
    success = calculate_success_rate(train_eq_data, train_eq_data)
    print("\n ** MDClassifier.py : success = " + str(success))
    successRate = 100*success/len(train_data["y"])
    print(" ** MDClassifier.py : sucess rate = " + str(successRate) + "%")
    computeEnd = time.time()
    computeDuration = computeEnd - computeStart
    if computeDuration > 60 :
        computeDuration = computeDuration / 60
        div = True
    unit = " min" if div else " s"
    print(" ** MDClassifier.py : computing duration = " + str(computeDuration) + unit)