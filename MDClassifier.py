import numpy as np
import matplotlib as plt
from scipy.io import loadmat
import time


print(" ** MDClassifier.py : starting program")
print(" ** MDClassifier.py : loading data files")

train_data = loadmat("../Data/train_32x32.mat")
test_data = loadmat("../Data/test_32x32.mat")

print(" ** MDClassifier.py : applying pre-processing")
print(" ** (for now there is no pre-processing)")

def regroup_by_class(data):

    print(" ** MDClassifier.py : splitting vectors by classes")
    allVectors = []
    class1 = []
    class2 = []
    class3 = []
    class4 = []
    class5 = []
    class6 = []
    class7 = []
    class8 = []
    class9 = []
    class10 = []

    allVectors.append(class1)
    allVectors.append(class2)
    allVectors.append(class3)
    allVectors.append(class4)
    allVectors.append(class5)
    allVectors.append(class6)
    allVectors.append(class7)
    allVectors.append(class8)
    allVectors.append(class9)
    allVectors.append(class10)
    
    for i in range(len(data['y'])):
        for j in range(1,11):
            if(data['y'][i] == j):
                allVectors[j - 1].append(data['X'][:, :, :, i])
    
    print(" ** MDClassifier.py : splitting done")
    return allVectors

def calculate_barycenters(vectors):
    barycenters = {}
    print(" ** MDClassifier.py : starting calculation of barycenters")
    for i in range(10):
        if (len(vectors[i]) != 0):
            barycenters[i+1] = np.average(vectors[i], axis = 0)
    print(" ** MDClassfier.py : calculation done")
    return barycenters


def find_classes(picture, barycenters):
    idClass = 1
    mtxNorm = np.linalg.norm(picture - barycenters[idClass])
    for i in range(2,11):
        if(np.linalg.norm(picture - barycenters[idClass]) < mtxNorm):
            mtxNorm = np.linalg.norm(picture - barycenters[idClass])
            idClass = i
    return idClass

def calculate_success_rate(test, train):
    success = 0
    splitvecs = regroup_by_class(train)
    bc = calculate_barycenters(splitvecs)
    for i in range(len(test["y"])):
        idClass = find_classes(test["X"][:, :, :, i], bc)
        if (idClass == test["y"][i]):
            success += 1
    return 100*success/len(test["y"])

if __name__ == "__main__":
    
    computeStart = time.time()
    div = False
    print(" ** MDClassifier.py : starting to compute learning vectors")

    successRate = calculate_success_rate(test_data, train_data)
    print("\n ** MDClassifier.py : success rate = " + str(successRate) + "%")

    computeEnd = time.time()
    computeDuration = computeEnd - computeStart
    if computeDuration > 60 :
        computeDuration = computeDuration / 60
        div = True
    unit = " min" if div else " s"
    print(" ** MDClassifier.py : computing duration = " + str(computeDuration) + unit)