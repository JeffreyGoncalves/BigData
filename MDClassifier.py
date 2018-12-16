import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time
import test
import preprocessing as pr
from sklearn.decomposition import PCA
import sklearn.svm as svm
import sklearn.neighbors as neighbors
import SKClassifier as sk

class DMIN() :
    
    def __init__(self) :
        self.barycenters = {}
    
    # Train the model by calculating barycenters for each label
    def train(self, train_data, train_labels) :
        print(" ** MDClassifier.py : starting training of DMIN classifier")
        print(" ** MDClassifier.py : splitting vectors by classes")
        nb_train, nb_pixels = np.shape(train_data)
        nb_labels = np.shape(train_labels)[0]
        
        # Verification 
        if nb_train > nb_labels :
        #Exception here
            print("Error : training not possible (DMIN.train())")
            return False
            
        allVectorsSt = []
        
        for i in range(10):
            allVectorsSt.append([])
        
        for i in range(nb_train):
            allVectorsSt[train_labels[i]-1].append(train_data[i])
        
        allVectors = np.array(allVectorsSt)
        print(" ** MDClassifier.py : splitting done")
        
        print(" ** MDClassifier.py : starting calculation of barycenters")
        for i in range(10):
            if (len(allVectors[i]) != 0):
                self.barycenters[i+1] = np.mean(allVectors[i], axis = 0)
        print(" ** MDClassifier.py : calculation done")
            
    # Predicts for the given picture the corresponding label. The picture 
    # must use the pre-processing of the training data
    def find_class(self, picture) :
        idClass = 1
        mtxNorm = np.linalg.norm(picture - barycenters[idClass])
        for i in range(2,11):
            if(np.linalg.norm(picture - barycenters[i]) < mtxNorm):
                mtxNorm = np.linalg.norm(picture - barycenters[i])
                idClass = i
        return idClass

    def find_class_array(self, test_data) :
        predicted_labels = []
        nb_test, nb_pixels = np.shape(test_data)
        
        for sel in range(nb_test) :
            picture = test_data[sel]
            actual_id_class = 1
            mtxNorm = np.linalg.norm(picture - self.barycenters[actual_id_class])
            for i in range(2,11):
                if(np.linalg.norm(picture - self.barycenters[i]) < mtxNorm):
                    mtxNorm = np.linalg.norm(picture - self.barycenters[i])
                    actual_id_class = i
            predicted_labels.append(actual_id_class)
            
        return predicted_labels
    
def calculate_success_rate(test_labels, predicted_labels) :
    success = 0
    for i in range(len(predicted_labels)):
        if (predicted_labels[i] == test_labels[i]):
            success += 1
    successRate = 100*success/len(predicted_labels)
    return success, successRate

if __name__ == "__main__":
    
    print(" ** MDClassifier.py : starting program \n")
    
    # Loading .mat files (originals or pre-processed)
    print(" ** MDClassifier.py : loading data files")
    train_mat = loadmat("../Data/train_32x32.mat")
    test_mat  = loadmat("../Data/test_32x32.mat")
    
    x, y, nsamples, nb_train = np.shape(train_mat['X'])
    x, y, nsamples, nb_test = np.shape(test_mat['X'])
    
    # Setting training data into line vectors
    train_data = np.ndarray(shape=(nb_train, x*y*nsamples), dtype=np.uint8)
    train_labels = np.ndarray(shape=(nb_train), dtype=np.uint8)
    for i in range(nb_train) :
        train_data[i] = train_mat['X'][:, :, :, i].copy().flatten()
        train_labels[i] = train_mat['y'][i][0]
    
    # Setting testing data into line vectors
    test_data = np.ndarray(shape=(nb_test, x*y*nsamples), dtype=np.uint8)
    test_labels = np.ndarray(shape=(nb_test), dtype=np.uint8)
    for i in range(nb_test) :
        test_data[i] = test_mat['X'][:, :, :, i].copy().flatten()
        test_labels[i] = test_mat['y'][i][0]
    
    print()
    
    # Initialising clock for time duration (PCA + normal)
    computeStart = time.time()
    div = False
    
    pca_train = sk.pca_filter(train_data, 20)
    print(" ** MDClassifier.py : final pca_dat shape = ", np.shape(pca_train))
    pca_test = sk.pca_filter(test_data, 20)
    print(" ** MDKClassifier.py : final pca_dat shape = ", np.shape(pca_test))
    
    
    # ********** DMIN classifier ******************* #
    # ~ print(" ** MDClassifier.py : starting to compute learning vectors")
    # ~ print(" ** MDClassifier.py : computing with default data")
    # ~ dmin = DMIN() 
    # ~ dmin.train(pca_train, train_labels)
    # ~ predicted_labels = dmin.find_class_array(pca_test)
    # ~ print()
    
    # ********** sklearn.SVM classifier ************ #
    svc = svm.SVC()
    print(" ** MDClassifier.py : starting SVC classifier")
    print(" ** MDClassifier.py : training SVC")
    svc.fit(pca_train, train_labels)
    
    div_mid = False
    computeMid = time.time()
    computeDurationMid = computeMid - computeStart
    if computeDurationMid > 60 :
        computeDurationMid = computeDurationMid / 60
        div_mid = True
    unit = " min" if div_mid else " s"
    print(" ** MDClassifier.py : computing duration (SVC : training) = " + str(computeDurationMid) + unit)
    
    print(" ** MDClassifier.py : starting SVC predictions on test data")
    predicted_labels = svc.predict(pca_test)
    print(" ** MDClassifier.py : end of SVC predictions on test data")
    
    # ********** sklearn.neighbors classifier ****** #
    # ~ print(" ** MDClassifier.py : starting Nearest Neighbors classifier")
    # ~ nb_neighbors = 1 # Must be int >= 1
    # ~ neighbors_kdtree = neighbors.NearestNeighbors(n_neighbors=nb_neighbors, algorithm='kd_tree')
    # ~ print(" ** MDClassifier.py : training Nearest Neighbors model")
    # ~ neighbors_kdtree.fit(pca_train, train_labels)
    # ~ print(" ** MDClassifier.py : starting Nearest Neighbors predictions on test data")
    # ~ distances, indices = neighbors_kdtree.kneighbors(pca_test)
    # ~ predicted_labels = np.ndarray(shape=(len(pca_test)))
    # ~ for i in range(len(pca_test)) :
        # ~ mean_distances = {}
        # ~ if nb_neighbors == 2 :
            # ~ predicted_labels[i] = train_labels[indices[i]]
        # ~ else : 
            # ~ for nb in range(nb_neighbors) :
                # ~ actual_label = train_labels[indices[i][nb]]
                # ~ actual_distance = distances[i][nb]
                
                # ~ if actual_label not in mean_distances :
                    # ~ mean_distances[actual_label] = []
                # ~ mean_distances[actual_label].append(actual_distance)
            
            # ~ md_keys = list(mean_distances.keys())
            # ~ min_label = md_keys[0]
            # ~ min_distance = np.mean(mean_distances[min_label])
            # ~ for label in md_keys :
                # ~ if label != min_label :
                    # ~ actual_distance = np.mean(mean_distances[label])
                    # ~ if actual_distance < min_distance :
                        # ~ min_label = label
                        # ~ min_distance = actual_distance
            # ~ predicted_labels[i] = min_label
        
    # ~ print(" ** MDClassifier.py : ending Nearest Neighbors predictions on test data")
    
    ##################################################
    
    # Determining success rate of model
    print(" ** MDClassifier.py : calculating success rate on test data")
    [success, successRate] = calculate_success_rate(test_labels, predicted_labels)
    # ~ [success, successRate] = calculate_success_rate(pca_test_data, predicted_labels)
    print(" ** MDClassifier.py : success = " + str(success))
    print(" ** MDClassifier.py : sucess rate = " + str(successRate) + "%")
    print()
    
    computeEnd = time.time()
    computeDuration = computeEnd - computeStart
    if computeDuration > 60 :
        computeDuration = computeDuration / 60
        div = True
    unit = " min" if div else " s"
    print(" ** MDClassifier.py : computing duration = " + str(computeDuration) + unit)
