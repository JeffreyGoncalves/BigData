import numpy as np
import time

class DMIN() :
    
    def __init__(self) :
        self.barycenters = {}
    
    # Train the model by calculating barycenters for each label
    def train(self, train_data, train_labels) :
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
        
        for i in range(10):
            if (len(allVectors[i]) != 0):
                self.barycenters[i+1] = np.mean(allVectors[i], axis = 0)
            
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

def predict_DMIN (train_data, train_labels, test_data) :
    print("Starting to compute DMIN Classifier")
    computeStart = time.time()
    convert_to_minutes = False
    
    # Initialization
    dmin = DMIN() 
    
    # Training
    print("DMIN : starting training")
    dmin.train(train_data, train_labels)
    
    # Predictions on test_data
    print("DMIN : starting predictions")
    predicted_labels = dmin.find_class_array(test_data)
    
    computeDuration = time.time() - computeStart
    if computeDuration > 60 :
        computeDuration = computeDuration / 60
        convert_to_minutes = True
    unit = " min" if convert_to_minutes else " s"
    print("DMIN Classifier finished, computing duration = " + str(computeDuration) + unit)
    
    return predicted_labels
