import numpy as np
import time
import sklearn.svm as svm

def predict_LinearSVC (train_data, train_labels, test_data) :
    print("Starting to compute Linear SVC")
    computeStart = time.time()
    convert_to_minutes = False
    
    # Initialization
    linear_svc = svm.LinearSVC()
    
    # Training
    print("Linear SVC : starting training")
    linear_svc.fit(train_data, train_labels)
    
    # Predictions on test_data
    print("Linear SVC : starting predictions")
    predicted_labels = linear_svc.predict(test_data)
    
    computeDuration = time.time() - computeStart
    if computeDuration > 60 :
        computeDuration = computeDuration / 60
        convert_to_minutes = True
    unit = " min" if convert_to_minutes else " s"
    print("Linear SVC finished, computing duration = " + str(computeDuration) + unit)
    
    return predicted_labels
