import numpy as np
import time
import sklearn.svm as svm

# Note : train_data should be limited at 10 000 pictures to make
# the computation possible in finite time
def predict_SVC (train_data, train_labels, test_data) :
    print("Starting to compute SVC with kernel poly")
    computeStart = time.time()
    convert_to_minutes = False
    
    # Initialization
    svc = svm.SVC(degree=3, gamma='auto', kernel='poly')
    
    # Training
    print("SVC : starting training")
    svc.fit(train_data, train_labels)
    
    # Predictions on test_data
    print("SVC : starting predictions")
    predicted_labels = svc.predict(test_data)
    
    computeDuration = time.time() - computeStart
    if computeDuration > 60 :
        computeDuration = computeDuration / 60
        convert_to_minutes = True
    unit = " min" if convert_to_minutes else " s"
    print("SVC finished, computing duration = " + str(computeDuration) + unit)
    
    return predicted_labels
