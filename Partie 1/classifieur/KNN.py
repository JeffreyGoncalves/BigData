import numpy as np
import time
import sklearn.neighbors as neighbors

# Note : train_data should be limited at 10 000 pictures to make
# the computation possible in finite time
def predict_KNN (train_data, train_labels, test_data, nb_neighbors=5) :
    print("Starting to compute KNeighbors Classifier")
    computeStart = time.time()
    convert_to_minutes = False
    
    # Initialization
    knn = neighbors.KNeighborsClassifier(n_neighbors=nb_neighbors)
    
    # Training
    print("KNeighbors Classifier : starting training")
    knn.fit(train_data, train_labels)
    
    # Predictions on test_data
    print("KNeighbors Classifier : starting predictions")
    predicted_labels = knn.predict(test_data)
    
    computeDuration = time.time() - computeStart
    if computeDuration > 60 :
        computeDuration = computeDuration / 60
        convert_to_minutes = True
    unit = " min" if convert_to_minutes else " s"
    print("KNeighbors Classifier finished, computing duration = " + str(computeDuration) + unit)
    
    return predicted_labels
