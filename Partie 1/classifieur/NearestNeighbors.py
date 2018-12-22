import numpy as np
import time
import sklearn.neighbors as neighbors

def predict_NearestNeighbors (train_data, train_labels, test_data, nb_neighbors=5) :
    print("Starting to compute Nearest Neighbors")
    computeStart = time.time()
    convert_to_minutes = False
    
    # Initialization
    neighbors_kdtree = neighbors.NearestNeighbors(n_neighbors=nb_neighbors, algorithm='kd_tree')
    
    # Training
    print("Nearest Neighbors : starting training")
    neighbors_kdtree.fit(train_data, train_labels)
    
    # Predictions on test_data
    print("Nearest Neighbors : starting predictions")
    distances, indices = neighbors_kdtree.kneighbors(test_data)
    predicted_labels = np.ndarray(shape=(len(test_data)))
    for i in range(len(test_data)) :
        if nb_neighbors == 1 :
            predicted_labels[i] = indices[i]
        else :
            mean_distances = {}
            for nb in range(nb_neighbors) :
                actual_label = train_labels[indices[i][nb]]
                actual_distance = distances[i][nb]
                if actual_label not in mean_distances :
                    mean_distances[actual_label] = []
                mean_distances[actual_label].append(actual_distance)
            
            md_keys = list(mean_distances.keys())
            min_label = md_keys[0]
            min_distance = np.mean(mean_distances[min_label])
            for label in md_keys :
                if label != min_label :
                    actual_distance = np.mean(mean_distances[label])
                    if actual_distance < min_distance :
                        min_label = label
                        min_distance = actual_distance
            predicted_labels[i] = min_label
        
    computeDuration = time.time() - computeStart
    if computeDuration > 60 :
        computeDuration = computeDuration / 60
        convert_to_minutes = True
    unit = " min" if convert_to_minutes else " s"
    print("Nearest Neighbors finished, computing duration = " + str(computeDuration) + unit)
    
    return predicted_labels
