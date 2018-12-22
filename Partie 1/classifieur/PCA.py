import numpy as np
import time
from sklearn import decomposition as dec

def pca_filter(data, ncomp=0):
    print("Starting PCA decomposition")
    computeStart = time.time()
    convert_to_minutes = False
        
    # Creating PCA unit from Scikit-learn
    if ncomp == 0 :
        pca = dec.PCA()
    else :
        pca = dec.PCA(ncomp)
    
    # Creating new ndarray for pca transformation
    pca_data = pca.fit_transform(data)
    
    print("PCA : final data shape = ", np.shape(pca_data))
    
    computeDuration = time.time() - computeStart
    if computeDuration > 60 :
        computeDuration = computeDuration / 60
        convert_to_minutes = True
    unit = " min" if convert_to_minutes else " s"
    print("PCA finished, computing duration = " + str(computeDuration) + unit)
    
    return np.array(pca_data, order='C')
