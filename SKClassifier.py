import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import decomposition as dec
import preprocessing as pr
import sys

def pca_filter(data, ncomp=0, ndata=0):
    print(" ** SKClassifier.py : starting PCA decomposition (PCA_filter) ...")
    
    # Retrieving shape of data (if not correct : crash)
    nb_data, nb_pixels = np.shape(data)
    
    # Setting up processing limit parameter
    ndata_int = int(ndata)
    if ndata_int > nb_data or ndata_int <= 0 :
        ndata_int = nb_data
        
    # Limiting data input to limit parameter
    nb_data = ndata_int
    data = data[:nb_data]
        
    # Creating PCA unit from Scikit-learn
    if ncomp == 0 :
        pca = dec.PCA()
    else :
        pca = dec.PCA(ncomp)
    
    # Creating new ndarray for pca transformation
    pca_data = pca.fit_transform(data)
    
    print(" ** SKClassifier.py : final pca_dat shape = ", np.shape(pca_data))
    
    print(" ** SKClassifier.py : ending PCA decomposition (PCA_filter) ...")
    return pca_data


def vectors_reduction(data, ncomp=2, ndata=0):

    # Setting up pprocessing limit parameter
    ndata_int = int(ndata)
    if ndata_int > len(data['y']) or ndata_int <= 0 :
        ndata_int = len(data['y'])
       
    # Retreiving shape of pictures (should be same for all)
    x, y, nsamples = data['X'][:, :, :, 0].shape
    data_type = type(data['X'][0, 0, 0, 0])
    
    # Creating a new dictionary for pca processing
    dict_data = {}
    labels = []
    for label in data['y'][:ndata_int] :
        labels.append(label)
    dict_data['y'] = np.array(labels)
    data_type = type(data['X'][0, 0, 0, 0])
    updated_data = np.ndarray(shape=(ncomp, x*y, ndata_int), dtype=data_type)

    pca = dec.PCA(ncomp)
    
    print(" ** SKClassifier.py : starting PCA decomposition (vectors_reduction) ...")
    for i in range(ndata_int):
        picture = data['X'][:, :, :, i].copy()
        tmp = picture.reshape(x*y, nsamples)
        pca_tmp = pca.fit_transform(tmp)
        updated_data[:, :, i] = pca_tmp.reshape(ncomp, x*y)
        
    dict_data['X'] = updated_data
    
    print(" ** SKClassifier.py : ending PCA decomposition (vectors_reduction) ...")
    
    return dict_data
    
def image_reduction(data, ncomp=24, ndata=0):

    # Setting up pprocessing limit parameter
    ndata_int = int(ndata)
    if ndata_int > len(data['y']) or ndata_int <= 0 :
        ndata_int = len(data['y'])
       
    # Retreiving shape of pictures (should be same for all)
    x, y, nsamples = data['X'][:, :, :, 0].shape
    data_type = type(data['X'][0, 0, 0, 0])
    dict_data = {}
    dict_data['y'] = data['y'][:ndata_int]
    updated_data = np.ndarray(shape=(nsamples*x, ncomp, ndata_int), dtype=data_type)

    pca = dec.PCA(ncomp)
    
    print(" ** SKClassifier.py : starting PCA decomposition (image_vector_reduction) ...")
    for i in range(ndata_int):
        picture = data['X'][:, :, :, i].copy()
        tmp = picture.reshape(nsamples*x , y)
        pca_tmp = pca.fit_transform(tmp)
        updated_data[:, :, i] = pca_tmp
        
    dict_data['X'] = updated_data
    
    print(" ** SKClassifier.py : ending PCA decomposition (image_vector_reduction) ...")
    
    return dict_data


if __name__ == "__main__":
    train_data = loadmat("../Data/train_32x32.mat")
    pca_filter(train_data, 0.98, 100)
    
