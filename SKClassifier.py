import numpy as np
import matplotlib as plt
from scipy.io import loadmat
from sklearn import decomposition as dec

def vectors_reduction(variance, data):

    dict_data = {}
    updated_data = []

    pca = dec.PCA(variance)
    for i in range(len(data['y'])):
        print(i)
        nsamples, x, y = data['X'][:, :, :, i].shape
        tmp = data['X'][:, :, :, i].reshape(nsamples, x*y)
        pca.fit(tmp)
        updated_data.append(pca.transform(tmp))

    dict_data['X'] = updated_data
    dict_data['y'] = data['y']
    print(data['X'].shape)
    print(np.shape(dict_data['X']))


if __name__ == "__main__":
    train_data = loadmat("../Data/train_32x32.mat")

    vectors_reduction(0.95, train_data)
    
