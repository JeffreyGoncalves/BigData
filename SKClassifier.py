import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import decomposition as dec

def pca_filter(data, ncomp=2, ndata=0):

    # Setting up pprocessing limit parameter
    ndata_int = int(ndata)
    if ndata_int > len(data['y']) or ndata_int <= 0 :
        ndata_int = len(data['y'])
       
    # Retreiving shape of pictures (should be same for all)
    x, y, nsamples = data['X'][:, :, :, 0].shape
    
    # Creating a new dictionary for pca processing
    dict_data = {}
    labels = []
    for label in data['y'][:ndata_int] :
        labels.append(label)
    dict_data['y'] = np.array(labels)
    data_type = type(data['X'][0, 0, 0, 0])
    updated_data = np.ndarray(shape=(x, y, nsamples, ndata_int), dtype=data_type)

    # Creating PCA unit from Scikit-learn
    pca = dec.PCA(ncomp)
    
    print("Starting PCA decomposition (PCA_filter) ...")
    for i in range(ndata_int):
        # Reshaping input picture to shape (nsamples, x*y) with x number
        # of lines and y number of columns
        picture = data['X'][:, :, :, i].copy()
        tmp = picture.reshape(nsamples, x*y)
        
        # Applying pca transformation and inverse transformation on pic
        pca_tmp = pca.fit_transform(tmp)
        tmp = pca.inverse_transform(pca_tmp).astype(data_type)
        
        # Replacing old picture with new picture
        updated_data[:, :, :, i] = tmp.reshape(x, y, nsamples)
        
    dict_data['X'] = updated_data
    
    print("Ending PCA decomposition (PCA_filter) ...")
    
    return dict_data


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
    print("shape of updated_data : ", np.shape(updated_data))

    pca = dec.PCA(ncomp)
    
    print("Starting PCA decomposition (vectors_reduction) ...")
    for i in range(ndata_int):
        picture = data['X'][:, :, :, i].copy()
        tmp = picture.reshape(x*y, nsamples)
        pca_tmp = pca.fit_transform(tmp)
        updated_data[:, :, i] = pca_tmp.reshape(ncomp, x*y)
        
    dict_data['X'] = updated_data
    
    print("Ending PCA decomposition (vectors_reduction) ...")
    
    return dict_data
    
def image_reduction(data, ncomp=768, ndata=0):

    # Setting up pprocessing limit parameter
    ndata_int = int(ndata)
    if ndata_int > len(data['y']) or ndata_int <= 0 :
        ndata_int = len(data['y'])
       
    # Retreiving shape of pictures (should be same for all)
    x, y, nsamples = data['X'][:, :, :, 0].shape
    data_type = type(data['X'][0, 0, 0, 0])
    
    dict_data = {}
    dict_data['y'] = data['y'][:ndata_int]
    updated_data = np.ndarray(shape=(nsamples, ncomp, ndata_int), dtype=data_type)
    print("shape of updated_data : ", np.shape(updated_data))

    pca = dec.PCA()
    print(ncomp)
    
    print("Starting PCA decomposition (vectors_reduction) ...")
    for i in range(ndata_int):
        tmp = data['X'][:, :, :, i].reshape(nsamples, x*y)
        print("shape of tmp : ", np.shape(tmp))
        pca_tmp = pca.fit_transform(tmp)
        updated_data[:, :, i] = pca_tmp
        print(np.shape(pca_tmp))
        
    dict_data['X'] = updated_data
    
    print("Ending PCA decomposition (vectors_reduction) ...")
    
    return dict_data


if __name__ == "__main__":
    train_data = loadmat("../Data/train_32x32.mat")
    plt.imshow(train_data['X'][:,:,:,0])
    plt.show()
    pca = dec.PCA(2)
    x, y, nsamples = train_data['X'][:, :, :, 0].shape
    tmp = np.swapaxes(train_data['X'][:, :, :, 0], 0, 2).reshape(nsamples, x*y)
    
    # testpic
    test_pic = np.zeros([32,32,3], dtype=np.uint8)
    for i in range(32) :
        for j in range(32) :
            for p in range(3) :
                test_pic[i,j,p] = tmp[p, i*32+ j]
    
    plt.imshow(test_pic)
    plt.show()
    
    tmp_transformed = pca.fit_transform(tmp)
    new_picture = np.swapaxes(pca.inverse_transform(tmp_transformed), 0, 1).reshape(x, y, nsamples)
    new_picture = np.swapaxes(new_picture, 0, 1)
    plt.imshow(new_picture)
    plt.show()
    # ~ vectors_reduction(1, train_data)
    
