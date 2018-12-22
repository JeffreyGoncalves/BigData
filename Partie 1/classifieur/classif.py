import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sp
import sys

from PCA import pca_filter
from DMIN import predict_DMIN
from KNN import predict_KNN
from NearestNeighbors import predict_NearestNeighbors
from SVC import predict_SVC 
from LinearSVC import predict_LinearSVC
from confusionMatrix import print_confusion_matrix

def loadmat(filepath) :
    data_mat = sp.loadmat(filepath)
    x, y, nsamples, nb_data = np.shape(data_mat['X'])
    
    data = np.ndarray(shape=(nb_data, x*y*nsamples), dtype=np.uint8, order='C')
    labels = np.ndarray(shape=(nb_data), dtype=np.uint8, order='C')
    for i in range(nb_data) :
        data[i] = data_mat['X'][:, :, :, i].copy().flatten()
        labels[i] = data_mat['y'][i][0]
        
    return data, labels
    
def calculate_success_rate(test_labels, predicted_labels) :
    success = 0
    for i in range(len(predicted_labels)):
        if (predicted_labels[i] == test_labels[i]):
            success += 1
    successRate = 100*success/len(predicted_labels)
    return success, successRate
    
if __name__ == '__main__':
    datap = False
    
    test_filepath = ""
    
    train_filepath_set = False
    train_filepath = "Data/train_32x32.mat"
    
    train_cap_set = False
    train_cap = 10000
    
    classifiers = ["DMIN", "SVC", "LinearSVC", "KNN", "NearestNeighbors"]
    
    PCA_used = False
    PCA_int = False;
    PCA_nbcomp_int = 0
    PCA_nbcomp_float = float(0)
    
    mode_chosen = False
    mode = "SVC"
    
    display_conf = False
    
    ##########################################################
    #                   Detection of options                 #
    ##########################################################
    nb_args = len(sys.argv)
    i = 1
    while i < nb_args :
        if sys.argv[i][:2] == "--" :
            option = sys.argv[i][2:]
            if option == "data" :
                if datap :
                    print("Error : --data should only be defined once")
                    datap = False
                    break
                elif i+1 < nb_args :
                    i = i+1
                    test_filepath = sys.argv[i]
                    datap = True
                else :
                    print("Error : argument n", i+1, "from 0 does not exist")
                    datap = False
                    break
        
            elif option == "trainset" :
                if train_filepath_set :
                    print("Error : --trainset should only be defined once")
                    datap = False
                    break
                elif i+1 < nb_args :
                    i = i+1
                    train_filepath = sys.argv[i]
                    train_filepath_set = True
                else :
                    print("Error : argument n", i+1, "from 0 does not exist")
                    datap = False
                    break
                
            elif option == "classifier" :
                if mode_chosen :
                    print("Error : --classifier should only be defined once")
                    datap = False
                    break
                elif i+1 < nb_args :
                    i = i+1
                    classifier = sys.argv[i]
                    if classifier in classifiers :
                        mode = classifier
                        mode_chosen = True
                    else :
                        print("Error : classifier entered does not exist")
                        datap = False
                        break
                else :
                    print("Error : argument n", i+1, "from 0 does not exist")
                    datap = False
                    break
                    
            elif option == "PCA":
                if PCA_used : 
                    print("Error : --PCA should only be defined once")
                    datap = False
                    break
                elif i+1 < nb_args :
                    i = i+1
                    nb_comp_str = sys.argv[i]
                    nb_comp = float(0)
                    nb_comp_int = 0
                    try : 
                        nb_comp = float(nb_comp_str)
                        if nb_comp > 0 and nb_comp < 1 :
                            PCA_int = False
                            PCA_nbcomp_float = nb_comp
                        elif nb_comp >= 1 and nb_comp <= 3072 :
                            PCA_int = True
                            PCA_nbcomp_int = int(nb_comp)
                        else :
                            print("Error : PCA number of components entered is not valid ( < 0 or > 3072)")
                            datap = False
                            break
                        PCA_used = True
                    except ValueError :
                        print("Error : PCA number of components entered is not convertible to float")
                        datap = False
                        break
                else :
                    print("Error : argument n", i+1, "from 0 does not exist")
                    datap = False
                    break
                    
            elif option == "conf":
                if not display_conf :
                    display_conf = True
                else :
                    print("Error : --conf option should be used only once")
                    datap = False
                    break 
                    
            elif option == "traincap":
                if not train_cap_set and i+1 < nb_args :
                    i = i+1
                    train_cap_str = sys.argv[i]
                    try : 
                        train_cap_temp = int(train_cap_str)
                        if train_cap_temp <= 0 :
                            print("Warning : Training cap entered is negative or equal to 0, will be ignored")
                        else :
                            train_cap = train_cap_temp
                        train_cap_set = True
                    except ValueError :
                        print("Error : Training cap entered is not convertible to int")
                        datap = False
                        break
                elif train_cap_set :
                    print("Error : --traincap option should be used only once")
                    datap = False
                    break 
                else:
                    print("Error : argument n", i+1, "from 0 does not exist")
                    datap = False
                    break
                    
            else :
                print("Error : argument n", i, "from 0 is an option but the option does not exist")
                datap = False
                break 
        else :
            print("Error : argument n", i, "from 0 is incorrect (missing option ?)")
            datap = False
            break 
        i = i+1
    ##########################################################
    
    if not datap :
        print()
        print("Usage : python classif.py --data [file_location]")
        print("Possible options :")
        print("--trainset [path_of_trainset] : replaces the usual training dataset by the one given")
        print("--classifier [name of classifier] : chooses the used classifier from the following ones :")
        print("'DMIN', 'SVC', 'LinearSVC', 'KNN', 'NearestNeighbors' (default : SVC)")
        print("--PCA [n] : uses PCA reduction with n components or, if 0 < n < 1, the variance for the PCA reduction (default : no PCA)")
        print("--conf : Displays confusion matrix with matplotlib.pyplot (default : False)")
        print("--traincap [n] : Sets the maximum number of pictures taken from train dataset to n")
    
    ##########################################################
    #                  Start of classifier                   #
    ##########################################################
    else :
        print("Classif.py : starting... \n")
    
        # Loading training and test sets
        if PCA_used :
            print("Loading training data")
            train_data_temp, train_labels = loadmat(train_filepath)
            print("Loading test data")
            test_data_temp, test_labels = loadmat(test_filepath)
            
            if PCA_int :
                print("Applying PCA to training data")
                train_data = pca_filter(train_data_temp[:train_cap], PCA_nbcomp_int)
                print("Applying PCA to test data")
                test_data = pca_filter(test_data_temp, PCA_nbcomp_int)
            else :
                print("Applying PCA to training data")
                train_data = pca_filter(train_data_temp[:train_cap], PCA_nbcomp_float)
                print("Applying PCA to test data")
                test_data = pca_filter(test_data_temp, np.shape(train_data)[1])
        else :
            print("Loading training data")
            train_data, train_labels = loadmat("Data/train_32x32.mat")
            print("Loading test data")
            test_data, test_labels = loadmat(test_filepath)
            
        if train_cap > len(train_data) :
            train_cap = len(train_data)
            
        # Calling chosen classifier
        if mode == "DMIN" :
            predicted_labels = predict_DMIN (train_data[:train_cap], train_labels[:train_cap], test_data)
        elif mode == "SVC" :
            predicted_labels = predict_SVC (train_data[:train_cap], train_labels[:train_cap], test_data)
        elif mode == "LinearSVC" :
            predicted_labels = predict_LinearSVC (train_data[:train_cap], train_labels[:train_cap], test_data)
        elif mode == "KNN" :
            predicted_labels = predict_KNN (train_data[:train_cap], train_labels[:train_cap], test_data)
        elif mode == "NearestNeighbors" :
            predicted_labels = predict_NearestNeighbors (train_data[:train_cap], train_labels[:train_cap], test_data)
            
        # Calculating and printing success / error rate
        success, successRate = calculate_success_rate(test_labels, predicted_labels)
        print("Labels predicted with success on test : ", success, "/", len(test_labels))
        print("Final success rate = " + str(successRate) + "%")
        print("--> final error rate = " + str(100 - successRate) + "%")
        print()
        
        # Showing confusion matrix
        print_confusion_matrix(test_labels, predicted_labels, display_conf)
