import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

# Function can be found at
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes, display=False):
    normalize=False
    title='Confusion matrix'
    cmap=plt.cm.Blues
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    if display :
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
                
        plt.show()
        
def print_confusion_matrix(test_labels, predicted_labels, display=False) :
    cnf_matrix = confusion_matrix(test_labels[:len(predicted_labels)], predicted_labels)
    plot_confusion_matrix(cnf_matrix, ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"], display)
