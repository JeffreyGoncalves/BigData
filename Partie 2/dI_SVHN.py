import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import time
from scipy.io import loadmat
torch.manual_seed(0)


class CNN(nn.Module):

    def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 20, 5)
            self.conv2 = nn.Conv2d(20, 50, 5)
            self.fc1 = nn.Linear(1250, 500)
            self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2(x), 2))
            x = x.view(x.shape[0], -1) # Flatten the tensor
            x = F.relu(self.fc1(x))
            x = F.log_softmax(self.fc2(x), dim=1)
            return x

class LeNet(nn.Module):

    def __init__(self):
            super(LeNet, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(400, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2(x), 2))
            x = x.view(x.shape[0], -1) # Flatten the tensor
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.log_softmax(self.fc3(x), dim=1)
            
            return x

class MLP(nn.Module):

    def __init__(self):
            super(MLP, self).__init__()
            self.fc1 = nn.Linear(3072, 500)
            self.fc2 = nn.Linear(500, 200)
            self.fc3 = nn.Linear(200, 100)
            self.fc4 = nn.Linear(100, 10)

    def forward(self, x):
            x= x.contiguous()
            x = x.view(x.shape[0], -1) # Flatten the tensor
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x)))
            x = F.log_softmax(self.fc4(x), dim=1)
            
            return x

def success_rate(class_predicted, real_classes) :
    size_cl = len(class_predicted)
    size_rc = len(real_classes)
    if size_rc == size_cl :
        success = 0
        for i in range(size_cl) :
            if class_predicted[i] == real_classes[i] :
                success = success + 1
        return success/size_cl*100
    else :
        return 0

if __name__ == '__main__':

    # Load the dataset
    train_data = loadmat('train_32x32.mat')
    test_data = loadmat('test_32x32.mat')

    train_label = train_data['y']
    train_label = np.where(train_label==10, 0, train_label)
    train_label = torch.from_numpy(train_label.astype('int')).squeeze(1)
    train_data = torch.from_numpy(train_data['X'].astype('float32')).permute(3, 2, 0, 1)

    test_label = test_data['y']
    test_label = np.where(test_label==10, 0, test_label)
    test_label = torch.from_numpy(test_label.astype('int')).squeeze(1)
    test_data = torch.from_numpy(test_data['X'].astype('float32')).permute(3, 2, 0, 1)

    # Hyperparameters
    epoch_nbr = 10
    batch_size = 5
    learning_rate = 1e-4

    net = CNN()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    # ~ scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.75)
    x_axis = np.arange(0, epoch_nbr, 1)
    y_axis = []
    
    ##########
    computeStart = time.time()
    div = False
    ##########
    
    for e in range(epoch_nbr):
        success = 0.0
        # ~ scheduler.step()
        print("Epoch", e)
        for i in range(0, train_data.shape[0], batch_size):
            optimizer.zero_grad() # Reset all gradients to 0
            predictions_train = net(train_data[i:i+batch_size])
            _, class_predicted = torch.max(predictions_train, 1)
            loss = F.nll_loss(predictions_train, train_label[i:i+batch_size])
            loss.backward()
            optimizer.step() # Perform the weights update
            success = success + success_rate(class_predicted, train_label[i:i+batch_size])
            # print("Success rate : ", success_rate(class_predicted, train_label[i:i+batch_size]), " %")
        success = success / (train_data.shape[0]/batch_size)
        print("Final success rate : ", success, " %")
        print()
        y_axis.append(success)
        
    ##########
    computeEnd = time.time()
    computeDuration = computeEnd - computeStart
    if computeDuration > 60 :
        computeDuration = computeDuration / 60
        div = True
    unit = " min" if div else " s"
    print(" training duration = " + str(computeDuration) + unit)
    ##########
    
    plt.plot(x_axis, y_axis)
    plt.gca().set_ylim([0,100])
    plt.gca().set_xlim([0,epoch_nbr-1])
    plt.show()

    ##########
    computeStart = time.time()
    div = False
    ##########
    
    # Predictions sur les données de test
    predictions_test = net(test_data)
    _, test_class_predicted = torch.max(predictions_test, 1)
    
    computeEnd = time.time()
    computeDuration = computeEnd - computeStart
    if computeDuration > 60 :
        computeDuration = computeDuration / 60
        div = True
    unit = " min" if div else " s"
    print(" ** MDClassifier.py : computing duration = " + str(computeDuration) + unit)
    
    
    print("Success rate on test : ", success_rate(test_class_predicted, test_label), " %")
    



