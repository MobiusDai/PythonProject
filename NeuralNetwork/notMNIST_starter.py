import numpy as np
import network


# converts a 1d python list into a (1,n) row vector
def rv(vec):
    return np.array([vec])
    
# converts a 1d python list into a (n,1) column vector
def cv(vec):
    return rv(vec).T
        
# creates a (size,1) array of zeros, whose ith entry is equal to 1    
def onehot(i, size):
    vec = np.zeros(size)
    vec[i] = 1
    return cv(vec)

    
#################################################################

# reads the data from the notMNIST.npz file,
# divides the data into training and testing sets, and encodes the training vectors in onehot form
# returns a tuple (trainingData, testingData), each of which is a zipped array of features and labels
def prepData():
    # loads the four arrays specified.
    # train_features and test_features are arrays of (28x28) pixel values from 0 to 255.0
    # train_labels and test_labels are integers from 0 to 9 inclusive, representing the letters A-J
    with np.load("data/notMNIST.npz", allow_pickle=True) as f:
        train_features, train_labels = f['x_train'], f['y_train']
        test_features, test_labels = f['x_test'], f['y_test']
        
    # need to rescale, flatten, convert training labels to one-hot, and zip appropriate components together
    # CODE GOES HERE
    train_features = (train_features - train_features.min()) / (train_features.max() - train_features.min())
    test_features = (test_features - test_features.min()) / (test_features.max() - test_features.min())
    
    train_flatten = train_features.reshape((-1, 784, 1))
    test_flatten = test_features.reshape((-1, 784, 1))

    train_labels_onehot = []
    for i in range(len(train_labels)):
        train_labels_onehot.append(onehot(train_labels[i], 10))

    trainingData = zip(train_flatten, train_labels_onehot)
    testingData = zip(test_flatten, test_labels)
    return (trainingData, testingData)
    
###################################################################
import time

trainingData, testingData = prepData()

net = network.Network([784, 32, 10])

start = time.time()
net.SGD(trainingData, 10, 10, 2, test_data = testingData)
stop = time.time()
print(f' the time required to train the net :{stop - start:.2f} s')

network.saveToFile(net, './ModelWeights/part2.pkl')