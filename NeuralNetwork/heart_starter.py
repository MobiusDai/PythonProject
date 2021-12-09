import csv
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

# given a data point, mean, and standard deviation, returns the z-score
def standardize(x, mu, sigma):
    return ((x - mu)/sigma)
    

##############################################

# reads number of data points, feature vectors and their labels from the given file
# and returns them as a tuple
def readData(filename):

    # CODE GOES HERE
    HeartData = []
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        for row in reader : HeartData.append(row[1:])

    Header = HeartData[0]
    HeartData = HeartData[1:]

    for item in HeartData:
        for i in range(len(Header)):
            if Header[i] == 'famhist':
                item[i] = 1 if item[i] == 'Present' else 0
            
            elif Header[i] == 'chd':
                item[i] = int(item[i])
            else:
                item[i] = float(item[i])
    
    n = len(HeartData)
    
    HeartData = np.array(HeartData)

    features = HeartData[:, 0 : -1]
    labels = HeartData[:, -1]

    features[:, 8] = features[:, 8] / features[:, 8].max()

    for i in range(9):
        if i == 4 or i == 8 : continue
        features[:, i] = standardize(features[:, i], features[:, i].mean(), features.std())
        
    return n, features, labels


################################################

# reads the data from the heart.csv file,
# divides the data into training and testing sets, and encodes the training vectors in onehot form
# returns a tuple (trainingData, testingData), each of which is a zipped array of features and labels
def prepData():

    n, features, labels = readData('data/heart.csv')

    # CODE GOES HERE
    train_num = n - 100
    train_data = features[:train_num, :]
    train_label = labels[:train_num]
    test_data = features[train_num:, :]
    test_label = labels[train_num:]

    train_label_onehot = []
    for i in range(train_num):
        train_label_onehot.append(onehot(int(train_label[i]), 2))

    print(train_data.shape)
    print(test_data.shape)
    train_data = train_data.reshape((-1, 9, 1))
    test_data = test_data.reshape((-1, 9, 1))
    trainingData = zip(train_data, train_label_onehot)
    testingData = zip(test_data, test_label)

    return (trainingData, testingData)


###################################################
import time

trainingData, testingData = prepData()

# net = network.Network([9,10,2])
# net.SGD(trainingData, 10, 10, .1, test_data = testingData)

net = network.Network([9, 32, 2])

start = time.time()
net.SGD(trainingData, 12, 10, 1, test_data = testingData)
stop = time.time()

print(f' the time required to train the net :{stop - start : .2f} s')
network.saveToFile(net, './ModelWeights/part3.pkl')
