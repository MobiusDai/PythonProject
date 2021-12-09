import numpy as np
import idx2numpy
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


##################################################
# NOTE: make sure these paths are correct for your directory structure

# training data
trainingImageFile = "./data/train-images.idx3-ubyte"
trainingLabelFile = "./data/train-labels.idx1-ubyte"

# testing data
testingImageFile = "./data/t10k-images.idx3-ubyte"
testingLabelFile = "./data/t10k-labels.idx1-ubyte"


# returns the number of entries in the file, as well as a list of integers
# representing the correct label for each entry
def getLabels(labelfile):
    file = open(labelfile, 'rb')
    file.read(4)
    n = int.from_bytes(file.read(4), byteorder='big') # number of entries
    
    labelarray = bytearray(file.read())
    labelarray = [b for b in labelarray]    # convert to ints
    file.close()
    
    return n, labelarray

# returns a list containing the pixels for each image, stored as a (784, 1) numpy array
def getImgData(imagefile):
    # returns an array whose entries are each (28x28) pixel arrays with values from 0 to 255.0
    images = idx2numpy.convert_from_file(imagefile) 

    # We want to flatten each image from a 28 x 28 to a 784 x 1 numpy array
    # CODE GOES HERE

    # print(f'the shape of original image is {images.shape}')
    image_flatten = images.reshape((-1, 784, 1))
    # convert to floats in [0,1] (only really necessary if you have other features, but we'll do it anyways)
    # CODE GOES HERE
    features = (image_flatten - image_flatten.mean())/image_flatten.std()
    features = (image_flatten - image_flatten.min())/(image_flatten.max() - image_flatten.min())
   
    return features


# reads the data from the four MNIST files,
# divides the data into training and testing sets, and encodes the training vectors in onehot form
# returns a tuple (trainingData, testingData), each of which is a zipped array of features and labels
def prepData():
    ntrain, train_labels = getLabels(trainingLabelFile)

    # CODE GOES HERE

    train_labels_onehot = []
    for item in train_labels:
        train_labels_onehot.append(onehot(item, 10))

    train_images = getImgData(trainingImageFile)
    test_images = getImgData(testingImageFile)
    ntest, test_labels = getLabels(testingLabelFile)
    
    trainingData = zip(train_images, train_labels_onehot)
    testingData = zip(test_images, test_labels)

    return (trainingData, testingData)
    

###################################################
import time

trainingData, testingData = prepData()

# layers = [784, 10, 10]   # original layers
# epochs = 10              # original epochs
# batch_size = 10          # original batch size
# learning_rate = 0.1      # original learning rate

# layers = [784, 16, 10]   # First change layers
# epochs = 10              # First change epochs
# batch_size = 10          # First change batch size
# learning_rate = 0.1      # First change learning rate

layers = [784, 30, 10]     # Second change layers
epochs = 20                # Second change epochs
batch_size = 10            # Second change batch size
learning_rate = 3          # Second change learning rate

net = network.Network(layers)
start = time.time()
net.SGD(trainingData, epochs, batch_size, learning_rate, test_data = testingData)
stop = time.time()
print(f' the time required to train the net :{stop - start : .2f} s')
network.saveToFile(net, './ModelWeights/part1.pkl')

### load model weights to predict the test images and find wrong predict
net = network.loadFromFile('./ModelWeights/part1.pkl')
count = 0
for index, (x, y) in enumerate(testingData):
    pred = np.argmax(net.feedforward(x))
    if pred == y : continue
    count += 1
    print(index, pred)
    if count == 3 : break