'''
Based on: https://stackoverflow.com/questions/40427435/extract-images-from-idx3-ubyte-file-or-gzip-via-python

Displays the image in the MNIST training dataset at the provided index, along with its correct label

Usage: python imageViewer.py index

CPSC 501, Fall 2021
Author: Janet Leahy
'''

import numpy as np
import idx2numpy
import matplotlib.pyplot as plt
import sys

imagefile = "data/t10k-images.idx3-ubyte"    # can change these to read training data
datafile = "data/t10k-labels.idx1-ubyte"
imagearray = idx2numpy.convert_from_file(imagefile)

file = open(datafile, 'rb')
file.read(4)
n = int.from_bytes(file.read(4), byteorder='big')

dataarray = bytearray(file.read())
file.close()

# index is the image you want to view
try:
    index = int(sys.argv[1])
except:
    index = 0

# TO DO: get label from other file
label = dataarray[index]


print("Displaying test image")
print(f"index: {index}, label: {label}")

plt.imshow(imagearray[index], cmap=plt.cm.binary)
plt.show()