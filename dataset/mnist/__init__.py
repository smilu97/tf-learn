'''
MNIST Dataset
'''

import os
import numpy as np
import pandas as pd
from urllib import request

url_train_images = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
url_train_labels = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
url_test_images = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
url_test_labels = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'

path_train_images = 'dataset/mnist/train_images.gz'
path_train_labels = 'dataset/mnist/train_labels.gz'
path_test_images = 'dataset/mnist/test_images.gz'
path_test_labels = 'dataset/mnist/test_labels.gz'

def download():
    '''
    Download csv files if not exists
    '''
    if not os.path.exists(path_train_images):
        request.urlretrieve(url_train_images, path_train_images)
    if not os.path.exists(path_train_labels):
        request.urlretrieve(url_train_labels, path_train_labels)
    if not os.path.exists(path_test_images):
        request.urlretrieve(url_test_images, path_test_images)
    if not os.path.exists(path_test_labels):
        request.urlretrieve(url_test_labels, path_test_labels)
    
def select_xy(arr):
    '''
    Divide array into input, output
    '''
    return arr[:,:4], arr[:,4]

def read_label(path):
    with open(path, 'rb') as fd:
        magic = np.frombuffer(fd.read(4), np.int32)[0]
        assert(magic == 2049)
        num = np.frombuffer(fd.read(4), np.int32)[0]
        labels = np.frombuffer(fd.read(), np.uint8)
        assert(len(labels) == num)
    return labels

def read_images(path):
    with open(path, 'rb') as fd:
        magic = np.frombuffer(fd.read(4), np.int32)[0]
        assert(magic == 2051)
        num = np.frombuffer(fd.read(4), np.int32)
        nrows = np.frombuffer(fd.read(4), np.int32)
        ncols = np.frombuffer(fd.read(4), np.int32)
        pixels = np.frombuffer(fd.read(), np.uint8)
        images = np.reshape(pixels, (num, nrows, ncols))
    return images

def get(test=False):
    '''
    Get data
    '''
    download()
    if test:
        x = read_images(path_test_images)
        y = read_label(path_test_labels)
    else:
        x = read_images(path_train_images)
        y = read_label(path_train_labels)
    return x, y
