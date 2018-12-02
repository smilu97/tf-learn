'''
Iris Dataset
'''

import os
import numpy as np
import pandas as pd
from urllib import request

url_training = 'http://download.tensorflow.org/data/iris_training.csv'
url_test = 'http://download.tensorflow.org/data/iris_test.csv'
path_training = 'dataset/iris/iris_training.csv'
path_test = 'dataset/iris/iris_test.csv'

def download():
    '''
    Download csv files if not exists
    '''
    if not os.path.exists(path_training):
        request.urlretrieve(url_training, path_training)
    if not os.path.exists(path_test):
        request.urlretrieve(url_test, path_test)
    
def select_xy(arr):
    '''
    Divide array into input, output
    '''
    return arr[:,:4], arr[:,4]

def get(test=False):
    '''
    Get data
    '''
    download()
    return select_xy(pd.read_csv(path_test if test else path_training).values)
