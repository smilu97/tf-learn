"""
TSP Dataset
"""

import os
import zipfile
import numpy as np
import progressbar as pbar
from .. import download_from_gdrive

GOOGLE_DRIVE_IDS = {
    # 'tsp5_train.zip': '0B2fg8yPGn2TCSW1pNTJMXzFPYTg',
    'tsp10_train.zip': '0B2fg8yPGn2TCbHowM0hfOTJCNkU',
    # 'tsp5-20_train.zip': '0B2fg8yPGn2TCTWNxX21jTDBGeXc',
    # 'tsp50_train.zip': '0B2fg8yPGn2TCaVQxSl9ab29QajA',
    # 'tsp20_test.txt': '0B2fg8yPGn2TCdF9TUU5DZVNCNjQ',
    # 'tsp40_test.txt': '0B2fg8yPGn2TCcjFrYk85SGFVNlU',
    # 'tsp50_test.txt.zip': '0B2fg8yPGn2TCUVlCQmQtelpZTTQ',
}

def download():
    for filename in GOOGLE_DRIVE_IDS.keys():
        file_id = GOOGLE_DRIVE_IDS[filename]
        filepath = 'dataset/tsp/' + filename
        if not os.path.exists(filepath):
            print('Download ' + filename)
            download_from_gdrive(file_id, filepath)

def preprocess(fd):
    x = []
    y = []
    print('Reading data')
    for line in pbar.progressbar(fd):
        items = line.decode('utf8').split()
        try:
            o_index = items.index('output')
        except:
            print('No output:', items)
            continue
        x.append(np.reshape(
            np.array(items[:o_index], dtype=np.float32),
            (-1, 2)))
        y.append(np.array(items[o_index+1:], dtype=np.int32))
    pn = np.array(tuple(len(k) for k in x), dtype=np.int32)
    max_pn = np.max(pn)
    nx = np.zeros((len(x), max_pn, 2))
    ny = np.zeros((len(y), max_pn + 1))
    print('Merging data')
    for i, (ix, iy) in pbar.progressbar(enumerate(zip(x, y))):
        nx[i,:pn[i],:] = ix
        ny[i,:pn[i]+1] = iy
    return nx, ny

def get():
    '''
    Get TSP dataset
    '''
    download()
    zip_filepath = 'dataset/tsp/tsp10_train.zip'
    with zipfile.ZipFile(zip_filepath) as zfile:
        with zfile.open('tsp10_test.txt', 'r') as fd:
            test_x, test_y = preprocess(fd)
        with zfile.open('tsp10.txt', 'r') as fd:
            train_x, train_y = preprocess(fd)
    return train_x, train_y, test_x, test_y
