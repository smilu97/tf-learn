"""
TSP Dataset
"""

import os
from .. import download_from_gdrive

GOOGLE_DRIVE_IDS = {
    'tsp5_train.zip': '0B2fg8yPGn2TCSW1pNTJMXzFPYTg',
    'tsp10_train.zip': '0B2fg8yPGn2TCbHowM0hfOTJCNkU',
    'tsp5-20_train.zip': '0B2fg8yPGn2TCTWNxX21jTDBGeXc',
    'tsp50_train.zip': '0B2fg8yPGn2TCaVQxSl9ab29QajA',
    'tsp20_test.txt': '0B2fg8yPGn2TCdF9TUU5DZVNCNjQ',
    'tsp40_test.txt': '0B2fg8yPGn2TCcjFrYk85SGFVNlU',
    'tsp50_test.txt.zip': '0B2fg8yPGn2TCUVlCQmQtelpZTTQ',
}

def download():
    for filename in GOOGLE_DRIVE_IDS.keys():
        file_id = GOOGLE_DRIVE_IDS[filename]
        filepath = 'dataset/tsp/' + filename
        if not os.path.exists(filepath):
            print('Download ' + filename)
            download_from_gdrive(file_id, filepath)

def get(test=False):
    '''
    Get TSP dataset
    '''
    download()
