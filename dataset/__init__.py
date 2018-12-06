"""
Util functions about dataset
"""

import requests
import progressbar as pbar

def download_from_gdrive(file_id, pathname):
    GDRIVE_URL = 'https://docs.google.com/uc?export=download'

    sess = requests.Session()
    res = sess.get(GDRIVE_URL, params={'id': file_id}, stream=True)

    token = None
    for key, value in res.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break
    
    if token:
        params = {'id': file_id, 'confirm': token}
        res = sess.get(GDRIVE_URL, params=params, stream=True)
    
    with open(pathname, 'wb') as fd:
        for chunk in pbar.progressbar(res.iter_content(32768)):
            if chunk:
                fd.write(chunk)

