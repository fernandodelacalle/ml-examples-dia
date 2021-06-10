import os
import requests
import hashlib
import gzip
import numpy as np

path='/workspace/data/mnist'

def fetch(url):
    fp = os.path.join(path, hashlib.md5(url.encode('utf-8')).hexdigest())
    if os.path.isfile(fp):
        with open(fp, "rb") as f:
            data = f.read()
    else:
        with open(fp, "wb") as f:
            data = requests.get(url).content
            f.write(data)
    data = np.frombuffer(gzip.decompress(data), dtype=np.uint8).copy()
    return data


def mnist_data():
    x_train = fetch("http://www.dia.fi.upm.es/~lbaumela/PracRF11/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
    x_train = x_train.astype('float32')
    y_train = fetch("http://www.dia.fi.upm.es/~lbaumela/PracRF11/train-labels-idx1-ubyte.gz")[8:]
    y_train = y_train.astype('int_')
    x_test = fetch("http://www.dia.fi.upm.es/~lbaumela/PracRF11/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28*28))
    x_test = x_test.astype('float32')
    y_test = fetch("http://www.dia.fi.upm.es/~lbaumela/PracRF11/t10k-labels-idx1-ubyte.gz")[8:]
    y_test = y_test.astype('int_')
    return (x_train, y_train), (x_test, y_test)