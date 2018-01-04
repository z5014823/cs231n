import pickle
import numpy as np
import os

def load_cifar10_batch(filename):
    with open(filename, 'rb') as file:
        dict = pickle.load(file, encoding = 'bytes')
        X = dict[b'data']
        Y = dict[b'labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('float')
        Y = np.array(Y)
    return X, Y 

def load_cifar10(root):
    Xtr = []
    Ytr = []
    for i in range(1, 6):
        f = os.path.join(root, 'data_batch_%d' % (i))
        X, Y = load_cifar10_batch(f)
        Xtr.append(X)
        Ytr.append(Y)
    Xtr = np.concatenate(Xtr)
    Ytr = np.concatenate(Ytr)
    del X, Y
    Xte, Yte = load_cifar10_batch(os.path.join(root, 'test_batch'))
    return Xtr, Ytr, Xte, Yte
