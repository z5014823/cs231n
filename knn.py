import numpy as np
import sys

class knn:
    def __init__(self):
        pass

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loops(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    def predict_labels(self, dists, k):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            closest_y = self.y_train[np.argsort(dists[i,:])[:k]]

            u, indices = np.unique(closest_y, return_inverse=True)
            y_pred[i] = u[np.argmax(np.bincount(indices))]
        return y_pred 

    def compute_distances_two_loops(self, X):
        number_test = X.shape[0]
        number_train = self.X_train.shape[0]
        dists = np.zeros((number_test, number_train))
        for i in range(number_test):
            for j in range(number_train):
                dists[i,j] = np.sum((X[i,:] - self.X_train[j,:]) ** 2)
        return dists

    def compute_distances_one_loop(self, X):
        '''
        Same as compute_distances_two_loops but uses just a single loop
        '''
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            dists[i,:] = np.sum((self.X_train - X[i,:]) ** 2, axis=1)
        return dists
    
    # This needs revision
    '''
    def compute_distances_no_loops(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))

        T = np.sum(X**2, axis=1)
        F = np.sum(self.X_train**2, axis=1).T
        F = np.tile(F, (500,5000))
        FT = X.dot(self.X_train.T)
        print(T.shape, F.shape, FT.shape, X.shape, self.X_train.shape)
        dists = T + F - 2*FT
        return dists
    '''
