import numpy as np

class KNearestNeighbour(object):
    """ a kNN classifier with L2 distance """

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

    def compute_distances_two_loops(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                dists[i][j] = np.sqrt(np.sum(np.square(self.X_train[j] - X[i])))
        return dists

    def compute_distances_one_loop(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            dists[i] = np.sqrt(np.sum(np.square(self.X_train - X[i]), axis=1))
        return dists

    def compute_distances_no_loops(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        X_squared = np.sum(X**2, axis=1)
        Y_squared = np.sum(self.X_train**2, axis=1)
        XY = np.dot(X, self.X_train.T)
        dists = np.sqrt(X_squared[:,np.newaxis] + Y_squared - 2*XY)
        return dists

    def predict_labels(self, dists, k=1):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            closest_y = []
        
            index = np.argpartition(
                dists[i,:], -(self.X_train.shape[0] - k))[:k]
            label = list(self.y_train[index])

            from collections import Counter
            y_pred[i] = int(Counter(label).most_common()[0][0])

        return y_pred
