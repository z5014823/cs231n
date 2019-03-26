import random
import numpy as np
from load_CIFAR10 import load_CIFAR10
import matplotlib.pyplot as plt
from knn_classifier import KNearestNeighbour

# matplotlib figure parameters
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

cifar10_dir = 'cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

# Visualise some examples from the dataset

# Subsample the data for more efficient code execution
num_training = 1000
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 100
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]

# Reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print(X_train.shape, X_test.shape)

# Create a kNN classifier instance, simply remembers the data
classifier = KNearestNeighbour()
classifier.train(X_train, y_train)

# Test implementation of compute_distances_two_loops
dists = classifier.compute_distances_two_loops(X_test)
print(dists.shape)

# Test implementation of predict_labels, we use k = 5 nearest neighbours
y_test_pred = classifier.predict_labels(dists, k=5)
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print('Got %d / %d correct => accuracy: %f' % (
    num_correct, num_test, accuracy))


# We speed up distance matrix computation by using partial vectorisation
#  with one loop
dists_one = classifier.compute_distances_one_loop(X_test)
# We check that our vectorised implementation is correct by computing
#  the Frobenius norm, which is the square root of the squared sum
#  of differences of all elements, so reshape the matrices into vectors
#  and compute the Euclidean distance between them
difference = np.linalg.norm(dists - dists_one, ord='fro')
print('Difference was: %f' % (difference))
if difference < 0.001:
    print('Good! The distance matrices are the same')
else:
    print('The distance matrices are different')

# We test the fully vectorised version inside compute_distances_no_loops
dists_two = classifier.compute_distances_no_loops(X_test)
difference = np.linalg.norm(dists - dists_two, ord='fro')
print('Difference was: {0}'.format(difference))
if difference < 0.001:
    print('Good! The distance matrices are the same')
else:
    print('The distance matrices are different')

# We compare how fast the implementations are
def time_function(f, *args):
    import time
    tic = time.time()
    f(*args)
    toc = time.time()
    return toc - tic

two_loop_time = time_function(classifier.compute_distances_two_loops, X_test)
print('Two loop version took %f seconds' % two_loop_time)
one_loop_time = time_function(classifier.compute_distances_one_loop, X_test)
print('One loop version took %f seconds' % one_loop_time)
no_loop_time = time_function(classifier.compute_distances_no_loops, X_test)
print('No loop version took %f seconds' % no_loop_time)

# We determine the best value of the hyperparameter k with cross-validation
num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]
X_train_folds = []
y_train_folds = []
# Split the training data into folds, after splitting X_train_folds
#  and y_train_folds should be lists of length num_folds, where
#  y_train_folds[i] is the label vector for the points in X_train_folds[i]
num_one_fold = X_train.shape[0] // num_folds # Size of fold
split_list = [(i + 1) * num_one_fold for i in range(num_folds - 1)]
    # num_folds - 1 is the number of splits
X_train_folds = np.split(X_train, split_list)
y_train_folds = np.split(y_train, split_list)

k_to_accuracies = {}

# Perform k-fold cross validation to find the best value of k
# For each possible value of k, run the kNN algorithm num_folds times
#  where in each case we use all but one of the folds as training data
#  and the last fold as a validation set
# Store the accuracies for all fold and all values of k
#  in the k_to_accuracies dictionary
for k in k_choices:
    print(k)
    k_to_accuracies[k] = []
    for i in range(num_folds):
        X_train_cv = np.concatenate(X_train_folds[:i] + X_train_folds[i+1:])
        y_train_cv = np.concatenate(y_train_folds[:i] + y_train_folds[i+1:])
        X_test_cv = X_train_folds[i]
        y_test_cv = y_train_folds[i]
        classifier.train(X_train_cv, y_train_cv)
        y_pred = classifier.predict(X_test_cv, k)
        score = np.mean(y_pred == y_test_cv)
        k_to_accuracies[k].append(score)

# Print out the computed accuracies
for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print('k = %d, accuracy = %f' % (k, accuracy))
