import random
import numpy as np
from load_cifar10 import load_cifar10
import matplotlib.pyplot as plt

cifar10_dir = 'cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_cifar10(cifar10_dir)

# Check size of training and test data
print ('Training data shape: ', X_train.shape)
print ('Training labels shape: ', y_train.shape)
print ('Test data shape: ', X_test.shape)
print ('Test labels shape: ', y_test.shape)

# Visualise some images from the dataset
classes = ['plane', 'car'] # Same order as in label_names
num_classes = len(classes)
samples_per_class = 4

for y, cls in enumerate(classes):
    idxs = np.flatnonzero(y_train == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(X_train[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
#plt.show()

# Subsample the data
num_training = 500
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 50
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]

# Reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print(X_train.shape, X_test.shape)

# Train a k-nearest-neighbour classifier
from knn import knn
classifier = knn()
classifier.train(X_train, y_train)
dists = classifier.compute_distances_two_loops(X_test)
print(dists.shape)

# Make predictions 
y_test_pred = classifier.predict_labels(dists, 6)
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))

# Speed up distance matrix computation and confirm result
dists_one = classifier.compute_distances_one_loop(X_test)
difference = np.linalg.norm(dists - dists_one, ord='fro')
print('Difference was: %f' % (difference))
if difference < 0.001:
    print('Distance matrices are the same')
else:
    print('Distance matrices are different')
    
# Compute distance matrix without loops and confirm result
'''
dists_two = classifier.compute_distances_no_loops(X_test)
difference = np.linalg.norm(dists - dists.two, ord='fro')
print('Difference was: %f' % (difference))
if difference < 0.001:
    print('Distance matrices are the same')
else:
    print('Distance matrices are different')
'''

# Compare how fast the implementations are


# Cross validation
