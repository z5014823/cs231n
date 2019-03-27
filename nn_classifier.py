import numpy as np
import matplotlib.pyplot as plt

class TwoLayerNet(object):
    """
    Two-layer fully-connected neural network.
    N, input dimension
    H, hidden layer dimension
    C, classification classes
    We train the network with a softmax loss function and L2 regularisation
     on the weight matrices.
    The network uses a ReLU nonlinearity after the first fully connected layer.
    Architecture: input -> fc -> ReLU -> fc -> softmax
    The outputs of the second fully connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # Compute the forward pass
        scores = None

        z1 = X.dot(W1) + b1
        a1 = np.maximum(0, z1)
        z2 = a1.dot(W2) + b2  #(N,C)
        scores = z2

        # If the targets are not given, we are done
        if y is None:
            return scores 
    
        # Compute the loss
        loss = None
        z2_exp = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
          # For each score element in each row, subtract the correct score
          # axis=1 is across rows, keepdims=True keeps column vec output
        a2 = z2_exp / np.sum(z2_exp, axis=1, keepdims=True)
        loss = (-1/N) * np.sum(np.log(a2[np.arange(N),y])) \
            + (reg/2) * (np.sum(np.square(W2)) + np.sum(np.square(W1)))

        # Backward pass, compute gradients
        grads = {}
        dloss_n = 1/N 
        a2_id = np.zeros_like(a2)
        a2_id[range(N),y] = 1
          # a2_id is an (N,C) indicator matrix mask to get correct labels
        dz2 = dloss_n * (a2 - a2_id)
        dW2 = a1.T.dot(dz2) + reg*W2  # 2*reg*W2?
        db2 = np.sum(dz2, axis=0)

        da1 = dz2.dot(W2.T)  #(N,H)
        da1_id = np.zeros_like(da1) + (da1 > 0)
          # da1_id is an (N,H) indicator matrix mask to get ReLU output
        dz1 = da1 * da1_id 
        dW1 = X.T.dot(dz1) + reg*W1  # 2*reg*W1?
        db1 = np.sum(dz1, axis=0)

        grads['W1'] = dW1
        grads['b1'] = db1
        grads['W2'] = dW2
        grads['b2'] = db2

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent (SGD)
        learning_rate, scalar giving learning rate for optimisation
        learning_rate_decay, scalar giving factor used to decay the learning rate
            after each epoch
        reg, scalar giving regularisation strength
        num_iters, number of steps to take when optimising
        batch_size, number of training examples to use per step
        verbose, boolean that prints progress during optimisation if true
        """

        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimise the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            # Create a random minibatch of training data and labels
            X_batch = None
            y_batch = None
            index = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[index]
            y_batch = y[index]

            #Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            for key in grads.keys():
                self.params[key] -= learning_rate * grads[key]

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)
                learning_rate *= learning_rate_decay
    
        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        y_pred = None
        a1 = np.maximum(0, X.dot(self.params['W1']) + self.params['b1'])
        z2 = a1.dot(self.params['W2']) + self.params['b2']
        y_pred = np.argmax(z2, axis=1)
        return y_pred
