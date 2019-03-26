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
        self.params['W2'] = std * np.random.randn(output_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        scores = None
        out_1 = X.dot(W1) + b1
        gate_1 = np.maximum(0, out_1)
        out_2 = gate_1.dot(W2) + b2
        scores = out_2 // (N, C)

        if y is None:
            return scores
    
        loss = None
        # Softmax classifier loss
        exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
        p = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        loss = -np.sum(np.log(p[np.arange(N), y])) / N 
            + 0.5*reg*(np.sum(np.square(W2)) + np.sum(np.square(W1)))
