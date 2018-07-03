#!/usr/bin/env python

import numpy as np
from numpy import exp

def sigmoid(p):
    return 1 / (1 + exp(-p))

def costFunction(theta, X, y, learning_rate, num_iters, regularize=False):
    costHistory = []
    m = y.shape[0] # number of training examples
    J = 0

    # initialize a gardient vector
    gradient = np.zeros((theta.shape))

    for i in range(num_iters):

        predictions = sigmoid(X * theta)
        delta = predictions - y

        gradient = X.T * delta
        gradient /= m   # Take the average cost derivative for each feature
        gradient *= learning_rate  # Multiply the gradient by the learning rate
        theta = theta - gradient

        # cost
        J = 1 / m * (-y.T * np.log(predictions) - (1 - y.T) * np.log(1 - predictions))
        costHistory.append(np.asscalar(J))


    return costHistory, theta