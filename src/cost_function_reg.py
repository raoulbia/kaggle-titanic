#!/usr/bin/env python

# https://www.yuthon.com/2016/08/05/Coursera-Machine-Learning-Week-3/
import logging
logger = logging.getLogger(name=__name__)

import numpy as np
np.set_printoptions(suppress=True)

from utils import sigmoid

def costFunctionReg(theta, X, y, learning_rate, reg_param, iterations):

    # init useful vars
    costHistory = []
    m = y.shape[0] # number of training examples
    J = 0

    # initialize a gardient vector
    gradient = np.zeros((theta.shape)) # theta is a column vector

    for i in range(iterations):

        predictions = sigmoid(X * theta) # [m x 1] = [m x n] x [n x 1]
        delta = predictions - y # delta is an [m x 1] column vector

        theta_rest = theta[1:] # [(n-1) x 1]

        grad_intercept = learning_rate * 1/m * X[:, 0 ].T * delta

        grad_rest      = learning_rate * 1/m * X[:, 1:].T * delta  # [(n-1) x 1] = [(n-1) x m] x [m x 1] + [(n-1) x 1]
        reg_term       = reg_param * 1/m * theta_rest # [(n-1) x m]
        grad_rest      = grad_rest + reg_term

        gradient = np.concatenate((grad_intercept, grad_rest))

        theta = theta - gradient

        # compute cost (i.e. training error)
        J        = -1/m * ( y.T * np.log(predictions) + (1 - y.T) * np.log(1 - predictions) )  # y.T is [1 x m] ; predictions is [m x 1]
        reg_term = reg_param * 1/(2 * m) * sum(np.square(theta_rest))
        J        = J + reg_term

        costHistory.append(np.asscalar(J))

    return costHistory, theta