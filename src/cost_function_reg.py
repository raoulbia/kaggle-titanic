#!/usr/bin/env python

# https://www.yuthon.com/2016/08/05/Coursera-Machine-Learning-Week-3/
import logging
logger = logging.getLogger(name=__name__)

import numpy as np
np.set_printoptions(suppress=True)

from utils import sigmoid

def costFunctionReg(theta, X, y, learning_rate, reg_term, iterations):

    # init useful vars
    costHistory = []
    m = y.shape[0] # number of training examples
    J = 0

    # initialize a gardient vector
    gradient = np.zeros((theta.shape)) # theta is a column vector

    for i in range(iterations):

        predictions = sigmoid(X * theta) # [m x 1] = [m x n] x [n x 1]
        delta = predictions - y # delta is an [m x 1] column vector

        # split theta
        theta_intercept = theta[0] # optional: not used any further
        theta_rest = theta[1:]


        # compute gradient wo/regularizing intercept term
        grad_intercept = X[:, 0].T * delta
        grad_intercept /= m
        grad_intercept *= learning_rate

        # compute gradient w/regularization of features except intercept term!)
        grad_rest = X[:, 1:].T * delta # [(n-1) x 1] = [(n-1) x m] x [m x 1] + [(n-1) x 1]
        grad_rest /= m
        grad_rest = grad_rest + (reg_term / m) * theta_rest
        grad_rest *= learning_rate

        # re-assemble
        gradient = np.concatenate((grad_intercept, grad_rest))

        theta = theta - gradient

        # compute cost
        J = y.T * np.log(predictions) + (1 - y.T) * np.log(1 - predictions)  # y.T is [1 x m] ; predictions is [m x 1]
        J /= -m
        J = J + (reg_term / (2 * m) * sum(np.square(theta_rest)))

        costHistory.append(np.asscalar(J))

    return costHistory, theta