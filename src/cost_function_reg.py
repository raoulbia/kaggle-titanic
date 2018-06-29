#!/usr/bin/env python

# https://www.yuthon.com/2016/08/05/Coursera-Machine-Learning-Week-3/

import math
import numpy as np
from numpy import exp
np.set_printoptions(suppress=True)
from scipy.special import expit

def sigmoid(p):
    return 1 / (1 + exp(-p))

def costFunctionReg(theta, X, y, learning_rate, reg_term, num_iters):
    costHistory = []
    m = y.shape[0] # number of training examples
    J = 0

    # initialize a gardient vector
    gradient = np.zeros((theta.shape))

    for i in range(num_iters):

        predictions = sigmoid(X * theta)
        delta = predictions - y

        theta_intercept = theta[0] # optional: not used any further
        theta_rest = theta[1:]

        # compute gradient (wo/regularizing intercept term!)
        grad_intercept = 1 / m * ( X[:, 0].T * delta )

        grad_intercept = X[:, 0].T * delta
        grad_intercept /= m  # Take the average cost derivative for each feature
        grad_intercept *= learning_rate  # Multiply the gradient by the learning rate

        grad_rest = X[:, 1:].T * delta
        grad_rest /= m
        grad_rest = grad_rest + (reg_term / m) * theta_rest
        grad_rest *= learning_rate


        gradient = np.concatenate((grad_intercept, grad_rest))
        theta = theta - gradient

        # compute cost
        J = 1 / m * (-y.T * np.log(predictions) - (1 - y.T) * np.log(1 - predictions)) + (reg_term / (2 * m) * sum(np.square(theta_rest)))
        costHistory.append(np.asscalar(J))

    return costHistory, theta