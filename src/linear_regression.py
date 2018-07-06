#!/usr/bin/env python

import logging
logger = logging.getLogger(name=__name__)

import numpy as np
# np.set_printoptions(suppress=True)

import utils


def learningCurveLogReg(X, y, X_val, y_val, learning_rate, iterations, reg_param):
    m = X.shape[0]
    # m = 30
    error_train = [] #np.zeros((m, 1))
    error_test = [] #np.zeros((m, 1))

    i = 0
    for subset_size in range(1, m, 10):
        X_train = X[:subset_size, :]
        y_train = y[: subset_size]

        # learn theta
        theta, _ = trainLogReg(X=X_train,
                               y=y_train,
                               learning_rate=learning_rate,
                               iterations = iterations,
                               reg_param=reg_param)

        cost_tr, _ = regularizedCostLogReg(X=X_train, theta=theta, y=y_train, learning_rate=1, reg_param=0)
        error_train.append(np.asscalar(cost_tr))

        cost_te, _ = regularizedCostLogReg(X=X_val, theta=theta, y=y_val, learning_rate=1, reg_param=0)
        error_test.append(np.asscalar(cost_te))
        i += 1

    return error_train, error_test


def trainLogReg(X, y, learning_rate, iterations, reg_param):
    cost_history = []
    theta = np.zeros((X.shape[1], 1)) # init theta col. vector
    for i in range(iterations):
        J, gradient = regularizedCostLogReg(X, theta, y, learning_rate, reg_param)
        theta = theta - gradient
        cost_history.append(J[0,0])
    return theta, cost_history


def regularizedCostLogReg(X, theta, y, learning_rate, reg_param):

    # init useful vars
    m = y.shape[0] # number of training examples

    predictions = utils.sigmoid(X * theta) # [m x 1] = [m x n] x [n x 1]
    delta = predictions - y # delta is an [m x 1] column vector

    theta_rest = theta[1:] # [(n-1) x 1]

    grad_intercept = learning_rate / m * X[:, 0 ].T * delta  # don't regularize intercept term
    grad_rest      = learning_rate / m * X[:, 1:].T * delta  # [(n-1) x 1] = [(n-1) x m] x [m x 1] + [(n-1) x 1]
    reg_term       = reg_param * 1/m * theta_rest # [(n-1) x m]
    grad_rest      = grad_rest + reg_term
    gradient = np.concatenate((grad_intercept, grad_rest))

    # compute cost for sanity checking (validation curve over lambda
    J = -1 / m * (y.T * np.log(predictions) + (1 - y.T) * np.log(1 - predictions))
    reg_term = (reg_param /(2*m)) * sum(np.square(theta_rest))
    J = J + reg_term

    return J, gradient


def predictLogReg(X, theta):

    predictions = utils.sigmoid(X * theta) # [m x 1] = [m x n] x [n x 1]

    pos = np.where(predictions >= 0.5) # index of values >= 0.5
    neg = np.where(predictions < 0.5)

    # set all indexes that were identified as positive to 1
    predictions[pos, 0] = 1 # the [pos, 0] the [ _ , 0] is just a way to index the the value in that position
    predictions[neg, 0] = 0
    return predictions # [m x 1]