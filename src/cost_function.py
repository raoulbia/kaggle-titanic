#!/usr/bin/env python

import numpy as np
from numpy import exp

def costFunction(predictions, y, reg_term):

    # init useful vars
    m = y.shape[0]  # number of training examples

    J = y.T * np.log(predictions) + (1 - y.T) * np.log(1 - predictions)  # y.T is [1 x m] ; predictions is [m x 1]
    J /= -m
    J = J + (reg_term / (2 * m) * sum(np.square(theta_rest)))
    return J