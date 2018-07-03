#!/usr/bin/env python

import numpy as np
from utils import sigmoid

def predict(X, theta):

    # initialize variables
    m = X.shape[0]  # number of training examples
    p = np.zeros((m, 1)).astype(int)

    predictions = sigmoid(X @ theta)

    pos = np.where(predictions >= 0.5) # index of values >= 0.5
    neg = np.where(predictions < 0.5)

    # set all indexes that were identified as positive to 1
    predictions[pos, 0] = 1 # the [pos, 0] index is needed to access the value in that position
    predictions[neg, 0] = 0
    return predictions