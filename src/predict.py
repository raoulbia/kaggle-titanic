
import numpy as np
from scipy.special import expit

def predict(X, theta):

    # initialize variables
    m = X.shape[0]  # number of training examples
    p = np.zeros((m, 1)).astype(int)
    print(p.shape)

    predictions = X * theta
    # print('sig shape', sig.shape)
    predictions = expit(predictions)
    # print(predictions[:5,:])
    pos = np.where(predictions >= 0.5)
    neg = np.where(predictions < 0.5)
    p[pos, 0] = 1
    p[neg, 0] = 0
    # print(p.shape)
    # print(p[:5,:])

    return p