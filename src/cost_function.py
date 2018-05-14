
import math
import numpy as np
from scipy.special import expit

def costFunction(theta, X, y, learning_rate, num_iters):
    m = y.shape[0] # number of training examples
    J = 0

    # initialize a gardient vector
    # the gradient of the cost is a vector of the same length as theta
    gradient = np.zeros((theta.shape))
    # print('grad shape', grad.shape)

    for i in range(num_iters):

        predictions = X * theta
        predictions = expit(predictions)
        # print('predictions shape', predictions.shape)
        # print(predictions[:5, :])
        # print(np.log(predictions[:5, :]))
        # print('pred shape', predictions.shape)
        # print(y.T.shape)

        J = 1 / m * (-y.T * np.log(predictions) - (1-y.T) * np.log(1 - predictions))
        error = predictions - y
        # print(error[:5, :])
        gradient = X.T * error
        gradient /= m   # Take the average cost derivative for each feature
        gradient *= learning_rate  # Multiply the gradient by the learning rate
        theta = theta - gradient
        # print(theta)
        # print('cost', J)

    return J, gradient