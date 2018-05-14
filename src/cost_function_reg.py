
import math
import numpy as np
from scipy.special import expit

def costFunctionReg(theta, X, y, _lambda, num_iters):
    m = y.shape[0] # number of training examples
    J = 0

    # initialize a gardient vector
    # the gradient of the cost is a vector of the same length as theta
    gradient = np.zeros((theta.shape))
    # print('grad shape', grad.shape)

    for i in range(num_iters):

        predictions = X * theta
        predictions = expit(predictions)

        theta_intercept = theta[0]
        theta_rest = theta[1:]

        # J = 1 / m * (-y.T * np.log(predictions) - (1-y.T) * np.log(1 - predictions))
        J = 1 / m * (-y.T * np.log(predictions) - (1-y.T) * np.log(1 - predictions)) + (_lambda / (2 * m) * sum(np.square(theta_rest)))

        error = predictions - y

        # % we need to return the gradient becuse we use the optimiser in this example
        grad_intercept = 1 / m * ( X[:, 0].T * error )
        grad_rest = 1 / m * ( X[:, 1:].T * error) + (_lambda / m) * theta_rest
        gradient = np.concatenate((grad_intercept, grad_rest))
        # print(grad_intercept)
        # print(grad_rest)
        # print(gradient)

        # gradient = X.T * error
        # gradient /= m   # Take the average cost derivative for each feature
        # gradient *= learning_rate  # Multiply the gradient by the learning rate
        theta = theta - gradient
        # print(theta)
        # print('cost', J)

    return J, gradient