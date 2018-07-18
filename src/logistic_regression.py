# -*- coding: utf-8 -*-

import logging

logger = logging.getLogger(name=__name__)

import numpy as np
from numpy import logspace
# np.set_printoptions(precision=3)
# np.set_printoptions(suppress=True)

from sklearn.model_selection import KFold
from sklearn import model_selection
from matplotlib import pyplot as plt
from operator import add

import utils
import utils_viz


def validationCurve(X, y, hp_name, hp_values, iterations, learned_alpha=None):
    # nbr kfolds

    folds = 5

    # init np.array to capture error for each kfold
    error_tr = np.zeros((1, 1))
    error_te = np.zeros((1, 1))

    # get cross validation folds
    kf = KFold(n_splits=folds, shuffle=True)

    # iterate over folds
    for train, val in kf.split(X=X, y=y):
        # get train / validation splits
        X_train, X_val, y_train, y_val = X[train], X[val], y[train], y[val]

        # store intermedaite results
        cost_tr = []
        cost_te = []

        # iterate over hyperparameter values
        for hp_value in hp_values:

            # Train the linear model with X_train, y_train and lambda
            if hp_name == 'alpha':
                theta, _ = gradientDescent(X=X_train,
                                           y=y_train,
                                           alpha=hp_value,
                                           _lambda=0.,
                                           # reg term is only useful for generalization thus not needed for finding best learning rate
                                           iterations=iterations)

                # Compute training error for each lambda value and add to list
                Jtr = computeCost(X=X_train, y=y_train, theta=theta, _lambda=0.)
                cost_tr.append(np.asscalar(Jtr))

                # Compute test error for each lambda value and add to list
                Jte = computeCost(X=X_val, y=y_val, theta=theta, _lambda=0.)
                cost_te.append(np.asscalar(Jte))

            if hp_name == 'lambda':
                theta, _ = gradientDescent(X=X_train,
                                           y=y_train,
                                           alpha=learned_alpha,
                                           _lambda=hp_value,
                                           iterations=iterations)

                # Compute training error for each lambda value and add to list
                Jtr = computeCost(X=X_train, y=y_train, theta=theta, _lambda=hp_value)
                cost_tr.append(np.asscalar(Jtr))

                # Compute test error for each lambda value and add to list
                Jte = computeCost(X=X_val, y=y_val, theta=theta, _lambda=hp_value)
                cost_te.append(np.asscalar(Jte))

        # item by item list addition of cost vlaues from current fold
        error_tr = error_tr + cost_tr
        error_te = error_te + cost_te

    # average over nbr of folds
    error_tr = error_tr / folds
    error_te = error_te / folds
    logger.info('training error for different values of {}:\n{}'.format(hp_name, error_tr))
    logger.info('test error for different values of {}:\n{}'.format(hp_name, error_te))

    # plot
    utils_viz.plot_validation_curve(hp_values=hp_values,
                                    errors_tr=error_tr.flatten(),
                                    errors_te=error_te.flatten(),
                                    hyperparam=hp_name)


def learningCurve(X, y, alpha, _lambda, iterations):
    """
    visualize how the error changes as the input data increases
    """

    # init var to be used to inform the nbr of loops for training-data batches (of in creasing size)
    m = y.shape[0]

    # init nbr of kfolds
    folds = 5

    # init np.array to capture error for each kfold
    error_tr = np.zeros((1, 1))
    error_te = np.zeros((1, 1))

    # get cross validation folds
    kf = KFold(n_splits=folds, shuffle=True)

    # iterate over folds
    for train, val in kf.split(X=X, y=y):

        # get train / validation splits
        X_train, X_val, y_train, y_val = X[train], X[val], y[train], y[val]

        # init
        cost_tr = []
        cost_te = []

        for subset_size in range(1, m, 100):
            """
            for each subset X_train [:i, :]
            compute error for the subset of training data
            compute error for ALL OF the validation data
            """

            X_train_subset = X_train[:subset_size, :]
            y_train_subset = y_train[: subset_size]

            # learn theta_optimized
            theta, _ = gradientDescent(X=X_train_subset,
                                       y=y_train_subset,
                                       alpha=alpha,
                                       _lambda=_lambda,
                                       iterations=iterations)
            # Compute training error for given training data subset
            Jtr = computeCost(X=X_train_subset, y=y_train_subset, theta=theta, _lambda=0)
            cost_tr.append(np.asscalar(Jtr))

            # Compute test error for validation data
            Jte = computeCost(X=X_val, y=y_val, theta=theta, _lambda=0)
            cost_te.append(np.asscalar(Jte))

        # item by item list addition of cost vlaues from current fold
        error_tr = error_tr + cost_tr
        error_te = error_te + cost_te

    # average over nbr of folds
    error_tr = error_tr / folds
    error_te = error_te / folds
    logger.info('training error for increasing traing set size:\n{}'.format(error_tr))
    logger.info('test error for increasing traing set size::\n{}'.format(error_te))

    # plot
    utils_viz.plot_learning_curve(error_tr.flatten(), error_te.flatten())


def gradientDescent(X, y, alpha, _lambda, iterations):
    """
    Performs gradient descent to learn theta
    """

    # init useful vars
    m = y.shape[0]  # number of training examples

    cost_history = []  # store cost history in list
    theta = np.zeros((X.shape[1], 1), dtype=np.float64)  # init theta col. vector

    for i in range(iterations):
        """
        Perform a single gradient step on the parameter vector theta
        """

        # compute error
        predictions = utils.sigmoid(X @ theta)  # [m x 1] = [m x n] x [n x 1]
        delta = predictions - y  # [m x 1]

        regularization = (_lambda / m) * theta  # [n x 1]
        regularization[0] = 0  # don't regularize intercept term

        # matrix-vector multiplication
        gradient = (X.T @ delta) / m  # [n x 1] = [n x m] x [m x 1]

        # normalize the gradient to prvent overflow !
        # see http://students.engr.scu.edu/~schaidar/expository/Stochastic_Gradient_Descent.pdf
        # gradient = gradient / np.linalg.norm(gradient) # not for logistic ?

        # multily by learning rate
        gradient = alpha * gradient

        # add regularization term
        gradient = gradient + regularization

        # update theta
        theta = theta - gradient

        # compute cost for current theta
        cost_history.append(computeCost(X=X, y=y, theta=theta, _lambda=_lambda))

    return theta, cost_history


def computeCost(X, y, theta, _lambda):
    """
    Compute the cost and gradient of regularized linear regression
    for a particular choice of theta.
    :param _lambda:
    """

    # init useful vars
    m = y.shape[0]  # number of training examples

    # compute error
    predictions = utils.sigmoid(X @ theta)  # [m x 1] = [m x n] x [n x 1]
    delta = predictions - y  # [m x 1]

    # compute cost
    sum_squared_errors = np.sum(np.square(delta))
    regularization = np.sum(_lambda * np.square(theta[1:]))  # don't compute reg term for intercept term
    mean_squared_error = (sum_squared_errors + regularization) / (2 * m)
    return mean_squared_error


def predictValues(X, theta):
    predictions = utils.sigmoid(X @ theta)  # [m x 1] = [m x n] x [n x 1]

    pos = np.where(predictions >= 0.5)  # index of values >= 0.5
    neg = np.where(predictions < 0.5)

    # set all indexes that were identified as positive to 1
    predictions[pos, 0] = 1  # the [pos, 0] the [ _ , 0] is just a way to index the the value in that position
    predictions[neg, 0] = 0

    return predictions  # [m x 1]

