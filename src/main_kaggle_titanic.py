# -*- coding: utf-8 -*-

import logging
import pandas as pd
pd.set_option('display.max_columns', 100)
pd.set_option('precision',3)
# pd.set_option('max_rows', 7)

logger = logging.getLogger(name=__name__)

import numpy as np
np.set_printoptions(edgeitems=30, linewidth=100000, precision=3)
np.set_printoptions(suppress=True)

from matplotlib import pyplot as plt


from .utils import sigmoid
from .logistic_regression import *
from .utils_stats import *
from .utils_viz import *


def model_data(train_file_path,
               test_file_path,
               results_file_path,
               num_iters,
               learn_hyperparameters,
               alpha,
               _lambda):

    # Load (cleaned) Kaggle training data set from CSV file
    kaggle_training_data_df = pd.read_csv(train_file_path, dtype=float)

    # Pull out columns for X (data to train with) and Y (value to predict)
    X = kaggle_training_data_df.drop('Survived', axis=1).values
    y = kaggle_training_data_df[['Survived']].values

    # Add INTERCEPT term
    X = np.append(np.ones((X.shape[0], 1)), X, axis=1)

    # inspect
    logger.info('features\n{}'.format(X[:5, :]))
    logger.info('target\n{}'.format(y[:5]))

    if learn_hyperparameters == 1:
        """
        Manual hyperparamter learning

        NOTE: THIS IS USEFUL FOR LEARNING HOW THINGS WORK BUT IN PRACTICE FINDING BEST HYPERPARAMS MANUALLY IS VERY TRICKY
              PROBABLY BEST TO RESORT TO GRID SEARCH FOR THIS TASK

                Step 1: find a good alpha - recall that lambda is for generalization so to find alpha we don't need lambda
                Step 2: find a good lambda using learned alpha
                Step 3: inspect learning curve using learned hyperparamters
        """

        """ Step 1 Validation Curve alpha

        - If we record the learning at each iteration and plot the learning rate (log) against loss;
        - we will see that as the learning rate increase, there will be a point where the loss stops decreasing
        and starts to increase.
        - in practice, our learning rate should ideally be somewhere to the left to the lowest point of the graph
        source: https://towardsdatascience.com/understanding-learning-rates-and-how-it-improves-performance-in-deep-learning-d0d4059c1c10
        """
        alpha_values = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]
        # validationCurve(X=X, y=y, hp_name='alpha', hp_values=alpha_values, iterations=num_iters)


        """ Step 2 Validation Curve lambda

        - use alpha value from previous step
        - find lambda that gives the lowest cross validation error
        """
        lambda_values = [0.000001, 0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3]
        # validationCurve(X=X, y=y, hp_name='lambda', hp_values=lambda_values, iterations=num_iters, learned_alpha=0.1)

        """ Step 3 Learning Curve
        - use learned values for alpha and lambda
        """
        learningCurve(X=X, y=y, alpha=0.1, _lambda=0.03, iterations=num_iters)


        # verify that cost decreases / converges
        # _, cost_history = gradientDescent(X=X, y=y, alpha=0.001, _lambda=0.03, iterations=num_iters)
        # utils_viz.costHistoryPlot(cost_history=cost_history)

    if learn_hyperparameters == -1:
        """Apply to Kaggle Test data"""

        # Load (cleaned) Kaggle testing data set from CSV file
        kaggle_test_data_df = pd.read_csv(test_file_path, dtype=float)

        # Pull out columns for X (data to train with) and Y (value to predict)
        X_testing = kaggle_test_data_df.drop('PassengerId', axis=1).values

        # add intercept terms column
        X_kaggle_test = np.append(np.ones((X_testing.shape[0], 1)), X_testing, axis=1)

        # Get theta
        theta, _ = gradientDescent(X=X,
                                   y=y,
                                   alpha=alpha,
                                   _lambda=_lambda,
                                   iterations=num_iters)

        ### Predict on Test data
        y_predicted = predictValues(X=X_kaggle_test, theta=theta)

        # Write results to file
        p = pd.DataFrame(y_predicted)
        res = pd.concat((kaggle_test_data_df[['PassengerId']].astype(int), p), axis=1)
        res.to_csv(results_file_path, index=False, header=['PassengerId', 'Survived'])
        logger.info('Done!')



if __name__ == '__main__':
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)

    model_data(train_file_path='/home/vagrant/vmtest/kaggle-titanic/local-data/titanic-train-clean.csv',
               test_file_path='/home/vagrant/vmtest/kaggle-titanic/local-data/titanic-test-clean.csv',
               results_file_path='/home/vagrant/vmtest/kaggle-titanic/local-data/results.csv',
               num_iters= 1500,
               learn_hyperparameters=-1,
               alpha= 0.1,
               _lambda= 0.03
               )