#!/usr/bin/env python

import argparse
import logging
import pandas as pd
pd.options.display.width = 320
# pd.set_option('precision',3)
# pd.set_option('max_rows', 7)

logger = logging.getLogger(name=__name__)

import numpy as np
np.set_printoptions(edgeitems=30, linewidth=100000, precision=3)

from matplotlib import pyplot as plt


import utils
from logistic_regression import *
import utils
import utils_stats
import utils_viz

def model_data(dataset, train_file_path, test_file_path, results_file_path,
               test_size, num_iters, learning_rate,
               cost_history_plot, learning_curve,
               reg_param):

    # get data
    df = utils.read_csv(train_file_path)

    ### Train Logistic Regression
    X, _, y, _ = utils.split_data(df=df, test_size=0, dataset=dataset)

    theta_optimized, cost_history = trainLogReg(X=X,
                                                y=y,
                                                learning_rate=learning_rate,
                                                iterations=num_iters,
                                                reg_param=1)

    logger.info('coeff: {}'.format(theta_optimized.T))

    # plot Cost History
    if cost_history_plot == 1:
        utils_viz.costHistoryPlot(cost_history=cost_history)

    ### Learning Curve
    if learning_curve == 1:
        # split X/y and train/test
        X_train, X_test, y_train, y_test = utils.split_data(df=df, test_size=test_size, dataset=dataset)

        error_train, error_test = learningCurveLogReg(X=X_train,
                                                      y=y_train,
                                                      X_val=X_test,
                                                      y_val=y_test,
                                                      learning_rate=learning_rate,
                                                      iterations=num_iters,
                                                      reg_param=reg_param)

        # print(error_train)
        # print(error_test)
        utils_viz.plot_learning_curve(errors=[error_train, error_test])

    ### Predict on Train data
    p = predictLogReg(X=X, theta=theta_optimized)
    utils.evaluate(y=y, p=p)

    ### Apply to Kaggle test data
    if learning_curve == -1:

        # get data
        X_kaggle_test = utils.read_csv(test_file_path)
        print(X_kaggle_test.shape)
        print(X_kaggle_test.head())

        # get Passenger Ids (for later use)
        X_kaggle_pass_ids = X_kaggle_test['PassengerId']

        # Drop features
        X_kaggle_test = X_kaggle_test.drop(['PassengerId'], axis=1)


        # add intercept terms column
        X_kaggle_test = np.append(np.ones((X_kaggle_test.shape[0], 1)), X_kaggle_test, axis=1)

        ### Predict on Test data
        p = predictLogReg(X=X_kaggle_test, theta=theta_optimized)

        res = pd.concat((X_kaggle_pass_ids, pd.DataFrame(p, dtype=int)), axis=1)
        res.to_csv(results_file_path, index=False, header=['PassengerId', 'Survived'])
        logger.info('Done!')



if __name__ == '__main__':
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str)
    argparser.add_argument('--train-file-path')
    argparser.add_argument('--test-file-path')
    argparser.add_argument('--results-file-path')
    argparser.add_argument('--test-size', type=float)
    argparser.add_argument('--num-iters', type=int)
    argparser.add_argument('--learning-rate', type=float)
    argparser.add_argument('--cost-history-plot', type=int)
    argparser.add_argument('--learning-curve', type=int)
    argparser.add_argument('--reg-param', type=float)
    args = argparser.parse_args()

    model_data(**vars(args))