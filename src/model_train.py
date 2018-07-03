#!/usr/bin/env python

# https://datascienceplus.com/building-a-logistic-regression-in-python-step-by-step/

# about normalization
# https://stats.stackexchange.com/questions/48360/is-standardization-needed-before-fitting-logistic-regression


import argparse
import logging
import pandas as pd
pd.options.display.width = 320
pd.set_option('precision',3)
# pd.set_option('max_rows', 7)

logger = logging.getLogger(name=__name__)

import numpy as np
np.set_printoptions(edgeitems=30, linewidth=100000, precision=3)

from matplotlib import pyplot as plt


from utils import read_csv, split_titanic_data, evaluate
from cost_function_reg import costFunctionReg
from predict import predict
from data_stats import explore_categories, explore_corr
from data_viz import bar_plot_class_var, missing_values, \
    distribution_by_feature, correlation_features, \
    scatterplot, regplot, pairplot, \
    costHistoryPlot


def model_data(train_file_path, test_file_path, results_file_path,
               test_size, num_iters, learning_rate, reg_term,
               apply_to_kaggle):

    # get data
    X = read_csv(train_file_path)

    # Drop features
    X = X.drop(['PassengerId'], axis=1)
    print(X.head())

    # Stats
    # explore_categories(data=train, num_cat=4)
    explore_corr(data=X)

    # Viz
    # bar_plot_class_var(data=X)
    # pairplot(data=X, features=['Age', 'Pclass', 'Sex',
    #                                'Fare', 'Embarked', 'Title',
    #                                'FSize', 'IsAlone'], outcome='Survived')
    # correlation_features(data=X)




    # tain/test split
    X_train, X_test, y_train, y_test = split_titanic_data(X, test_size=test_size)


    costPlot = []


    ### Train Logistic Regression

    # add intercept terms column
    X_train = np.append(np.ones((X_train.shape[0], 1)), X_train, axis=1)

    # initial_theta
    initial_theta = np.zeros((X_train.shape[1], 1))
    # print('shape initial theta', initial_theta.shape)

    # get cost and optimized theta
    costHistory, theta_optimized = costFunctionReg(theta=initial_theta, X=X_train, y=y_train,
                                                   learning_rate=learning_rate,
                                                   reg_term=reg_term,
                                                   iterations=num_iters)

    # print optimized feature weights
    logger.info('coeff: {}'.format(theta_optimized.T))

    # plot Cost History
    # costHistoryPlot(cost_history=costHistory)



    ### Predict on X_train
    p = predict(X=X_train, theta=theta_optimized)
    evaluate(y=y_train, p=p)


    ### Predict on X_test
    # add intercept terms column
    X_test = np.append(np.ones((X_test.shape[0], 1)), X_test, axis=1)
    p = predict(X=X_test, theta=theta_optimized)
    evaluate(y=y_test, p=p)


    ### Apply to Kaggle test data

    if apply_to_kaggle == 1:

        # get data
        X_kaggle_test = read_csv(test_file_path)
        print(X_kaggle_test.shape)

        # get Passenger Ids (for later use)
        X_kaggle_pass_ids = X_kaggle_test['PassengerId']

        # Drop features
        X_kaggle_test = X_kaggle_test.drop(['PassengerId'], axis=1)


        # add intercept terms column
        X_kaggle_test = np.append(np.ones((X_kaggle_test.shape[0], 1)), X_kaggle_test, axis=1)
        p = predict(X=X_kaggle_test, theta=theta_optimized)

        res = pd.concat((X_kaggle_pass_ids, pd.DataFrame(p, dtype=int)), axis=1)
        res.to_csv(results_file_path, index=False, header=['PassengerId', 'Survived'])
        logger.info('Done!')



if __name__ == '__main__':
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train-file-path')
    argparser.add_argument('--test-file-path')
    argparser.add_argument('--results-file-path')
    argparser.add_argument('--test-size', type=float)
    argparser.add_argument('--num-iters', type=int)
    argparser.add_argument('--learning-rate', type=float)
    argparser.add_argument('--reg-term', type=int)
    argparser.add_argument('--apply-to-kaggle', type=int)
    args = argparser.parse_args()

    model_data(**vars(args))