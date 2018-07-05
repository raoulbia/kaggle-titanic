#!/usr/bin/env python

import logging
logger = logging.getLogger(name=__name__)

import numpy as np
import pandas as pd
pd.options.display.width = 320
pd.set_option('precision',3)
# pd.set_option('max_rows', 7)

from sklearn import model_selection
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from numpy import exp
from scipy.special import expit

def read_csv(train_file_path):
        df = pd.read_csv(train_file_path)
        return df

def split_data(df, test_size, dataset):
    X = np.zeros(shape=(df.shape[0], df.shape[1]))
    y = np.zeros(shape=(df.shape[0], 1))
    if dataset == 'titanic':
        X = np.matrix(df.ix[:, 1:])
        y = np.matrix(df['Survived']).T
    elif dataset == 'diabetes':
        X = np.matrix(df.ix[:, : -1])
        y = np.matrix(df['Outcome']).T

    # add intercept term
    X = np.append(np.ones((X.shape[0], 1)), X, axis=1)
    return model_selection.train_test_split(X, y, test_size=test_size)


def sigmoid(p):
        return expit(p)

def evaluate(y, p):

        average_precision = average_precision_score(y_true=y, y_score=p)
        print('Average precision-recall score: {0:0.2f}'.format(average_precision))

        cnf_matrix = pd.DataFrame(confusion_matrix(y_true=y, y_pred=p, labels=[1, 0]),
                                  index=['true:yes', 'true:no'],
                                  columns=['pred:yes', 'pred:no'])

        logger.info('\nConfusion Matrix:\n {}'.format(cnf_matrix))