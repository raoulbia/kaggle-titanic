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

def split_titanic_data(df, test_size):
        # print(df.head())
        X = np.matrix(df.ix[:, 1:])
        y = np.matrix(df['Survived']).T
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