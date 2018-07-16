#!/usr/bin/env python

import logging
logger = logging.getLogger(name=__name__)

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 100)
pd.set_option('precision',3)
# pd.set_option('max_rows', 7)

from sklearn import model_selection
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from numpy import exp
from scipy.special import expit


def sigmoid(p):
    return expit(p)


def evaluate(y, p):

    average_precision = average_precision_score(y_true=y, y_score=p)
    print('Average precision-recall score: {0:0.2f}'.format(average_precision))

    cnf_matrix = pd.DataFrame(confusion_matrix(y_true=y, y_pred=p, labels=[1, 0]),
                              index=['true:yes', 'true:no'],
                              columns=['pred:yes', 'pred:no'])

    logger.info('\nConfusion Matrix:\n {}'.format(cnf_matrix))



def normalize_features(df):
    ## Normalize the features in the data set.
    mu = df.mean()

    sigma = df.std()

    if (sigma == 0).any():
        raise Exception("One or more features had the same value for all samples, and thus could " + \
                        "not be normalized. Please do not include features with only a single value " + \
                        "in your model.")
    df_normalized = (df - df.mean()) / df.std()

    return df_normalized, mu, sigma