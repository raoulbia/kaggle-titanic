#!/usr/bin/env python

# https://datascienceplus.com/building-a-logistic-regression-in-python-step-by-step/

import pandas as pd
pd.options.display.width = 320
pd.set_option('precision',3)
# pd.set_option('max_rows', 7)

import numpy as np
np.set_printoptions(edgeitems=30, linewidth=100000, precision=3)

from scipy.special import expit
from sklearn.preprocessing import normalize
from matplotlib import pyplot as plt

from src.cost_function import costFunction
from src.cost_function_reg import costFunctionReg
from src.predict import predict
from src.data_stats import explore_categories, explore_corr
from src.data_viz import bar_plot_class_var, missing_values, \
    distribution_by_feature, correlation_features, \
    scatterplot, regplot, pairplot, \
    costHistoryPlot



train=pd.read_csv("../local-data/train-clean.csv", sep=",", header=0, index_col=0)
print(train.head())

### Stats
# explore_categories(data=train, num_cat=4)
explore_corr(data=train)

### Viz
# bar_plot_class_var(data=train)
# pairplot(data=train, features=['Age', 'Pclass', 'Sex',
#                                'Fare', 'Embarked', 'Title',
#                                'FSize', 'IsAlone'], outcome='Survived')
# correlation_features(data=train)

### Drop features
train = train.drop(['Fare'], axis=1)


### Seperate class from features
X = np.matrix(train.ix[:, 1:])
X = np.matrix(normalize(X, axis=0, norm='max'))
print("\nData after normalization:\n", X[:5,:])
y = np.matrix(train.ix[:, :1])


### Logistic Regression

# add intercept terms
a = np.ones((X.shape[0], 1))
X = np.append(a, X, axis=1)

# initial_theta
initial_theta = np.zeros((X.shape[1], 1))
# print('initial theta:\n', initial_theta)
# print('shape initial theta', initial_theta.shape)

# init param
learn = 0.001
reg = 1000

# without regularization
# costHistory, theta_optimized = costFunction(theta=initial_theta, X=X, y=y, learning_rate=learn, num_iters=1500)

# with regularization
costHistory, theta_optimized = costFunctionReg(theta=initial_theta, X=X, y=y, learning_rate=learn, reg_term=reg , num_iters=1500)


# Compute accuracy on our training set
p = predict(X=X, theta=theta_optimized)
# print(p.shape)
print('Train Accuracy:', round(np.mean(p == y),2) * 100)
print('Expected Train Accuracy: 38%')

costHistoryPlot(cost_history=costHistory)



