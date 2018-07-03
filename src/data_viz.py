#!/usr/bin/env python

import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
from matplotlib import pyplot as plt


#### data exploration

def bar_plot_class_var(data):
    sns.countplot(x='Survived',data=data, palette='hls')
    plt.show()

def missing_values(data):
    print(data.isnull().sum())

def distribution_by_feature(data):
    sns.countplot(y="Age", data=data)
    plt.show()

def correlation_features(data):
    # Check the independence between the independent variables
    sns.heatmap(data.corr())
    plt.show()

def scatterplot(data, feature, outcome):
    plt.scatter(x=feature, y=outcome, data=data)
    plt.show()

def regplot(data, feature, outcome):
    sns.regplot(x=feature, y=outcome, data=data)
    plt.show()

def pairplot(data, features, outcome):
    sns.pairplot(data=data,
                 x_vars=features,
                 y_vars=['Survived'])
    plt.show()

def costHistoryPlot(cost_history):
    x = []
    for index, g in enumerate(cost_history):
        x.append(index)
    plt.plot(x, cost_history)
    plt.show()

def plot_learning_curve(errors):
    x = []
    for index, g in enumerate(errors):
        x.append(index)
    plt.plot(x, errors)
    plt.show()