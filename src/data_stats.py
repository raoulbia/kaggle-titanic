#!/usr/bin/python3
"""
source: https://www.kaggle.com/schmitzi/cleaning-titanic-data-and-running-scikitlearn/code
"""
import pandas as pd
import numpy as np

pd.set_option('display.width', 320)

train=pd.read_csv("../data/train.csv", sep=",", header=0, index_col=0)
test=pd.read_csv("../data/test.csv", sep=",", header=0, index_col=0)
test.insert(loc=0, column='Survived', value=-9)

data = train.append(train)
# data = train

def explore(data):
    print(data.columns, "\n\n", data.dtypes)
    print()
    print(data.head())

def explore_stats(data):
    print(data.describe())

def explore_corr(data):
    """
    Compute pairwise correlation of columns, excluding NA/null values

    –1. A perfect downhill (negative) linear relationship
    –0.70. A strong downhill (negative) linear relationship
    –0.50. A moderate downhill (negative) relationship
    –0.30. A weak downhill (negative) linear relationship
     0. No linear relationship
    +0.30. A weak uphill (positive) linear relationship
    +0.50. A moderate uphill (positive) relationship
    +0.70. A strong uphill (positive) linear relationship
    Exactly +1. A perfect uphill (positive) linear relationship
    """
    print("\nCorrelation Coefficients:\n", data.corr(method='pearson'), "\n")

def explore_categories(data, num_cat):
    print("Columns with < %s categories:" % num_cat)
    for i in data.columns:
        catdat = pd.Categorical(data[i])
        if len(catdat.categories)>9:
            continue
        print(i," ",pd.Categorical(data[i]), '\n')
