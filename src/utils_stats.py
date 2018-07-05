#!/usr/bin/python3
import pandas as pd
pd.set_option('display.width', 320)

def explore(data):
    print(data.columns, "\n\n", data.dtypes)


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
