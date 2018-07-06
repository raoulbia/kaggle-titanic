#!/usr/bin/python3
"""
source: https://github.com/Shreyas3108/house-price-prediction/blob/master/housesales.ipynb
"""
import re
import pandas as pd
import numpy as np

from scipy.stats import skew

pd.set_option('display.width', 320)
pd.set_option('display.max_columns', None)

data = pd.read_csv("/home/vagrant/vmtest/github-raoulbia-kaggle-titanic/data/house-price-train-toy.csv")
print(data.head())

labels = data['price']
conv_dates = [1 if values == 2014 else 0 for values in data.date ]
data['date'] = conv_dates
train = data.drop(['id', 'date', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long'],axis=1)

print(train.head())



train.to_csv("/home/vagrant/vmtest/github-raoulbia-kaggle-titanic/local-data/house-price-train-toy-clean.csv", index=False)

print('Done!')