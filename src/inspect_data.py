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

data = train.append(test)
# data = train


print("Data structure:")
print("***************")
print(data.columns)
print(data.dtypes)

print("\nExample:")
print("**********")
print(data.head())

print("\nStatistics:")
print("*************")
print(data.describe())

print("Correlations:")
print("*************")
print(data.corr())

print("*************")
print("Columns with <10 categories:")
for i in data.columns:
    catdat = pd.Categorical(data[i])
    if len(catdat.categories)>9:
        continue

    print(i," ",pd.Categorical(data[i]), '\n')