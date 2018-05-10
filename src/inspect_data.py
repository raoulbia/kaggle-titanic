#!/usr/bin/python3

import pandas as pd
import numpy as np

train=pd.read_csv("../input/train.csv", sep=",", header=0, index_col=0)
test=pd.read_csv("../input/test.csv", sep=",", header=0, index_col=0)

data = train.append(test)

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
print(data.corr())
print("*************")
print("Columns with <10 categories:")
for i in data.columns:
    catdat = pd.Categorical(data[i])
    if len(catdat.categories)>9:
        continue

    print(i," ",pd.Categorical(data[i]))