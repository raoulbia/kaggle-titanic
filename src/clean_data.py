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

# print(train.head())
# print(test.head())

data = train.append(test)
# print(data.head())
# print(data.tail())

# Handle NA values and replace by mean or "usual" value

data.Age.fillna(value=data.Age.mean(), inplace=True)
data.Fare.fillna(value=data.Fare.mean(), inplace=True)
data.Embarked.fillna(value=(data.Embarked.value_counts().idxmax()), inplace=True)
data.Survived.fillna(value=-1, inplace=True) # the test data have NA

# Extract title/salutation from name string

print("Extracting titles and adding column...")
titles = pd.DataFrame(data.apply(lambda x: x.Name.split(",")[1].split(".")[0], axis=1), columns=["Title"])
print(pd.Categorical(titles.Title))
data = data.join(titles)

# Add family size as a combination of the other 2 columns

print("Calculating family size and adding column...")
fsiz = pd.DataFrame(data.apply(lambda x: x.SibSp+x.Parch, axis=1), columns=["FSize"])
data = data.join(fsiz)

# replace columns that are not usable by numeric algorithms and have no use (cabin, ticket...) or have been substituted (parch, sibsp)

# drop useless columns
data.drop('Name', axis=1, inplace=True)
data.drop('Cabin', axis=1, inplace=True)
data.drop('Ticket', axis=1, inplace=True)

# no need for the following as the sum is used
data.drop('Parch', axis=1, inplace=True)
data.drop('SibSp', axis=1, inplace=True)


# generate numerical output
print("Conveting to numerical output...")

for col in data.select_dtypes(exclude=["number"]).columns:
    print("Converting column "+col+"...")
    data[col] = data[col].astype('category')
    print(data[col].cat.categories)
    data[col] = data[col].cat.codes

train = data[data['Survived']!=-9]
train.to_csv("../local-data/train-clean.csv")

test = data[data['Survived']==-9]
test.drop('Survived', axis=1, inplace=True)
test.to_csv("../local-data/test-clean.csv")