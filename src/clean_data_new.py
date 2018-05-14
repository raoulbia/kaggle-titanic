#!/usr/bin/python3
"""
https://www.kaggle.com/sinakhorami/titanic-best-working-classifier?scriptVersionId=566580/code
"""
import re
import pandas as pd
import numpy as np

pd.set_option('display.width', 320)

def get_title(name):
	title_search = re.search(' ([A-Za-z]+)\.', name)
	# If the title exists, extract and return it.
	if title_search:
		return title_search.group(1)
	return ""

train = pd.read_csv("../data/train.csv", sep=",", header=0, index_col=0, dtype={'Age': np.float64})
test = pd.read_csv("../data/test.csv", sep=",", header=0, index_col=0, dtype={'Age': np.float64})
test.insert(loc=0, column='Survived', value=-9)

# print(train.head())
# print(test.head())

data = train.append(test)
print(data.head())
# print(data.info())


data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
# data = data.loc[data['Survived'] != -9]
# print (data[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())

data['IsAlone'] = 0
data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1

data['Embarked'] = data['Embarked'].fillna('S')

data['Fare'] = data['Fare'].fillna(train['Fare'].median())

# title
data['Title'] = data['Name'].apply(get_title)

# age
age_avg = data['Age'].mean()
age_std = data['Age'].std()
age_null_count = data['Age'].isnull().sum()
age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
data['Age'][np.isnan(data['Age'])] = age_null_random_list
data['Age'] = data['Age'].astype(int)


# Mapping Sex
data['Sex'] = data['Sex'].map({'female': 0, 'male': 1}).astype(int)

# Mapping titles
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
data['Title'] = data['Title'].map(title_mapping)
data['Title'] = data['Title'].fillna(0)

# Mapping Embarked
data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

# Mapping Fare
data.loc[data['Fare'] <= 7.91, 'Fare'] = 0
data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare'] = 1
data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'Fare'] = 2
data.loc[data['Fare'] > 31, 'Fare'] = 3
data['Fare'] = data['Fare'].astype(int)

# Mapping Age
data.loc[data['Age'] <= 16, 'Age'] = 0
data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age'] = 1
data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2
data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3
data.loc[data['Age'] > 64, 'Age'] = 4

# # Feature Selection
drop_elements = ['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'FamilySize']
data = data.drop(drop_elements, axis=1)
print(data.head())
print(data.tail())

train = data[data['Survived']!=-9]
train.to_csv("../local-data/train-clean.csv")

test = data[data['Survived']==-9]
test.drop('Survived', axis=1, inplace=True)
test.to_csv("../local-data/test-clean.csv")