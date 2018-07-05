#!/usr/bin/env python

import pandas as pd
pd.options.display.width = 320

import numpy as np
np.set_printoptions(edgeitems=30, linewidth=100000, precision=3)

import utils_stats
import utils_viz

train_df = pd.read_csv("/home/vagrant/vmtest/github-raoulbia-kaggle-titanic/data/titanic-train.csv")
test_df  = pd.read_csv("/home/vagrant/vmtest/github-raoulbia-kaggle-titanic/data/titanic-test.csv")

# Stats
utils_stats.explore_categories(data=train_df, num_cat=4)
utils_stats.explore_corr(data=train_df)

# Viz
utils_viz.bar_plot_class_var(data=train_df)
utils_viz.pairplot(data=X, features=['Age', 'Pclass', 'Sex',
                               'Fare', 'Embarked', 'Title',
                               'FSize', 'IsAlone'], outcome='Survived')
utils_viz.correlation_features(data=d)