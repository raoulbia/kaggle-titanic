import pandas as pd
pd.set_option('display.width', 1000)

train_df = pd.read_csv('../data/train.csv')

print(train_df.head())