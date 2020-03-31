# !usr/bin/python

import pandas as pd

train_data = pd.read_csv("part2/train_data.tsv", delimiter='\t', header=None, encoding="utf-8")
print(train_data.head(10))
cols = [0, 1]
lines = train_data[cols].loc[train_data[4] == 1].values.tolist()
print(train_data.index[train_data[4] == 1].tolist())