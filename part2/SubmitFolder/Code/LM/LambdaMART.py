#! usr/bin/python

import pandas as pd
import xgboost as xgb
from xgboost import DMatrix
import spacy
import numpy as np
# The final LM.txt is LM-2.txt, so only run codes about params2 is enough to get the final result.

# Use the middle 10000 lines of training data 2310000 - 2320000 as they contains 87 (relatively large number) of
# relevant query-passage pairs, which is good to train the model [qid pid query passage relevancy]
data = pd.read_csv("../../part2/train_data.tsv", skiprows=1, delimiter='\t', header=None, encoding='utf-8')[2310000:2320000]
dataList = data.sort_values(by=[0]).values.tolist()

# spacy embedding model
model = spacy.load("en_core_web_md")

# Training data
X_data = pd.read_csv("../embed10000.tsv", delimiter='\t', header=None, encoding='utf-8')
X_data.drop(X_data.columns[[600]], axis=1, inplace=True)

X_data = X_data.T
print(X_data.shape)

#input matrix X
h = len(dataList)
w = 600
X = [[0 for x in range(w)] for y in range(h)]
for i in range(h):
    X[i] = X_data[i]
X_train = np.array(X)
print("X generated")

# output matrix Y - the actual target value
Y = data[4].values.tolist()
Y_train = np.reshape(Y, [len(dataList), 1])
print("Y generated")

train_data = DMatrix(X_train, Y_train)

# eta is learning rate; objective: 'rank:ndcg' - Use LambdaMART to perform list-wise ranking
# use default
# params1 = {'max_depth': 6, 'min_child_weight': 1, 'gamma': 0, 'eta': 0.3, 'objective':'rank:ndcg'}
# xgb_model1 = xgb.train(params1, train_data, num_boost_round=6)
# xgb_model1.save_model('xgb1.json')

# different learning rate eta
params2 = {'max_depth': 6, 'min_child_weight': 1, 'gamma': 0, 'eta': 0.03, 'objective':'rank:ndcg'}
xgb_model2 = xgb.train(params2, train_data, num_boost_round=60)
xgb_model2.save_model('xgb2.json')

# params3 = {'max_depth': 6, 'min_child_weight': 1, 'gamma': 0, 'eta': 0.003, 'objective':'rank:ndcg'}
# xgb_model3 = xgb.train(params3, train_data, num_boost_round=600)
# xgb_model3.save_model('xgb3.json')

# parameter tuning to prevent overfiting:
# add randomness to make training robust to noise:
# subsample=0.7 (default 1);  colsample_bytree=0.7
# params4 = {'max_depth': 6, 'min_child_weight': 1, 'gamma': 0, 'eta': 0.3, 'subsample': 0.7, 'colsample_bytree': 0.7, 'objective':'rank:ndcg'}
# xgb_model4 = xgb.train(params4, train_data, num_boost_round=10)
# xgb_model4.save_model('xgb4.json');

