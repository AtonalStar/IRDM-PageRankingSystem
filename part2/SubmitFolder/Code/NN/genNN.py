# !usr/bin/python

import pandas as pd
import datetime
import numpy as np
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable

start = datetime.datetime.now()
validation_data = pd.read_csv("../../part2/validation_data.tsv", skiprows=1, delimiter='\t', header=None, encoding='utf-8')
queries = pd.read_csv("../all_queries.tsv", delimiter='\t', header=None, encoding='utf-8').values.tolist()

NNFile = open("NN.txt", "w", encoding="utf-8").close()
NNFile = open("NN.txt", "a", encoding="utf-8")

embed = spacy.load("en_core_web_md")

# NN definition
class NN(nn.Module):
    def __init__(self, input, hidden1, hidden2, output):
        super(NN, self).__init__()
        self.hidden1 = nn.Linear(input, hidden1)
        self.hidden2 = nn.Linear(hidden1, hidden2)
        self.out = nn.Linear(hidden2, output)

    def forward(self, x):
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x = self.out(x)
        return x
# Load NN model
model = torch.load("nn.pkl")

for x in range(len(queries)):
    print(x)
    qid = queries[x][0]
    qV = embed(str(queries[x][1])).vector
    pcols = [1, 3, 4]
    passages = validation_data[pcols].loc[validation_data[0] == qid].values.tolist()
    length = len(passages)
    X = [[0 for x in range(600)] for y in range(length)]
    Y = [0 for x in range(length)]

    for i in range(length):
        pid = passages[i][0]
        pV = embed(str(passages[i][1])).vector
        qpV = np.append(np.array(qV), np.array(pV))
        X[i] = qpV
        Y[i] = passages[i][2]
    X = torch.from_numpy(np.array(X)).type(torch.FloatTensor)
    X = Variable(X)
    Y = torch.tensor(np.array(Y)).type(torch.LongTensor)
    Y = Variable(Y)

    prediction = model(X)

    # Accuracy Prediction
    prediction = torch.max(F.softmax(prediction), 1)[1]
    pred_Y = prediction.data.numpy().squeeze()
    # label_Y = Y.data.numpy()
    # accuracy = sum(pred_Y == label_Y) / Y.size()
    # print("The Accuracy is：", accuracy)

    RelList = []
   
    for i in range(len(pred_Y)):
        if(pred_Y[i] == 1):
            RelList.insert(0, (qid, passages[i][0], pred_Y[i]))
        else:
            RelList.append((qid, passages[i][0], pred_Y[i]))

    for i in range(len(RelList)):
        NNFile.write("{}\tA1\t{}\trank{}\t{}\tNN\n".format(qid, RelList[i][1], str(i + 1), str(RelList[i][2])))


NNFile.close()

end = datetime.datetime.now()
print("start: " + start.strftime("%H:%M:%S") + "\tend: " + end.strftime("%H:%M:%S"))


# 4h