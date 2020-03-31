# !usr/bin/python

import pandas as pd
import datetime
import numpy as np
import spacy


# The writing of files takes many hours, after the comparison between different models created by different learning rate,
# it turns out that models with learning rate = 0.002 has the best performance, the final LR.txt is based on that. 
# Therefore, only running code relevant to 0.002 is enough.

start = datetime.datetime.now()
validation_data = pd.read_csv("../part2/validation_data.tsv", skiprows=1, delimiter='\t', header=None, encoding='utf-8')
queries = pd.read_csv("all_queries.tsv", delimiter='\t', header=None, encoding='utf-8').values.tolist()
W = pd.read_csv("LRWeight_10000.txt", skiprows=1, delimiter='\t', index_col=False, header=None, encoding='utf-8')

# W001 = np.matrix(W[2].squeeze()).reshape((600, 1)) # 0.001
W002 = np.matrix(W[1].squeeze()).reshape((600, 1)) # 0.002
# W003 = np.matrix(W[0].squeeze()).reshape((600, 1)) # 0.003

model = spacy.load("en_core_web_md")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def getLRScore(X ,W):
    z = X.dot(W)
    return sigmoid(z)

# LRFile1 = open("LR001.txt", "w", encoding="utf-8").close()
# LRFile1 = open("LR001.txt", "a", encoding="utf-8")

LRFile2 = open("LR002.txt", "w", encoding="utf-8").close()
LRFile2 = open("LR002.txt", "a", encoding="utf-8")

# LRFile3 = open("LR003.txt", "w", encoding="utf-8").close()
# LRFile3 = open("LR003.txt", "a", encoding="utf-8")

for i in range(len(queries)):
    print(i)
    qid = queries[i][0]
    qV = model(str(queries[i][1])).vector
    pcols = [1, 3]
    passages = validation_data[pcols].loc[validation_data[0] == qid].values.tolist()
    length = len(passages)
    #q_pScores001 = []  # store (pid, score) 0.001
    q_pScores002 = [] # store (pid, score) 0.002
    #q_pScores003 = []  # store (pid, score) 0.003
    for p in passages:
        pid = p[0]
        pV = model(str(p[1])).vector
        qpV = np.append(np.array(qV), np.array(pV))
        #score001 = getLRScore(qpV, W001)
        #q_pScores001.append((pid, score001))
        score002 = getLRScore(qpV, W002)
        q_pScores002.append((pid, score002))
        #score003 = getLRScore(qpV, W003)
        #q_pScores003.append((pid, score003))
    #rankScore001 = sorted(q_pScores001, key=lambda x: x[1], reverse=True)
    rankScore002 = sorted(q_pScores002, key=lambda x: x[1], reverse=True)
    #rankScore003 = sorted(q_pScores003, key=lambda x: x[1], reverse=True)
    for i in range(length):
        #LRFile1.write("{}\tA1\t{}\t{}\t{}\tLR\n".format(qid, rankScore001[i][0], (i + 1), str(rankScore001[i][1])[2:-2]))
        LRFile2.write("{}\tA1\t{}\t{}\t{}\tLR\n".format(qid, rankScore002[i][0], (i+1), str(rankScore002[i][1])[2:-2]))
        #LRFile3.write("{}\tA1\t{}\t{}\t{}\tLR\n".format(qid, rankScore003[i][0], (i+1), str(rankScore003[i][1])[2:-2]))
#LRFile1.close()
LRFile2.close()
#LRFile3.close()
end2 = datetime.datetime.now()
print("start: " + start.strftime("%H:%M:%S") + "\tend: " + end2.strftime("%H:%M:%S"))

