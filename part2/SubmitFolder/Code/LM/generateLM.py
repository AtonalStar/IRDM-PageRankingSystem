# !usr/bin/python

import pandas as pd
import datetime
import numpy as np
import spacy
import xgboost as xgb

# The final LM.txt is LM-2.txt, only codes about LMF2 is enough, but it still takes 4-5 hours
start = datetime.datetime.now()
validation_data = pd.read_csv("../../part2/validation_data.tsv", skiprows=1, delimiter='\t', header=None, encoding='utf-8')
queries = pd.read_csv("../all_queries.tsv", delimiter='\t', header=None, encoding='utf-8').values.tolist()

model = spacy.load("en_core_web_md")

# LMF1 = open("LM-1.txt", "w", encoding="utf-8").close()
# LMF1 = open("LM-1.txt", "a", encoding="utf-8")
LMF2 = open("LM-2.txt", "w", encoding="utf-8").close()
LMF2 = open("LM-2.txt", "a", encoding="utf-8")
# LMF3 = open("LM-3.txt", "w", encoding="utf-8").close()
# LMF3 = open("LM-3.txt", "a", encoding="utf-8")
# LMF4 = open("LM-4.txt", "w", encoding="utf-8").close()
# LMF4 = open("LM-4.txt", "a", encoding="utf-8")

for i in range(len(queries)):
    print(i)
    qid = queries[i][0]
    qV = model(str(queries[i][1])).vector
    pcols = [1, 3, 4]
    passages = validation_data[pcols].loc[validation_data[0] == qid].values.tolist()
    length = len(passages)
    # Generate test input matrix X -> DMatrix test
    h = length
    w = 600
    X = [[0 for x in range(w)] for y in range(h)]
    Y = [0 for x in range(length)]
    for i in range(length):
        pid = passages[i][0]
        pV = model(str(passages[i][1])).vector
        qpV = np.append(np.array(qV), np.array(pV))
        X[i] = qpV
        Y[i] = passages[i][2]
    X_test = np.array(X)
    X_test = xgb.DMatrix(X_test)
    Y_test = np.array(Y)

    # Load xgb models
    # xgb1 = xgb.Booster()
    # xgb1.load_model("xgb1.json")
    xgb2 = xgb.Booster()
    xgb2.load_model("xgb2.json")
    # xgb3 = xgb.Booster()
    # xgb3.load_model("xgb3.json")
    # xgb4 = xgb.Booster()
    # xgb4.load_model("xgb4.json")

    # make predictions
    # pred1 = xgb1.predict(X_test)
    # scoreList1 = []
    pred2 = xgb2.predict(X_test)
    scoreList2 = []
    # pred3 = xgb3.predict(X_test)
    # scoreList3 = []
    # pred4 = xgb4.predict(X_test)
    # scoreList4 = []

    for i in range(length):
        pid = passages[i][0]
        # scoreList1.append((qid, pid, pred1[i]))
        # scoreList1 = sorted(scoreList1, key=lambda x: x[2], reverse=True)
        scoreList2.append((qid, pid, pred2[i]))
        scoreList2 = sorted(scoreList2, key=lambda x: x[2], reverse=True)
        # scoreList3.append((qid, pid, pred3[i]))
        # scoreList3 = sorted(scoreList3, key=lambda x: x[2], reverse=True)
        # scoreList4.append((qid, pid, pred4[i]))
        # scoreList4 = sorted(scoreList4, key=lambda x: x[2], reverse=True)

    for i in range(length):
        # LMF1.write("{}\tA1\t{}\trank{}\t{}\tLM\n".format(qid, scoreList1[i][1], str(i+1), str(scoreList1[i][2])))
        LMF2.write("{}\tA1\t{}\trank{}\t{}\tLM\n".format(qid, scoreList2[i][1], str(i+1), str(scoreList2[i][2])))
        # LMF3.write("{}\tA1\t{}\trank{}\t{}\tLM\n".format(qid, scoreList3[i][1], str(i + 1), str(scoreList3[i][2])))
        # LMF4.write("{}\tA1\t{}\trank{}\t{}\tLM\n".format(qid, scoreList4[i][1], str(i + 1), str(scoreList4[i][2])))

# LMF1.close()
LMF2.close()
# LMF3.close()
# LMF4.close()

end = datetime.datetime.now()
print("start: " + start.strftime("%H:%M:%S") + "\tend: " + end.strftime("%H:%M:%S"))

# 5 hours