# !usr/bin/python

import pandas as pd
from Metrics import EvaluateMetrics

# This file evaluates models using the evaluate metrics defined in EvaluateMetrics.py

# Calculate from ranking files
def getMetrics(rank):
    queries = pd.read_csv("all_queries.tsv", delimiter='\t', header=None, encoding='utf-8').values.tolist()
    avePrecisionList = []  # List of average precision: (qid, average precision)
    avePreSum = 0
    ndcgList = []  # List of NDCG: (qid, NDCG@100)
    ndcgSum = 0

    for i in range(len(queries)):
        qid = queries[i][0]
        # Evaluate Metrics
        metrics = EvaluateMetrics(rank, qid)
        ap = metrics.get_avePrecisionByQid()
        avePrecisionList.append((str(qid), round(ap, 3)))
        avePreSum += ap
        ndcg = metrics.get_ndcg()
        ndcgList.append((str(qid), ndcg))
        ndcgSum += ndcg
        if i % 10 == 0:
            print(i)
    print(avePrecisionList)
    mean_average_precision = round(avePreSum / len(queries), 3)
    print(ndcgList)
    mean_ndcg = round(ndcgSum / len(queries), 3)
    return mean_average_precision, mean_ndcg

# BM25 (part1 model)
BM25 = pd.read_csv("part1_code_for_BM25/BM25.txt", delimiter='\t', header=None, encoding='utf-8')
metrics = getMetrics(BM25)
print("Mean average precision for BM25 = "+str(metrics[0]))
print("Mean NDCG@100 for BM25 = " + str(metrics[1]))
#######################################################################################################

# LR
def getLRMetrics(lr):
    metrics = getMetrics(lr)
    print("Mean average precision for LR = "+str(metrics[0]))
    print("Mean NDCG@100 for LR = " + str(metrics[1]))

# LM
def getLMMetrics(lm):
    metrics = getMetrics(lm)
    print("Mean average precision for LM = "+str(metrics[0]))
    print("Mean NDCG@100 for LM = " + str(metrics[1]))

# NN
def getNNMetrics(nn):
    metrics = getMetrics(nn)
    print("Mean average precision for NN = "+str(metrics[0]))
    print("Mean NDCG@100 for NN = " + str(metrics[1]))


# LR
lr001 = pd.read_csv("LR/LR001-1.txt", delimiter='\t', header=None, encoding='utf-8')
lr002 = pd.read_csv("LR/LR002-1.txt", delimiter='\t', header=None, encoding='utf-8') # Final LR.txt
lr003 = pd.read_csv("LR/LR003-1.txt", delimiter='\t', header=None, encoding='utf-8')
getLRMetrics(lr001)
getLRMetrics(lr002)
getLRMetrics(lr003)

# LM
lm1 = pd.read_csv("LM/LM-1.txt", delimiter='\t', header=None, encoding='utf-8')
lm2 = pd.read_csv("LM/LM-2.txt", delimiter='\t', header=None, encoding='utf-8') # Final LM.txt
lm3 = pd.read_csv("LM/LM-3.txt", delimiter='\t', header=None, encoding='utf-8')
getLMMetrics(lm1)
getLMMetrics(lm2)
getLMMetrics(lm3)

# NN
nn = pd.read_csv("NN/NN.txt", delimiter='\t', header=None, encoding='utf-8') # Final NN.txt
getNNMetrics(nn)