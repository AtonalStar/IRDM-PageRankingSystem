# !usr/bin/python

import pandas as pd
from Metrics import EvaluateMetrics

# This file evaluates models using the evaluate metrics defined in EvaluateMetrics.py

 # BM25
def getBM25Metrics():
    BM25 = pd.read_csv("BM25.txt", delimiter='\t', header=None, encoding='utf-8')
    queries = pd.read_csv("all_queries.tsv", delimiter='\t', header=None, encoding='utf-8')
    queries = queries.sort_values(by=[0]).values.tolist()
    avePrecisionList_bm = [] # List of average precision: (qid, average precision)
    avePreSum_bm = 0
    ndcgList_bm = [] # List of NDCG: (qid, NDCG@100)
    ndcgSum_bm = 0

    for i in range(len(queries)):
        qid = queries[i][0]
        # Evaluate Metrics
        metrics = EvaluateMetrics(BM25, qid)
        ap = metrics.get_avePrecisionByQid()
        avePrecisionList_bm.append((str(qid), round(ap, 3)))
        avePreSum_bm += ap
        ndcg = metrics.get_ndcg()
        ndcgList_bm.append((str(qid), ndcg))
        ndcgSum_bm += ndcg
        if i % 10 == 0:
            print(i)
    print(avePrecisionList_bm)
    mean_average_precision_bm = round(avePreSum_bm / len(queries), 3)
    print(ndcgList_bm)
    mean_ndcg_bm = round(ndcgSum_bm / len(queries), 3)
    print("Mean average precision for BM25 = "+str(mean_average_precision_bm))
    print("Mean NDCG@100 for BM25 = " + str(mean_ndcg_bm))

# LG
def getLRMetrics(lr):
    queries = pd.read_csv("all_queries.tsv", delimiter='\t', header=None, encoding='utf-8').values.tolist()
    avePrecisionList_lr = [] # List of average precision: (qid, average precision)
    avePreSum_lr = 0
    ndcgList_lr = [] # List of NDCG: (qid, NDCG@100)
    ndcgSum_lr = 0

    for i in range(len(queries)):
        qid = queries[i][0]
        # Evaluate Metrics
        metrics = EvaluateMetrics(lr, qid)
        ap = metrics.get_avePrecisionByQid()
        avePrecisionList_lr.append((str(qid), round(ap, 3)))
        avePreSum_lr += ap
        ndcg = metrics.get_ndcg()
        ndcgList_lr.append((str(qid), ndcg))
        ndcgSum_lr += ndcg
        if i % 10 == 0:
            print(i)
    print(avePrecisionList_lr)
    mean_average_precision_lr = round(avePreSum_lr / len(queries), 3)
    print(ndcgList_lr)
    mean_ndcg_lr = round(ndcgSum_lr / len(queries), 3)
    print("Mean average precision for LR = "+str(mean_average_precision_lr))
    print("Mean NDCG@100 for LR = " + str(mean_ndcg_lr))


# LM

# NN

# getBM25Metrics()
lr001 = pd.read_csv("LR001-1.txt", delimiter='\t', header=None, encoding='utf-8')
lr002 = pd.read_csv("LR002-1.txt", delimiter='\t', header=None, encoding='utf-8')
lr003 = pd.read_csv("LR003-1.txt", delimiter='\t', header=None, encoding='utf-8')
getLRMetrics(lr001)
getLRMetrics(lr002)
getLRMetrics(lr003)