# !usr/bin/python

import pandas as pd
import math

# This file calculate the average precision and NDCG metrics for each query
class EvaluateMetrics:
    # Model File DataFrame [qid, A1, pid, rank, score, algoName]
    model = pd.DataFrame()
    qid = 0
    ModelRank = [] # Extract [pid] where [qid] = qid from model

    # Candidate data [qid, pid, query, passage, relevancy]
    query_candidate = pd.read_csv("../part2/validation_data.tsv", skiprows=1, delimiter='\t',
                                  header=None, encoding='utf-8')

    # for IDCG
    bestRank = [] # list of [pid] for the largest DCG

    # init
    def __init__(self, model, qid):
        self.model = model
        self.qid = qid

        mcols = [2]
        self.ModelRank = model[mcols].loc[model[0] == qid]
        self.ModelRank.columns = ['pid']

        qcols = [1, 4]
        relevancy = self.query_candidate[qcols].loc[self.query_candidate[0] == qid]
        relevancy.columns = ['pid', 'rel']

        self.ModelRank = pd.merge(self.ModelRank, relevancy, on='pid')
        self.bestRank = self.ModelRank.sort_values(by='rel', ascending=False).reset_index(drop=True)



    # Arguments for the average precision: (Sum of p_relevant / p_retrieve [when isRelevant=1.0]) / totalRelevant
    # 1) current number of retrieved passages - p_retrieve;
    # 2) current number of retrieved passages that are relevant - p_relevant;
    # 3) whether the new retrieved passage is relevant (relevancy - 0.0 or 1.0) - isRelevant;
    # 4) total number of retrieved passages that are relevant - totalRelevant;

    # calculate average precisions for a single query
    def get_avePrecisionByQid(self):
        # average precision arguments
        p_retrieve = 0
        p_relevant = 0
        totalRelevant = 0
        precisionSum = 0
        for index, row in self.ModelRank.iterrows():
            p_retrieve += 1
            isRelevant = row['rel']
            if (isRelevant != 0):
                p_relevant += 1
                totalRelevant += 1
                precisionSum += p_relevant / p_retrieve
        if(totalRelevant==0):
            return 0
        else:
            return round(precisionSum / totalRelevant, 3)

    # Arguments for the NDCG metrics: DCG@k / IDCG@k
    # 1) DCG@k = Sum of (2^rel_{i}-1 / log2(i+1));
    # 2) IDCG@k - the largest DCG up to top k ranks: by sorting the ranking by true relevancy first.
    # This implementation will calculate the metrics for top 100 results. - k=100

    # get optimal rank
    #def get_newRank(self):


    # calculate DCGs of the top 100 ranks, return a list of DCGs
    def get_dcgByQid(self, ranking):
        DCG = 0
        for i in range(len(ranking)):
            rank = i + 1
            if(rank <= 100):
                rel = ranking.get('rel')[i]
                if(rel != 0):
                    DCG = DCG + round((math.pow(2, rel) - 1) / math.log2(rank + 1), 3)
            else:
                break
        return DCG

    # calculate NDCG for the top 100 ranks
    def get_ndcg(self):
        DCG = self.get_dcgByQid(self.ModelRank)
        IDCG = self.get_dcgByQid(self.bestRank)
        if (IDCG == 0):
            return 0
        else:
            NDCG = round(DCG / IDCG, 3)
            return NDCG
