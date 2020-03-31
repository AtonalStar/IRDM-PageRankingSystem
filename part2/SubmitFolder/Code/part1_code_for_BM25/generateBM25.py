#!usr/bin/python
# coding:utf-8

import pandas as pd
import re
from operator import itemgetter
import math
import datetime

# generate inverted_index and calculate BM25 ranking together

# The class that define the Inverted index
class Inverted_index:
    qid = 0
    # Init
    def __init__(self, qid):
        self.qid = qid

    # Pre-process the passage collection: tokenisation; lowercase; remove punctuations; remove numbers; remove stopwords.
    def preProcess(self, passages, stopwords):
        preProcessDict = {}
        for i in range(len(passages)):
            wlist = []
            termStr = passages.values[i][1]
            reTerms = re.sub('[^A-Za-z -]+', '', termStr)
            terms = reTerms.lower().split()
            for item in terms:
                if item not in stopwords:
                    if re.sub('[\-]+', '', item) != "":
                        wlist.append(item)
            preProcessDict[str(passages.values[i][0])] = wlist
        return preProcessDict

    # Count term frequency for a term list O(n)
    def countFreq(self, termList):
        wDict = {}
        for i in termList:
            if wDict.get(i) is None:
                wDict[i] = 1
            else:
                wDict[i] = wDict[i] + 1
        return wDict

    # Create inverted index
    def createInvertedIndex(self, passages, stopwords):
        invertedIndex = {}
        tokenPDict = self.preProcess(passages, stopwords)
        for p in tokenPDict:
            freqDict = self.countFreq(tokenPDict[p])
            for k in freqDict:
                if invertedIndex.get(k) is None:
                    pfList = []
                    invertedIndex[k] = pfList
                invertedIndex[k].append((str(p), freqDict[k]))
        return invertedIndex

# Start here
# Calculate the score using BM25 model (iterate through every term in the query)
# The parameters needed are list here:
# qf(i) - the number(frequency) of term i in the query;
# r and R - no relevance information, r = R = 0;
# N - the document total number, len(passages) of a query;
# n(i) - the document frequency, df, of term i of the query, number of documents in the inverted-index entry i
# f(i) - the term frequency (number of terms) of term i in the current document
# dl/avdl - document length / average document length
# k1 = 1.2, b = 0.75, k2 = 100
# K = k1*((1-b)+b*(dl/avdl)) = 1.2 * (0.25 + 0.75*(dl/avdl)) = 0.3 + 0.9*(dl/avdl)
# The formula is For i in QueryTerms: Sum { ln[((r+0.5)/(R-r+0.5)) / ((n(i)-r+0.5)/(N-n(i)-R+r+0.5))] *
# [((k1+1)*f(i)) / (K+f(i))] * [((k2+1)*qf(i)) / (k2+qf(i))] }

start = datetime.datetime.now()
queries = pd.read_csv("all_queries.tsv", delimiter='\t', header=None, encoding='utf-8')
# clear the old file
BMFile = open("../BM25.txt", "w", encoding='utf-8').close()
# create files to write
BMFile = open("../BM25.txt", "a", encoding='utf-8')
# stopwords are articles, be and pronouns (totally 46 words)
stopWords = {u'a', u'an', u'the', u'be', u'been', u'am', u'is', u'are', u'was', u'were', u'isnt', u'arent', u'wasnt',
             u'werent', u'i', u'me', u'my', u'myself', u'you', u'your', u'yours', u'yourself', u'yourselves', u'we',
             u'our', u'ourselves', u'ours', u'they', u'their', u'theirs', u'them', u'themselves', u'he', u'him', u'his',
             u'himself', u'she', u'her', u'hers', u'herself', u'it', u'its', u'itself', u'this', u'these', u'those'}
# sort queries by qid in ascending order
queries = queries.sort_values(by=[0]).values.tolist()
for i in range(len(queries)):
    qid = queries[i][0]
    qStr = queries[i][1]
    qTerms = re.sub("[^A-Za-z -]", "", qStr).split()
    # remove duplicates
    qTerms = list(dict.fromkeys(qTerms))
    # remove stopwords
    qTs = []
    for qt in qTerms:
        if qt not in stopWords:
            if re.sub('[\-]+', '', qt) != "":
                qTs.append(qt)

    passages = pd.read_csv("Passages/passages" + str(qid) + ".tsv", delimiter='\t',
                           header=None, encoding='utf-8')
    invIndex = Inverted_index(qid)
    invIndexDict = invIndex.createInvertedIndex(passages, stopWords)
    # BM25 parameters preparation
    N = len(passages)
    k1 = 1.2
    k2 = 100
    b = 0.75
    r = 0
    R = 0
    # calculate avdl
    allPassages = open("Passages/passages" + str(qid) + ".tsv", "r", encoding='utf-8')
    lines = allPassages.readlines()
    pStr = ' '.join([line.strip() for line in lines])
    pStr = re.sub('[^a-zA-Z]', ' ', pStr)
    avdl = len(pStr.lower().split()) / N

    # Store scores in a list of tuple (pid, score)
    bmScores = []
    for p in range(len(passages)):
        pid = passages.values[p][0]
        pTerms = re.sub("[^A-Za-z -]", "", passages.values[p][1]).split()
        # remove duplicates
        pTerms = list(dict.fromkeys(pTerms))
        # remove stopwords
        pTs = []
        for pt in pTerms:
            pt = pt.lower()
            if pt not in stopWords:
                pTs.append(pt)

        # BM25: calculate dl, K
        pCon = passages.values[p][1]
        pCon = re.sub('[^a-zA-Z]', ' ', pCon)
        dl = len(pCon.lower().split())
        K = k1 * ((1 - b) + b * (dl / avdl))

        # calculate scores
        bmScore = 0
        for item in qTs:
            if item in pTs:
                # BM25 - f(i)
                f = 0
                pfList = invIndexDict[item]
                for pair in pfList:
                    if pair[0] == str(pid):
                        f = pair[1]
                # BM25 - qf(i), n(i)
                qf = qTerms.count(item)
                n = len(pfList)
                bmScore = bmScore + (math.log(((r + 0.5) / (R - r + 0.5)) / ((n - r + 0.5) / (N - n - R + r + 0.5))) * (
                            ((k1 + 1) * f) / (K + f)) * (((k2 + 1) * qf) / (k2 + qf)))
        bmScores.append((str(pid), round(bmScore, 3)))
    # sort Scores
    bmScores.sort(key=itemgetter(1), reverse=True)
    print(i)

    # write to file
    for r in range(len(bmScores)):
        BMFile.write(
            str(qid) + "\tA1\t" + str(bmScores[r][0]) + "\t" + str(r + 1) + "\t" + str(bmScores[r][1]) + "\tBM25\n")
BMFile.close()
end = datetime.datetime.now()

print("start: " + start.strftime("%H:%M:%S") + "\tend: " + end.strftime("%H:%M:%S"))

# About 20 min

