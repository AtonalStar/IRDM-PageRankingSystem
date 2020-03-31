#!usr/bin/python
# coding:utf-8

import codecs
import pandas as pd
import re
from operator import itemgetter
import os
import datetime

# The class that define the Inverted index
class Inverted_index:
    qid = 0

    # Init
    def __init__(self, qid):
        self.qid = qid

    # Pre-process the passage collection: tokenisation; lowercase; remove punctuations; remove numbers; remove stopwords.
    def preProcess(self, passages):
        # stopwords are articles, be and pronouns (totally 46 words)
        stopWords = {u'a', u'an', u'the', u'be', u'been', u'am', u'is', u'are', u'was', u'were', u'isnt', u'arent', u'wasnt', u'werent', u'i', u'me', u'my', u'myself', u'you', u'your', u'yours', u'yourself', u'yourselves', u'we', u'our', u'ourselves', u'ours', u'they', u'their', u'theirs', u'them', u'themselves', u'he', u'him', u'his', u'himself', u'she', u'her', u'hers', u'herself', u'it', u'its', u'itself', u'this', u'these', u'those'}
        preProcessDict = {}
        for i in range(len(passages)):
            wlist = []
            termStr = passages.values[i][1]
            reTerms = re.sub('[^A-Za-z -]+', '', termStr)
            terms = reTerms.lower().split()
            for item in terms:
                if item not in stopWords:
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
    def createInvertedIndex(self):
        passages = pd.read_csv("passages" + str(self.qid) + ".tsv", delimiter='\t', header=None, encoding='utf-8')
        #clear the previous inverted index first
        invIndexClear = codecs.open("invIndex" + str(self.qid) + ".tsv", "w", encoding='utf-8', errors='ignore').close()
        invIndex = codecs.open("invIndex" + str(self.qid) + ".tsv", "a", encoding='utf-8', errors='ignore')
        invertedIndex = {}
        tokenPDict = self.preProcess(passages)
        for p in tokenPDict:
            freqDict = self.countFreq(tokenPDict[p])
            for k in freqDict:
                if invertedIndex.get(k) is None:
                    pfList = []
                    invertedIndex[k] = pfList
                invertedIndex[k].append((int(p), freqDict[k]))
        for term in sorted(invertedIndex):
            invIndex.write(str(term) + ":\t" + str(sorted(invertedIndex[term], key=itemgetter(1), reverse=True)) + "\n")
        invIndex.close()


# Main body that iterates through the 200 qid in test-queries.tsv
# Change the directory to the corresponding qid folder
# Create Inverted_index instance, and call function createInvertedIndex() to write invIndex[qid].tsv to the folder
start = datetime.datetime.now()
queries = pd.read_csv("../dataset/test-queries.tsv", delimiter='\t', header=None, usecols=[0], encoding='utf-8')
originDir = os.getcwd() + '/../invertedResult'
for i in range(len(queries)):
    qid = queries.values[i][0]
    os.chdir(originDir + "/" + str(qid))
    invIndex = Inverted_index(qid)
    invIndex.createInvertedIndex()
    print(i)
end = datetime.datetime.now()
print("start: " + start.strftime("%H:%M:%S") + "\tend: " + end.strftime("%H:%M:%S"))

# Running time: about 1 min 40s
