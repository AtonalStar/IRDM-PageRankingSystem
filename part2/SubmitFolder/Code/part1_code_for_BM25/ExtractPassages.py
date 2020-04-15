#!usr/bin/python
# coding:utf-8

import codecs
import pandas as pd
import os
import datetime

start = datetime.datetime.now()
# extract all query id
all_queries = codecs.open("../all_queries.tsv", "w", encoding='utf-8', errors='ignore').close()
all_queries = codecs.open("../all_queries.tsv", "a", encoding='utf-8', errors='ignore')
# get passage collection from dataset
query_candidate = pd.read_csv("../../part2/validation_data.tsv", skiprows=1, delimiter='\t', header=None,
                              usecols=[0, 1, 2, 3],
                              encoding='utf-8')
# store (qid: (pid, p)) dictionary
candidateDict = {}

# arrange passages by query in a (qid : (pid - passage)) dictionary
qcols = [0, 2]
query = query_candidate[qcols].drop_duplicates().values.tolist()
# sort queries by qid in ascending order
query = sorted(query, key=lambda x: x[0])
# pid - passage
pcols = [1, 3]
num = 0
for q in query:
    all_queries.write(str(q[0])+"\t"+ str(q[1])+"\n")
    candidateDict[str(q[0])] = query_candidate[pcols].loc[query_candidate[0] == q[0]].values.tolist()
    print(num)
    num = num + 1

# Write the 1000 passages for each query in separate files called passages[qid].tsv and store it in the corresponding
# directory.
originDir = os.getcwd() + '/Passages'
d_index = 0
for k in candidateDict:
    os.chdir(originDir)
    passagesClear = codecs.open("passages" + k + ".tsv", "w", encoding='utf-8', errors='ignore').close()
    passages = codecs.open("passages" + k + ".tsv", "a", encoding='utf-8', errors='ignore')
    for item in candidateDict[k]:
        passages.write(str(item[0]) + "\t" + str(item[1]) + "\n")
    passages.close()
    print(d_index)
    d_index = d_index + 1

end = datetime.datetime.now()

print("start: " + start.strftime("%H:%M:%S") + "\tend: " + end.strftime("%H:%M:%S"))
# Running time: About 1 min
