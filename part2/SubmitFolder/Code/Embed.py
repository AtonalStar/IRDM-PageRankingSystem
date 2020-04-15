#!usr/bin/python

import pandas as pd
import spacy

# Generate the file contains training data embedding result for easy training later.
# Use the middle 10000 lines of training data 2310000 - 2320000
data = pd.read_csv("../part2/train_data.tsv", skiprows=1, delimiter='\t', header=None, encoding='utf-8')[
       2310000:2320000]
dataList = data.sort_values(by=[0]).values.tolist()
# spacy embedding model
# "en_core_web_md" is a pre-trained model that contains 20k 300-dimensional word vectors,
# with GloVe trained on Common Crawl - OntoNotes5
embed = spacy.load("en_core_web_md")
Embed = open("embed10000.tsv", "w", encoding="utf-8").close()
Embed = open("embed10000.tsv", "a", encoding="utf-8")

# Generate training data
# input matrix X (numpy array)
h = len(data)
w = 600
X = [[0 for x in range(w)] for y in range(h)]
for i in range(h):
    query = str(dataList[i][2])
    passage = str(dataList[i][3])
    qVector = embed(query).vector
    pVector = embed(passage).vector
    for x in range(300):
        Embed.write("{}\t{}\t".format(qVector[x], pVector[x]))
    Embed.write("\n")

    if i % 100 == 0:
        print(i)

Embed.close()
