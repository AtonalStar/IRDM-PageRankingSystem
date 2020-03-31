# !usr/bin/python

import pandas as pd
import spacy
import datetime
import numpy as np
import matplotlib.pyplot as plt

# Logistic regression using V = [query_vector] concat [passage_vector] as the input
# Sigmoid function h_{θ}(x) = 1 / (1 + exp(-x)) where x = V dot W (Weight_matrix)
start = datetime.datetime.now()
# Use the middle 10000 lines of training data 2310000 - 2320000 as they contains 774 (relatively large number) of
# relevant query-passage, which is good to train the model [qid pid query passage relevancy]
data = pd.read_csv("../../part2/train_data.tsv", skiprows=1, delimiter='\t', header=None, encoding='utf-8')[2310000:2320000]
dataList = data.values.tolist()
# "en_core_web_md" is a pre-trained model that contains 20k 300-dimensional word vectors,
# with GloVe trained on Common Crawl - OntoNotes5
model = spacy.load("en_core_web_md")

# Create Logistic Regression Model
#input matrix X
h = len(dataList)
w = 600
X = [[0 for x in range(w)] for y in range(h)]
for i in range(h):
    query = str(dataList[i][2])
    passage = str(dataList[i][3])
    qVector = model(query).vector
    pVector = model(passage).vector
    Vector = np.append(np.array(qVector), np.array(pVector))
    X[i] = Vector
    if i % 100 == 0:
        print(i)
X = np.array(X)
print("X generated")

# output matrix Y - the actual target value
Y = data[4].values.tolist()
Y = np.reshape(Y, [len(dataList), 1])
print("Y generated")

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Logistic Regression model
def lrModel(X, Y, learning_rate):
    # number of iteration
    iterNum = 100
    # Cost
    J = pd.Series(np.arange(iterNum), dtype=float)
    # initialise parameter matrix
    Theta = np.zeros((600, 1))
    for i in range(iterNum):
        # model output
        h = sigmoid(X.dot(Theta))
        # Cost function
        J[i] = np.sum(-Y * np.log(h) - (1 - Y) * np.log(1 - h)) / len(dataList)
        if i % 10 == 0:
            print("iteration = %d, loss = %3f" % (i, J[i]))
        # gradient
        grad = np.dot(X.T, (h - Y))
        Theta -= learning_rate * grad
    return J.squeeze(), Theta

# learning rates to be depended
alpha = [0.005, 0.003, 0.002, 0.001, 0.0005]
modelsCost = {}
Weights = {}
for a in alpha:
    results = lrModel(X, Y, a)
    modelsCost[a] = results[0]
    Weights[str(a)] = results[1]

print(Weights["0.002"][1][0])
for a in alpha:
    plt.plot(modelsCost[a], label=str(a))

# plot the diagram of cost function - iterations
plt.ylabel('cost')
plt.xlabel('iterations')
legend = plt.legend(loc="upper right", shadow=True)
plt.savefig("lr_loss10000.png")

LRWeightFile = open("LRWeight_10000.txt", "w", encoding="utf-8").close()
LRWeightFile = open("LRWeight_10000.txt", "a", encoding="utf-8")

LRWeightFile.write("0.003\t0.002\t0.001\n")
for i in range(0, 600):
    LRWeightFile.write(str(Weights["0.003"][i][0])+"\t"+str(Weights["0.002"][i][0])+"\t"+str(Weights["0.001"][i][0])+"\n")

LRWeightFile.close()

end1 = datetime.datetime.now()
print("start: " + start.strftime("%H:%M:%S") + "\tend: " + end1.strftime("%H:%M:%S"))
