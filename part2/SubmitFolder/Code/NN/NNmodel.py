#! usr/bin/python

import pandas as pd
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data

# Build Neural Network Model
class NN(nn.Module):
    def __init__(self, input, hidden1, hidden2, output):
        super(NN, self).__init__()
        self.hidden1 = nn.Linear(input, hidden1)
        self.hidden2 = nn.Linear(hidden1, hidden2)
        self.out = nn.Linear(hidden2, output)

    def forward(self, x):
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x = self.out(x)
        return x

start = datetime.datetime.now()
# Use the middle 10000 lines of training data 2310000 - 2320000 as they contains 87 (relatively large number) of
# relevant query-passage, which is good to train the model [qid pid query passage relevancy]
data = pd.read_csv("../../part2/train_data.tsv", skiprows=1, delimiter='\t', header=None, encoding='utf-8')[2310000:2320000]
dataList = data.sort_values(by=[0]).values.tolist()

# Training data
X_data = pd.read_csv("../embed10000.tsv", delimiter='\t', header=None, encoding='utf-8')
X_data.drop(X_data.columns[[600]], axis=1, inplace=True)
X_data = X_data.T
print(X_data.shape)

# Generate training data
# input matrix X (numpy array)
print("start make X")
h = len(dataList)
w = 600
X = [[0 for x in range(w)] for y in range(h)]
for i in range(h):
    X[i] = X_data[i]
X = np.array(X)
print("X generated")

# target matrix - Y (numpy array)
Y = data[4].values.tolist()
Y = np.reshape(Y, [len(dataList), 1])
print("Y generated")

# Change numpy array to torch Tensor Float type or Long type
X = torch.from_numpy(X)
Y = torch.from_numpy(Y).squeeze(1)
X = X.type(torch.FloatTensor)
Y = Y.type(torch.LongTensor)
# print(X.size())
# print(Y.size())

# Load data
train_dataset = Data.TensorDataset(X, Y)
train_loader = Data.DataLoader(
    dataset = train_dataset,
    batch_size = 1
)
print("Finish Load data")

# Set parameters and loss function: input is the number feature dimensions (600) of X, output is number of Y type (0 / 1)
model = NN(600, 256, 64, 2)
print(model)
criterion = nn.CrossEntropyLoss()


# Training
# 4 epoch, each epoch iterates all entries of training data
epoch_num = 15
J = pd.Series(np.arange(epoch_num), dtype=float)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(epoch_num):
    Y_train_loss = []
    for step, (X, Y) in enumerate(train_loader):
        out_data = model.forward(X)
        loss = criterion(out_data, Y) # current iteration's loss
        Y_train_loss.append(loss)

        optimizer.zero_grad() # clear the last gradient
        loss.backward() # calculate the gradient using the loss function
        optimizer.step() # update the weight
    J[epoch] = np.sum(Y_train_loss) / 10000
    J[epoch] = J[epoch].item()
    print("The average loss of epoch {} is {};".format(epoch+1, J[epoch]))

plt.plot(J)
# plot the diagram of loss - epoch
plt.ylabel('loss')
plt.xlabel('epoch')
plt.savefig("nnLoss.png")

torch.save(model, 'nn.pkl')

end = datetime.datetime.now()
print("Start: {}, End: {}.".format(start, end))