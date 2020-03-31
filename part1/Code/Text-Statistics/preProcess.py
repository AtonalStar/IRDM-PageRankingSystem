# coding:utf-8
# !usr/bin/python

import re
import matplotlib.pyplot as plt
import math
import datetime
import os

start = datetime.datetime.now()
# load the passage_collection for read from dataset
path = os.getcwd()
passage = open(path + '/../dataset/passage_collection_new.txt', 'r', encoding='utf-8', errors='ignore')
# create proportion.txt to write the Zipf's parameters
para = open('proportion.txt', 'a', encoding='utf-8', errors='ignore')

# join the lines in the file into one string
lines = passage.readlines()
pStr = ' '.join([line.strip() for line in lines])
print("Finish lines join")

# remove numbers and punctuations.
# replace characters except a-z A-Z as space
reStr = re.sub('[^a-zA-Z]', ' ', pStr)

# lowercase and split string as term tokens
tList = reStr.lower().split()
print("Finish term list creation")

# Create a dictionary of word-frequency pairs and sort in descending order by the frequency.
# count terms in the term list:
# time complexity: iterate the whole tList once - O(n)
tCountDict = {}
for term in tList:
    if tCountDict.get(term) is not None:
        tCountDict[term] = tCountDict[term] + 1
    else:
        tCountDict[term] = 1

# Sort tCountDict by value
sortedList = [(tCountDict[key], key) for key in tCountDict]
sortedList.sort(reverse=True)
print("Finish creating sorted term count dictionary")

# the total number of terms
totalTerms = len(tList)
# write the Zipf's Law parameter rank, frequency, proportion, rank*proportion for each term
para.write("Term"+"\t"+"Freq" + "\t" + "r" + "\t" + "Pr(%)" + "\t" + "r*Pr\n")
for r in range(len(sortedList)):
    term = sortedList[r][1]
    freq = sortedList[r][0]
    rank = r+1
    pr = freq / totalTerms
    rp = round(rank * pr, 3)
    para.write(term + "\t" + str(freq) + "\t" + str(rank) + "\t" + str(round(pr * 100, 3)) + "\t" + str(rp) + "\n")

# generate the lists for plotting
r = []
logR = []
f_real = []
f_zipf_raw = []
f_zipf = []
p_real = []
p_zipf = []
# For Zipf's law: log(f(i)) = log(c) - log(i)
for i in range (len(sortedList)):
    r.append(i+1)
    logR.append(math.log10(i+1))
    f_real.append(math.log10(sortedList[i][0]))
    # the constant is 6 by f1
    f_zipf.append((-1) * math.log10(i + 1) + 6)
    f_zipf_raw.append((-1) * math.log10(i + 1))
    p_real.append(sortedList[i][0] / totalTerms)
    # the constant is 0.1 by observing "proportion.txt"
    p_zipf.append(0.1/(i+1))
print("Finish loading data for plotting.")

# plotting points as a scatter plot
# The green stars show the real probability of the term;
# The red line shows the probability according to the Zipf's Law

f1 = plt.figure(1)
plt.plot(logR, f_zipf_raw, label="Zipf's Law", color="r")
plt.scatter(logR, f_real, label="Actual Values", color="g", marker="*", s=20)
# x-axis label
plt.xlabel('log(Rank)')
# frequency label
plt.ylabel('log(Frequency)')
# legend
plt.legend(loc='best')
plt.savefig("rank-freq-raw.png")

f2 = plt.figure(2)
plt.plot(logR, f_zipf, label="Zipf's Law", color="r")
plt.scatter(logR, f_real, label="Actual Values", color="g", marker="*", s=20)
# x-axis label
plt.xlabel('log(Rank)')
# frequency label
plt.ylabel('log(Frequency)')
# legend
plt.legend(loc='best')
plt.savefig("rank-freq.png")

f3 = plt.figure(3)
plt.plot(r, p_zipf, label="Zipf's Law", color="r")
plt.scatter(r, p_real, label="Actual Values", color="g", marker="*", s=20)
# x-axis label
plt.xlabel('Rank (by decreasing frequency)')
# frequency label
plt.ylabel('Probability (of occurrence)')
# legend
plt.legend(loc='best')
plt.savefig("rank-prob.png")

passage.close()
para.close()
end = datetime.datetime.now()
print("start: " + start.strftime("%H:%M:%S") + "\tend: " + end.strftime("%H:%M:%S"))

# running time: about 15 s
