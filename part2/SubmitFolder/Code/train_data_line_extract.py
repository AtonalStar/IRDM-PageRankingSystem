# !usr/bin/python

import pandas as pd

train_data = pd.read_csv("part2/train_data.tsv", delimiter='\t', header=None, encoding="utf-8")
print(len(train_data)) # 4364340
lines = train_data.index[train_data[4] == 1].tolist()
print(lines)
# Copy to train_data_line_1.txt

# Extract the number of valid query-passage pairs per 10000 lines of train_data
File = open("validList.txt", "w").close()
File = open("validList.txt", "a")

line_block = 0
lines_idx = 0
validList = []
for x in range(437):
    line_block += 10000
    num = 0
    while (lines_idx < len(lines)) and (int(lines[lines_idx]) < line_block):
        num += 1
        lines_idx += 1
    validList.append((line_block, num))
for item in validList:
    File.write("{}\t{}\n".format(item[0], item[1]))
File.close()