import os
from os.path import expanduser

c2c = open(expanduser("~/tweetnet/logs/Feb/c2c2017-03-04_13:19.log"), "rb")
t2c = open(expanduser("~/tweetnet/logs/t2c2017-03-04_13:42.log"), "rb")

cnt = 0

correctDic = {}

lines = c2c.read()
lines = lines.split("\n\n")

for blocks in lines:
    if blocks[0:5] == "input":
        blocks = blocks.split("\n")
        if "True" in blocks[2]:
            correctDic[cnt] = 1
        else:
            correctDic[cnt] = 0
    cnt += 1

cnt = 0
lines = t2c.read()
lines = lines.split("\n\n")

for blocks in lines:
    if blocks[0:5] == "input":
        blocks = blocks.split("\n")
        if "True" in blocks[2]:
            if correctDic[cnt] == 0:
                correctDic[cnt] = 1
    cnt += 1

accuracy = sum(correctDic.values()) * 1.0/ len(correctDic) 
print accuracy
