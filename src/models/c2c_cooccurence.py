import cPickle as pickle
import numpy as np
from numpy import random
from random import shuffle
from os.path import expanduser
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","utils")))

from logger import logger

def create_coocc_dict(trainHt):
    coocc_dict = {}
    for htstr in trainHt:
        ht = htstr.split(" ")
        if coocc_dict.get(ht[0]) == None:
            coocc_dict[ht[0]] = {ht[1]:1}
        else:
            if coocc_dict[ht[0]].get(ht[1]) == None:
                coocc_dict[ht[0]][ht[1]] = 1
            else:
                coocc_dict[ht[0]][ht[1]] += 1
    return coocc_dict

def predict(testHt, coocc_dict):
    correct = 0
    name = "cooc" + time.strftime("%Y-%m-%d_%H:%M") + ".log"    
    log = []
    
    for htstr in testHt:
        ht = htstr.split(" ")
        if coocc_dict.get(ht[0]) == None:
            continue
    	dic = coocc_dict[ht[0]]
    	dic_key = dic.keys()
    	dic_val = dic.values()
    	idx = np.argsort(dic_val)
        prediction = []
    	for i in range(topN):
            if i < len(dic_val):
            	prediction.append(dic_key[idx[i]])
        isCorrect = False
        if ht[1] in prediction:
            correct += 1
            isCorrect = True
        
        log.append([ht[0],ht[1],isCorrect,prediction])
    
    accuracy=correct*1.0/len(testHt)
    log.append([correct,accuracy])
    
    logger(log,name) 

hashtags = pickle.load(open(expanduser("~/tweetnet/data/englishHashtag.pkl"), "rb"))
hashtagFreq = pickle.load(open(expanduser("~/tweetnet/data/hashtagFreq.pkl"), "rb"))

idx_shuf = range(len(hashtags))
shuffle(idx_shuf)
freqThreshold = 84
hashtagFreqCnt = {}
hashtags_shuf = []

for i in idx_shuf:
    ht = hashtags[i].split(" ")
    if hashtagFreq[ht[2]] >= freqThreshold:
        if hashtagFreqCnt.get(ht[2]) == None:

            hashtagFreqCnt[ht[2]] = 1
            hashtags_shuf.append(ht[1] + " " + ht[2])

        elif hashtagFreqCnt[ht[2]] < freqThreshold:

            hashtagFreqCnt[ht[2]] += 1
            hashtags_shuf.append(ht[1] + " " + ht[2])

hashtags = hashtags_shuf

trainPercent = 0.95
nTrainData = np.round(len(hashtags)*trainPercent).astype(int)
topN = 4
trainHt = hashtags[0:nTrainData]
testHt = hashtags[nTrainData:]
coocc_dict = create_coocc_dict(trainHt)
predict(testHt, coocc_dict)


