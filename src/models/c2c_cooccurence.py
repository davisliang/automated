import cPickle as pickle
import numpy as np
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
        if coocc_dict.get(ht[1]) == None:
            coocc_dict[ht[1]] = {ht[2]:1}
        else:
            if coocc_dict[ht[1]].get(ht[2]) == None:
                coocc_dict[ht[1]][ht[2]] = 1
            else:
                coocc_dict[ht[1]][ht[2]] += 1
    return coocc_dict

def predict(testHt, coocc_dict):
    correct = 0
    name = "cooc" + time.strftime("%Y-%m-%d_%H:%M") + ".log"    
    log = []
    
    for htstr in testHt:
        ht = htstr.split(" ")
        if coocc_dict.get(ht[1]) == None:
            continue
    	dic = coocc_dict[ht[1]]
    	dic_key = dic.keys()
    	dic_val = dic.values()
    	idx = np.argsort(dic_val)
        prediction = []
    	for i in range(topN):
            if i < len(dic_val):
            	prediction.append(dic_key[idx[i]])
        isCorrect = False
        if ht[2] in prediction:
            correct += 1
            isCorrect = True
        
        log.append([ht[1],ht[2],isCorrect,prediction])
    
    accuracy=correct*1.0/len(testHt)
    log.append([correct,accuracy])
    
    logger(log,name) 

hashtags = pickle.load(open(expanduser("~/tweetnet/data/englishHashtag.pkl"), "rb"))
trainPercent = 0.99
nTrainData = np.round(len(hashtags)*trainPercent).astype(int)
topN = 10
trainHt = hashtags[0:nTrainData]
testHt = hashtags[nTrainData + 1 :]
coocc_dict = create_coocc_dict(trainHt)
predict(testHt, coocc_dict)


