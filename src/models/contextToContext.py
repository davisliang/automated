from os.path import expanduser
import os
import sys
import numpy
from numpy import shape
from numpy import random
from random import shuffle
import cPickle as pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","utils")))
import time
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.layers import BatchNormalization
import keras.callbacks
from logger import logger
from predContext import predContext, createHtDict
from keras.layers import PReLU
hashtags = pickle.load(open(expanduser("~/tweetnet/data/englishHashtag.pkl"),"rb"))
dictionary = pickle.load(open(expanduser("~/tweetnet/data/word2vec_dict.pkl"), "rb"))
hashtagFreq = pickle.load(open(expanduser("~/tweetnet/data/hashtagFreq.pkl"), "rb"))

idx_shuf = range(len(hashtags))
shuffle(idx_shuf)
freqThreshold = 84
hashtags_shuf = []
context_shuf = []
hashtagFreqCnt = {}

for i in idx_shuf:
    ht = hashtags[i].split(" ")
    if hashtagFreq[ht[2]] >= freqThreshold:
	if hashtagFreqCnt.get(ht[2]) == None:

            hashtagFreqCnt[ht[2]] = 1
            hashtags_shuf.append(ht[2])
            context_shuf.append(ht[1])

        elif hashtagFreqCnt[ht[2]] < freqThreshold:

            hashtagFreqCnt[ht[2]] += 1
            hashtags_shuf.append(ht[2]) 
            context_shuf.append(ht[1])

data = numpy.zeros([len(hashtags_shuf),300])
label = numpy.zeros([len(hashtags_shuf),300])
inputStringLabel = []
outputStringLabel = []
for i in range(len(hashtags_shuf)):
    data[i,:]=dictionary[context_shuf[i]]
    label[i,:]=dictionary[hashtags_shuf[i]]
    inputStringLabel.append(context_shuf[i])
    outputStringLabel.append(hashtags_shuf[i])

htDic = createHtDict(dictionary, outputStringLabel)

# Train and Test split
trainPercent = 0.95
nTrainData = numpy.round(len(data)*trainPercent).astype(int)
topN = 4
nEpoch = 5000
logAllPredictions = True
trainData = data[0 : nTrainData]
testData = data[nTrainData :]
testInputStringLabel = inputStringLabel[nTrainData:]
print testData.shape
trainLabel = label[0 : nTrainData]
testOutputStringLabel = outputStringLabel[nTrainData:]


model = Sequential()

model.add(Dense(512, input_shape=(300,)))
model.add(PReLU())
model.add(BatchNormalization())

model.add(Dense(512))
model.add(PReLU())
model.add(BatchNormalization())

model.add(Dense(300))
model.add(PReLU())

optimizer = RMSprop(lr=0.005)
model.compile(loss='mse', optimizer=optimizer)

name = "c2c" + time.strftime("%Y-%m-%d_%H:%M") + ".log"
for epoch in range(nEpoch):
    model.fit(trainData, trainLabel, nb_epoch=1, batch_size=128, validation_split=0.1)
    
    correctCnt = 0
    randIdx = numpy.random.randint(0, len(testData), 10)
    log = []    
    log.append([epoch]) 
    for testIdx in range(len(testData)):
	modelOutput = model.predict(numpy.expand_dims(testData[testIdx, :], axis=0))
	topNht, isCorrect, topNdist = predContext(htDic, modelOutput, topN, testOutputStringLabel[testIdx])
	if isCorrect:
            correctCnt += 1.0
        if logAllPredictions:
            #verbose logging
	    log.append([testInputStringLabel[testIdx],testOutputStringLabel[testIdx],isCorrect,topNht])
    
    accuracy = correctCnt*1.0 / len(testData)
    #always log accuracy
    log.append([correctCnt, accuracy])
    logger(log,name)
        
