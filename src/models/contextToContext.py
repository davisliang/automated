from os.path import expanduser
import os
import sys
import numpy
from numpy import shape
import cPickle as pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","utils")))

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.optimizers import SGD
from keras.layers import BatchNormalization
import keras.callbacks
from predContext import predContext, createHtDict

hashtags = pickle.load(open(expanduser("~/tweetnet/data/englishHashtag.pkl"),"rb"))
dictionary = pickle.load(open(expanduser("~/tweetnet/data/word2vec_dict.pkl"), "rb"))

data = numpy.zeros([len(hashtags),300])
label = numpy.zeros([len(hashtags),300])
inputStringLabel = []
outputStringLabel = []
for i in range(len(hashtags)):
    line = hashtags[i]
    listHashtag = line.split()
    data[i,:]=dictionary[listHashtag[1]]
    label[i,:]=dictionary[listHashtag[2]]
    inputStringLabel.append(listHashtag[1])
    outputStringLabel.append(listHashtag[2])

htDic = createHtDict(dictionary, outputStringLabel)

# Train and Test split
trainPercent = 0.99
nTrainData = numpy.round(len(data)*trainPercent).astype(int)
topN = 5
nEpoch = 5000

trainData = data[0 : nTrainData]
testData = data[nTrainData + 1 :]
testInputStringLabel = inputStringLabel[nTrainData + 1 :]
print testData.shape
trainLabel = label[0 : nTrainData]
testOutputStringLabel = outputStringLabel[nTrainData + 1 :]


model = Sequential()

model.add(Dense(300, input_shape=(300,)))
model.add(Activation('tanh'))
model.add(BatchNormalization())

model.add(Dense(300))
model.add(Activation('tanh'))
model.add(BatchNormalization())

model.add(Dense(300))
model.add(Activation('tanh'))

optimizer = SGD(lr=0.01)
model.compile(loss='mse', optimizer=optimizer)

for epoch in range(nEpoch):
    model.fit(trainData, trainLabel, nb_epoch=1, batch_size=128, validation_split=0.1)
    
    correctCnt = 0
    randIdx = numpy.random.randint(0, len(testData), 10)
    for testIdx in range(len(testData)):
	modelOutput = model.predict(numpy.expand_dims(testData[testIdx, :], axis=0))
	topNht, isCorrect, topNdist = predContext(htDic, modelOutput, topN, testOutputStringLabel[testIdx])
        if testIdx in randIdx:
            print "Generating for example ", testIdx
            print "True input is ", testInputStringLabel[testIdx]
	    print "True label is ", testOutputStringLabel[testIdx]
            print "Top ", topN, " hashtags are ", topNht
	if isCorrect:
            correctCnt += 1.0
    print correctCnt
    accuracy = correctCnt*1.0 / len(testData)
    print "Test accuracy is ", accuracy
        
