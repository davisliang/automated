'''
Keras backend. LSTM Model.

Using standard default range:
Input: (65x1) 64 unique chars, 1 EOS char
Output: (65x1) 64 unique chars, 1 EOS char

'''
import cPickle as pickle
import  numpy as np
import h5py
import os
import sys
from os.path import expanduser
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","utils")))
from loadDataText2Hashtag import loadData
from predContext import predContext, createHtDict
from keras import backend as K
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import PReLU
from keras.layers import Activation
from keras.layers import TimeDistributed
from keras.optimizers import RMSprop
from keras.optimizers import Adagrad
from keras.layers import Dropout
from keras.layers import BatchNormalization
from tweetGenerator_lstm import generateText
from keras.callbacks import ModelCheckpoint
from logger import logger
import time

#sequenceLength: sequence length (k in BPTTk)
sequenceLength = 30
#Number of symbols
vocabLen = 66
#train test split
trainPercent = 0.95
#threshold on hashtag frequency 
freqThreshold = 84

logAllPredictions=True
#X: [# Seuqences, 40 (sequenceLength), 65(inputSize)].
#y: [# Sequences, 300]

print("Start loading data ...")
trainTweets, trainHashtags, testTweets, testHashtags, trainX, trainY, testX, testY, trainTweetSequence, trainHashtagSequence, testTweetSequence, testHashtagSequence, dictionary, trainStartIdx, testStartIdx = loadData({},np.array([]), sequenceLength, trainPercent, freqThreshold)
print("Finished loading data")


#initialize some hyper-parameters
#topN = np.ceil(0.05*nUniqueHt).astype(int)

#embeddingLength: size of the word embedding
embeddingLength = 300

#inputSize: size of each input vector (default: 365x1)
inputSize = vocabLen + embeddingLength

#numHiddenFirst: size of first hidden layer
numHiddenFirst = 512

#Number of testing/training tweets
nTestData = len(testTweets)
nTrainData = len(trainTweets)
nTestSequences = len(testTweetSequence)
nTrainSequences = len(trainTweetSequence)
print "Number of testing sequences: ", nTestSequences
print "Number of training sequences: ", nTrainSequences
print "Number of testing tweets: ", nTestData
print "Number of training tweets: ", nTrainData

dictionary = pickle.load(open(expanduser("~/tweetnet/data/word2vec_dict.pkl")))

# Create the hashtag dictionary
htDic = createHtDict(dictionary, testHashtags)

numEpochs=50

#building cLSTM model
print("Start building model ....")

model = Sequential()

model.add(TimeDistributed(Dense(numHiddenFirst), input_shape=(sequenceLength, inputSize)))
model.add(LSTM(numHiddenFirst, return_sequences=True))
model.add(BatchNormalization())
model.add(LSTM(numHiddenFirst, return_sequences=True))
model.add(BatchNormalization())
model.add(LSTM(numHiddenFirst))
model.add(BatchNormalization())
model.add(Dense(numHiddenFirst))
model.add(PReLU())
model.add(BatchNormalization())

model.add(Dense(embeddingLength))
model.add(PReLU())
model.add(BatchNormalization())

optimizer = RMSprop(lr=0.005)

getTopOutput = K.function([model.layers[0].input, K.learning_phase()], [model.layers[-1].output])

model.compile(loss='mean_squared_error', optimizer=optimizer)
print("Finished building model.")

name = "t2c"+time.strftime("%Y-%m-%d_%H:%M") + ".log"
for epoch in range(numEpochs):

    prevOutput = np.zeros((1, 1, embeddingLength))
    accumError = []
    
    for idx in range(len(trainX)):
	if idx % 500 == 0:
            if idx != 0:
                print "average error from ", idx-500, "to", idx, "is: ", np.mean(np.array(accumError)) 
	    accumError = []
        oneSequence = trainX[idx, :, :]
        
        # Shape of oneSequence: [1, 40, 66]
	oneSequence = np.expand_dims(oneSequence, axis=0)

	if idx in trainStartIdx:
            # If the sequence is the first sequence in the tweet, set prevOutput to random / 0
            prevOutput = (np.random.rand(1, sequenceLength, embeddingLength) - 1) * 0.1

            # Concatenate random context vector to the end of sequence vector
	    oneSequence = np.concatenate((oneSequence, prevOutput), axis=2)
	    
            hist = model.fit(oneSequence, np.reshape(trainY[idx, :], (1, embeddingLength)), nb_epoch=1, batch_size=1)
            accumError.append(hist.history.values()[0][0])
            prevOutput = getTopOutput([oneSequence, 1])[0]
           
            # Reshape and repeat the context vector to [1, 40, 300]
            prevOutput = np.expand_dims(prevOutput, 0)
            prevOutput = np.repeat(prevOutput, sequenceLength, axis=1)
	else:    

            # Concatenate two vectors to => [1, 40, 366]
	    oneSequence = np.concatenate((oneSequence, prevOutput), axis=2)
            
            hist = model.fit(oneSequence, np.reshape(trainY[idx, :], (1, embeddingLength)), nb_epoch=1, batch_size=1)
            accumError.append(hist.history.values()[0][0])
            prevOutput = getTopOutput([oneSequence, 1])[0]

            # Reshape and repeat the context vector to [1, 40, 300]
            prevOutput = np.expand_dims(prevOutput, 0)
            prevOutput = np.repeat(prevOutput, sequenceLength, axis=1)
        #print "First 10 for text: ", oneSequence[0, 0, 0:10]
        #print "First 10 for text2: ", oneSequence[0, 1, 0:10]
        #print "Last 10 for text: ", oneSequence[0, 0, 56:66]
        #print "Last 10 for text2: ", oneSequence[0, 1, 56:66]
        #print "Last 10 for context: ", oneSequence[0, 0, 356: 366]
        #print "Last 10 for context2: ", oneSequence[0, 1, 356: 366]
     
    #correctCnt = 0
    #randIdx = np.random.randint(0, nTestData, 10)
 
    #tweetCnt = 0
    #tweetStartIdx = 0
    #log = []
    #log.append([epoch])
    #for testIdx in range(nTestSequences):
    #    # Stack the windows (1 x 40 x 65) of each tweet as a 3D matrix (#windows x 40 x 65)
    #    if testTweetSequence[testIdx][-1] == chr(3):
    #        oneTweet = testX[tweetStartIdx:testIdx+1, :, :]
    #        modelOutput = model.predict(oneTweet)
    #        topNht, isCorrect, topNdist = predContext(htDic, modelOutput, topN, testHashtags[tweetCnt])
    #        tweetStartIdx = testIdx + 1
    #        if isCorrect:
    #            correctCnt += 1
    #            isCorrect = True
    #        if tweetCnt in randIdx:
    #            print testTweets[tweetCnt][:-2]
    #            print "Given label is ", testContextSequence[testIdx]
    #            print "True label is ", testHashtags[tweetCnt]
    #            print "Top ", topN, " hashtags are ", topNht
            
    #        if logAllPredictions:
    #            log.append([testTweets[tweetCnt][:-2],testHashtags[tweetCnt],isCorrect,topNht])
    #        tweetCnt += 1
    #accuracy = correctCnt*1.0 / nTestData
    #log.append([correctCnt, accuracy])        
    #logger(log,name)




