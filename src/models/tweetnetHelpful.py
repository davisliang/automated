import sys
import os
from os.path import expanduser
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","utils")))
from loadKaggleHelpful import loadTrain, loadTest
from ReducedAsciiDictionary import ReducedAsciiDictionary
from numpy import random
from random import shuffle
import cPickle as pickle
import numpy

def heuristic(text, helpful, outOf):
    heurText = []
    heurHelpful = []
    heurOutOf = []
    for i in range(len(outOf)):
        if outOf[i] == 0:
            continue
        heurText.append(text[i])
        heurHelpful.append(helpful[i])
        heurOutOf.append(outOf[i])
    return heurText, heurHelpful, heurOutOf

def clean(text, dictionary, charLimit):
    cleanText = []
    for textBlock in text:
        if(len(textBlock) > charLimit):
            textBlock = textBlock[:charLimit]
        cleanBlock = []
        for character in textBlock:
            if(character in dictionary):
                cleanBlock.append(character)
            elif(character >= 'A' and character <= 'Z'):
                character = chr(ord(character)-ord('A')+1+64)
                cleanBlock.append(character)
            else:
                continue
        cleanText.append(cleanBlock)
    return cleanText

#print("Loading word2vec dictionary")
#word2vecDict = pickle.load(open(expanduser("~/tweetnet/data/word2vec_dict.pkl"),"rb"))
#print("Finished loading word2vec dictionary")

#load reduced ascii dictionary
print("Loading reduced ascii dictionary")
dictionary = ReducedAsciiDictionary({},numpy.array([])).dictionary

#get review data and metadata
print("Loading Training Data")
trainText, trainHelpful, trainOutOf, trainUserID, trainItemID = loadTrain()
trainText, trainHelpful, trainOutOf = heuristic(trainText, trainHelpful, trainOutOf)
print("Loading Testing Data")
testText, testOutOf, testUserID, testItemID = loadTest()



#clean text data
print("Cleaning text")
trainText = clean(trainText, dictionary,300)
testText = clean(testText, dictionary,300)


#set up train sequence and labels
trainInput = []
#trainInputContext = []
trainLabel = []
sequenceLength = 10

for i in range(len(trainText)):
    textBlock = trainText[i]
    helpfulBlock = trainHelpful[i]
    outOfBlock = trainOutOf[i]
    if(outOfBlock == 0):
        outOfBlock = 1
    helpfulnessRate = helpfulBlock*1.0/outOfBlock

    #reviewerID = trainUserID[i]
    for c in range(0, len(textBlock) - sequenceLength):
        trainInput.append(textBlock[c:c+sequenceLength])
        trainLabel.append(helpfulnessRate)
        #trainInputContext.append(reviewerID)
print('Number of sequences in training set: ', len(trainInput))

trainX = numpy.zeros((len(trainInput), sequenceLength, len(dictionary)), dtype=numpy.float64)
trainY = numpy.zeros(len(trainInput))
numExamples = 1000000
for i, seq in enumerate(trainInput):
    if i % 1000 == 0:
        if i > numExamples:
            break
        print("loading review ", i)
    for j, ch in enumerate(seq):
        oneHotIndex = dictionary.get(ch)
        trainX[i,j,oneHotIndex] = 1
    trainY[i] = trainLabel[i]

pickle.dump(trainX, open(expanduser("~/tweetnet/data/helpX"), "wb"))
pickle.dump(trainY, open(expanduser("~/tweetnet/data/helpY"),"wb"))



