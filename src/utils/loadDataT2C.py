'''
loads twitter dataset from storm API.
'''
import sys
import os
import numpy as np
import cPickle as pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","utils")))
from ReducedAsciiDictionary import ReducedAsciiDictionary
from getEnglishHashTweets import checkHashtags
from numpy import random
from random import shuffle
from os.path import expanduser

def normalizeByFreq(tweets, hashtags, hashtagFreq, freqThreshold):
    tweets_shuf = []
    hashtags_shuf = []
    hashtagFreqCnt = {}
    idx_shuf = range(len(tweets))
    #shuffle(idx_shuf)
    
    for i in idx_shuf:
        ht = hashtags[i].split(" ")
        if hashtagFreq[ht[0]] >= freqThreshold:
            if hashtagFreqCnt.get(ht[0]) == None:
                hashtagFreqCnt[ht[0]] = 1
                tweets_shuf.append(tweets[i])
                hashtags_shuf.append(ht[0])
            elif hashtagFreqCnt[ht[0]] < freqThreshold:
                hashtagFreqCnt[ht[0]] += 1
                tweets_shuf.append(tweets[i])
                hashtags_shuf.append(ht[0])
    return tweets_shuf, hashtags_shuf

def loadData(dictionary,ranges,sequenceLength,trainPercent, freqThreshold):
	''' Creates dataset based on dictionary, a set of ascii
	ranges, and pickled twitter data from Apache Storm.


        X: [#sequences, 40, 65]
        y: [#sequences, 300]
	vocabLen: (dictionary length)
	tweetLength: (numTweets)
	'''

        #load tweets with >=2 hashtags and corresponding english hashtags
        tweets = pickle.load(open(expanduser("~/tweetnet/data/multitaskTweets.pkl"), "rb"))
        hashtags = pickle.load(open(expanduser("~/tweetnet/data/multitaskHashtags.pkl"), "rb"))
        
        #load hashtag frequency dictionary
        hashtagFreq = pickle.load(open(expanduser("~/tweetnet/data/hashtagFreq.pkl"), "rb"))
        
        #modifiedTweets = []
        #for i in range(len(tweets)):
	#    # Get rid of the "text: " and add start of text and end of text
        #    modifiedTweets.append(chr(2) + tweets[i][6:] + chr(3))
        #tweets = modifiedTweets

        #Normalize data by frequency
        tweets_shuf, hashtags_shuf = normalizeByFreq(tweets, hashtags, hashtagFreq, freqThreshold)
 
        nTweet = len(tweets_shuf)
        nTrainData = np.ceil(nTweet*trainPercent).astype(int)
        
        #Split the tweets and hashtags into training and testing set 
        trainTweets = tweets_shuf[0: nTrainData]
        trainHashtags = hashtags_shuf[0: nTrainData]
        testTweets = tweets_shuf[nTrainData: nTweet]
        testHashtags = hashtags_shuf[nTrainData: nTweet]
        nTestData = len(testTweets)

        
        #load word2vec dictionary
        print("Loading word2vec dictionary")
        word2vecDict = pickle.load(open(expanduser("~/tweetnet/data/word2vec_dict.pkl"), "rb"))
        print("Finished loading word2vec dictionary")
        
	#create character dictionary for tweets.
	dictionary = ReducedAsciiDictionary({},ranges).dictionary
        dictionary[chr(2)] = len(dictionary)
        dictionary[chr(3)] = len(dictionary)
	#number of unique characters in dataset
	vocabLen = len(dictionary)

	#initialize datastore arrays
        trainTweetSequence = []
        trainHashtagSequence = []
        testTweetSequence = []
        testHashtagSequence = []

        #vector in word2vec is 300
        embeddingLength = 300

        #Split data into sequences of length 40 for training
        for i in range(nTrainData):
            oneTweet = trainTweets[i]
            for j in range(0, len(oneTweet) - sequenceLength + 1, 1):
                trainTweetSequence.append(oneTweet[j : j+sequenceLength])
                trainHashtagSequence.append(trainHashtags[i])
        print('Number of sequences in training data: ', len(trainTweetSequence))
        print('Number of hashtags in training data: ', len(trainHashtagSequence))


        #Split data into sequences of length 40 for testing
        for i in range(nTestData):
            oneTweet = testTweets[i]
            ht = hashtags[i].split(" ")
            for j in range(0, len(oneTweet) - sequenceLength + 1, 1):
                testTweetSequence.append(oneTweet[j : j+sequenceLength])
                testHashtagSequence.append(testHashtags[i])
        print('Number of sequences in testing data: ', len(testTweetSequence))
        print('Number of hashtags in testing data: ', len(testHashtagSequence))
	
	
        # for each sequence, create onehot encoding for each character
	print("Vectorization...")
        
        # trainX: [#training sequences, 40, 65]
        # trainy: [#training sequences, 300]
        trainX = np.zeros((len(trainTweetSequence), sequenceLength, vocabLen), dtype=np.bool)
        trainY = np.zeros((len(trainTweetSequence), embeddingLength))
	for i, seq in enumerate(trainTweetSequence):
            if i % 10000 == 0:
                print("Loading training tweet ", i)
	    for j, ch in enumerate(seq):
		oneHotIndex = dictionary.get(ch)
		trainX[i,j,oneHotIndex] = 1
                trainY[i] = word2vecDict[trainHashtagSequence[i]]	    
        
        # testX: [#testing sequences, 40, 65]
        # testy: [#testing sequences, 300]
        testX = np.zeros((len(testTweetSequence), sequenceLength, vocabLen), dtype=np.bool)
        testY = np.zeros((len(testTweetSequence), embeddingLength))
        
	for i, seq in enumerate(testTweetSequence):
            if i % 10000 == 0:
                print("Loading testing tweet ", i)
	    for j, ch in enumerate(seq):
		oneHotIndex = dictionary.get(ch)
		testX[i,j,oneHotIndex] = 1
                testY[i] = word2vecDict[testHashtagSequence[i]]	  

        tweet2hashtagParam = [trainTweets, trainHashtags, testTweets, testHashtags, trainX, trainY, testX, testY, trainTweetSequence, trainHashtagSequence, testTweetSequence, testHashtagSequence]
          

	return trainTweets, trainHashtags, testTweets, testHashtags, trainX, trainY, testX, testY, trainTweetSequence, trainHashtagSequence, testTweetSequence, testHashtagSequence, word2vecDict


if __name__ == "__main__":
    trainTweets, trainHashtags, testTweets, testHashtags, trainX, trainY, testX, testY, trainTweetSequence, trainHashtagSequence, testTweetSequence, testHashtagSequence, dictionary = loadData({},np.array([]), 40, 0.99, 84)
    dic = {}
    for i in range(1000):
        if dic.get(trainHashtagSequence[i]) == None:
            dic[trainHashtagSequence[i]] = 1
    print dic.keys()
    
