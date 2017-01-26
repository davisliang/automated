'''
loads twitter dataset from storm API.
'''
import numpy as np
import cPickle as pickle
from ReducedAsciiDictionary import ReducedAsciiDictionary
from os.path import expanduser
import sys

def loadData(dictionary,ranges,sequenceLength):
	''' Creates dataset based on dictionary, a set of ascii
	ranges, and pickled twitter data from Apache Storm.

	X: (numTweets, 141, dictionaryLength + embeddings length)
	vocabLen: (dictionary length)
	tweetLength: (numTweets)
	'''

	#load tweets and hashtag embeddings
	tweets = pickle.load(open(expanduser("~/tweetnet/data/preprocessed_new_tweets.pkl"),"rb"))
        np.random.shuffle(tweets)
        tweets = tweets[0:7000]
        print "Number of tweets ", len(tweets)
	#create character dictionary for tweets.
	dictionary = ReducedAsciiDictionary({},ranges).dictionary

	#total number of tweets
	numData = len(tweets)

	#number of unique characters in dataset
	vocabLen = len(dictionary)+1

	#initialize datastore arrays
        tweetSequence = []
        nextChar = []
	tweetLength = np.zeros(numData)

        #Split data into sequences of length 40 and create nextChar array
        for i in range(numData):
            oneTweet = tweets[i]
            for j in range(0, len(oneTweet) - sequenceLength - 1, 1):
                tweetSequence.append(oneTweet[j : j+sequenceLength])
                nextChar.append(oneTweet[j+sequenceLength])
            tweetSequence.append(oneTweet[len(oneTweet)-sequenceLength - 1:len(oneTweet) - 1])
            nextChar.append("EOS")
        print('Number of sequences: ', len(tweetSequence))

	# for each sequence, create onehot encoding for each character
	print("Vectorization...")
        X = np.zeros((len(tweetSequence), sequenceLength, vocabLen), dtype=np.bool)
        y = np.zeros((len(tweetSequence), vocabLen), dtype=np.bool)

        for i, seq in enumerate(tweetSequence):
            if i % 10000 == 0:
                print "Loading tweet ", i
	    for j, ch in enumerate(seq):
		oneHotIndex = dictionary.get(ch)
		X[i,j,oneHotIndex] = 1
	    
            if len(nextChar[i]) == 1:
                y[i, dictionary.get(nextChar[i])] = 1
            else:
                y[i, len(dictionary)] = 1
        return X, y, vocabLen, dictionary, tweetSequence, nextChar, tweets

if __name__ == "__main__":
    a,b,c,d,e,f = loadData({},np.array([]),40)
    print e[0]
