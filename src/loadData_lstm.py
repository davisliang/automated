'''
loads twitter dataset from storm API.
'''
import numpy as np
import cPickle as pickle
from ReducedAsciiDictionary import ReducedAsciiDictionary
from os.path import expanduser

def loadData(dictionary,ranges):
	''' Creates dataset based on dictionary, a set of ascii
	ranges, and pickled twitter data from Apache Storm.

	X: (numTweets, 141, dictionaryLength + embeddings length)
	vocabLen: (dictionary length)
	tweetLength: (numTweets)
	'''

	#load tweets and hashtag embeddings
	tweets = pickle.load(open(expanduser("~/tweetnet/data/preprocessed_new_tweets.pkl"),"rb"))

	#visualize data
	#print "tweets (ELEMENT TYPE): ", type(tweets[0])
	#print "tweets (Number Of Tweets): ", len(tweets)
	#print "hashtag (ELEMENT TYPE): ", type(embeddings[0])
	#print "hashtag (SHAPE): ", embeddings.shape

	#create character dictionary for tweets.
	dictionary = ReducedAsciiDictionary({},ranges).dictionary

	#total number of tweets
	numData = len(tweets)

	#number of unique characters in dataset
	vocabLen = len(dictionary)+1

	#initialize datastore arrays
	X = np.zeros([numData, 140+1, vocabLen])
	tweetLength = np.zeros(numData)

	# for each tweet create onehot encoding for each character
	for twt in range(numData):
		if(twt%1000==0):
			print "loaded: ", twt, " of ", numData
		tweetLength[twt] = len(tweets[twt])-6+1
		currTweet = tweets[twt][6:len(tweets[twt])]

		for ch in range(len(currTweet)):
			oneHotIndex = dictionary.get(currTweet[ch])
			X[twt,ch,oneHotIndex] = 1
			
		#end of tweet character (EOS)
		X[twt,len(currTweet),len(dictionary)]=1

	return X, vocabLen, tweetLength, dictionary
