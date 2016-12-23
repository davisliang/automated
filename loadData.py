import numpy as np
import cPickle as pickle
from ReducedAsciiDictionary import ReducedAsciiDictionary
'''
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
'''
def loadData(dictionary,ranges):
	#load tweets and hashtag embeddings
	tweets = pickle.load(open("preprocessed_new_tweets.pkl","rb"))
	embeddings = pickle.load(open("new_embeddings.pkl","rb"))

	#visualize data
	print "tweets (ELEMENT TYPE): ", type(tweets[0])
	print "tweets (Number Of Tweets): ", len(tweets)

	print "hashtag (ELEMENT TYPE): ", type(embeddings[0])
	print "hashtag (SHAPE): ", embeddings.shape

	#create character dictionary for tweets.
	dictionary = ReducedAsciiDictionary({},ranges).dictionary
	numData = len(tweets)

	#load first 1000 tweets
	#recall numpy array should be of shape (samples, time steps, features)

	vocabLen = len(dictionary)+1
	X = np.zeros([numData, 140+1, vocabLen + embeddings.shape[1]])
	tweetLength = np.zeros(numData)

	# for each tweet
	for twt in range(numData):
		tweetLength[twt] = len(tweets[twt])-6+1
		currTweet = tweets[twt][6:len(tweets[twt])]
		for ch in range(len(currTweet)):
		    oneHotIndex = dictionary.get(currTweet[ch])
		    X[twt,ch,oneHotIndex] = 1
			
		    for embIndex in range(embeddings.shape[1]):
				X[twt,ch,embIndex+vocabLen] = embeddings[twt,embIndex]
		#end of tweet indentifier is when first and second are ones
		X[twt,len(currTweet),len(dictionary)]=1

	return X, vocabLen, tweetLength