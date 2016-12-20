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

#load tweets and hashtag embeddings
tweets = pickle.load(open("preprocessed_new_tweets.pkl","rb"))
embeddings = pickle.load(open("new_embeddings.pkl","rb"))

#visualize data
print "tweets (ELEMENT TYPE): ", type(tweets[0])
print "tweets (Number Of Tweets): ", len(tweets)

print "hashtag (ELEMENT TYPE): ", type(embeddings[0])
print "hashtag (SHAPE): ", embeddings.shape

#create character dictionary for tweets.
dictionary = ReducedAsciiDictionary({},np.array([])).dictionary


#load first 1000 tweets
#recall numpy array should be of shape (samples, time steps, features)
numData = 1000
timeSteps = 10
X = np.zeros([numData, 140+1, len(dictionary) + embeddings.shape[1]])

# for each tweet
for twt in range(numData):
	currTweet = tweets[twt][6:len(tweets[twt])]
	for ch in range(len(currTweet)):
	    oneHotIndex = dictionary.get(currTweet[ch])
	    X[twt,ch,oneHotIndex] = 1
		
	    for embIndex in range(embeddings.shape[1]):
			X[twt,ch,embIndex+len(dictionary)] = embeddings[twt,embIndex]
	#end of tweet indentifier is when first and second are ones
	X[twt,len(currTweet),0]=-1
 
print "data (TYPE): ", type(X)
print "data (SHAPE): ", X.shape

print "first tweet, first character: ", tweets[0][6]
print "first tweet, first character index: ", dictionary.get(tweets[0][6])
print "first tweet, first character vector: ", X[0,0]

print "random tweet, random character: ", tweets[253][12+6]
print "random tweet, random cahracter index: ", dictionary.get(tweets[253][12+6])
print "random tweet, random character vector:", X[253,12]
