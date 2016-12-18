import numpy as np
import cPickle as pickle

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
print "tweets (ELEMENT TYPE):", type(tweets[0])
print "tweets (SHAPE): ", len(tweets)

print "hashtag (ELEMENT TYPE):", type(embeddings[0])
print "hashtag (SHAPE): ", embeddings.shape

'''
tweets (ELEMENT TYPE): <type 'str'>
tweets (SHAPE):  34714
hashtag (ELEMENT TYPE): <type 'numpy.ndarray'>
hashtag (SHAPE):  (34714, 300)
'''

#create character dictionary for tweets.
