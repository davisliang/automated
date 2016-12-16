import cPickle as pickle
import numpy as np

tweets = pickle.load(open("preprocessed_new_tweets.pkl","rb"))
embeddings = pickle.load(open("new_embeddings.pkl","rb"))

print "tweet array shape: ", len(tweets)
print "embeddings array shape: ", embeddings.shape
print "tweet array type: ", type(tweets[0])
print "embeddings array type: ", type(embeddings[0])

for i in range(100):
	print tweets[i]
