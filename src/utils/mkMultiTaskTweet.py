'''
loads twitter dataset from storm API for multitasking model training
Task 1: Hashtag prediction
Task 2: missing word completion

Data format: tweet -- hashtag -- missing word
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
import re
import string
from stop_words import get_stop_words

def mkMissingWord(text, word2vecDict):
    
    punctuation = set(string.punctuation)
    stop_words = get_stop_words('english')
    words = text.split()
    cnt = 0
    while cnt <= 7:
        idx = 1 + random.randint(len(words) - 1)
        w = words[idx]
        w = ''.join([c for c in w.lower() if not c in punctuation])
        if len(w)==1 or word2vecDict.get(w) == None or w in stop_words:
            cnt += 1
        else:
            missingWord = w
            words[idx] = "UNK"
            text = " ".join(words)
            return (text, missingWord)
    return (None,None)
        
    

def tweetsForMultiTask(tweets, hashtags, word2vecDict):


    tweets_shuf = []
    hashtags_shuf = []
    missingWords = []
    idx_shuf = range(len(tweets))
    shuffle(idx_shuf)
    for i in idx_shuf:
        ht = hashtags[i].split(" ")

        text, missingWord = mkMissingWord(tweets[i], word2vecDict)
        if text != None and missingWord != None:
            tweets_shuf.append(text)
            hashtags_shuf.append(ht[2])
            missingWords.append(missingWord)
            print (text, ht[2], missingWord)

    return tweets_shuf, hashtags_shuf, missingWords


def loadData(dictionary,ranges,sequenceLength,trainPercent):
	''' Creates dataset based on dictionary, a set of ascii
	ranges, and pickled twitter data from Apache Storm.


        X: [#sequences, 40, 65]
        y: [#sequences, 300]
	vocabLen: (dictionary length)
	tweetLength: (numTweets)
	'''


        #load tweets with >=2 hashtags and corresponding english hashtags
        tweets = pickle.load(open(expanduser("~/tweetnet/data/englishHashtagTweet.pkl"), "rb"))
        hashtags = pickle.load(open(expanduser("~/tweetnet/data/englishHashtag.pkl"), "rb"))
        modifiedTweets = []

        #load word2vec dictionary
        print("Loading word2vec dictionary")
        word2vecDict = pickle.load(open(expanduser("~/tweetnet/data/word2vec_dict.pkl"), "rb"))
        print("Finished loading word2vec dictionary")
        
        for i in range(len(tweets)):
	    # Get rid of the "text: " and add start of text and end of text
            modifiedTweets.append(chr(2) + tweets[i][6:] + chr(3))
        
        tweets = modifiedTweets
        tweets, hashtags, missingWords = tweetsForMultiTask(tweets, hashtags, word2vecDict)

	nTweet = len(tweets)
        
        print "Number of remaining tweets: ", nTweet
        
        print "Saving data to files ..."
        with open(expanduser("~/tweetnet/data/multitaskTweets.pkl"), "wb") as file1:
            pickle.dump(tweets, file1, pickle.HIGHEST_PROTOCOL)
        with open(expanduser("~/tweetnet/data/multitaskHashtags.pkl"), "wb") as file2:
            pickle.dump(hashtags, file2, pickle.HIGHEST_PROTOCOL)
        with open(expanduser("~/tweetnet/data/multitaskTweetMw.pkl"), "wb") as file3:
            pickle.dump(missingWords, file3, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    loadData({},np.array([]), 40, 0.9)
