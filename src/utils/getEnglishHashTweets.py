from os.path import expanduser
import numpy
import cPickle as pickle

def checkHashtags(hashtagStr,dictionary):
    hasMultiEnglishHashtag = False 
    returnHt = "hashtags:"
    htStr = hashtagStr[9:]
    htTokens = htStr.split(" ")
    nEnglishHashtag = 0
    for token in htTokens:
        try:
            if(len(dictionary[token])>0):
                returnHt = returnHt + " " + token
		nEnglishHashtag += 1
        except KeyError:
            continue
    if nEnglishHashtag >= 2:
	hasMultiEnglishHashtag = True
    return returnHt, hasMultiEnglishHashtag


dictionary = pickle.load(open(expanduser("~/tweetnet/data/word2vec_dict.pkl"), "r"))

textFile = open(expanduser("~/tweetnet/data/dump.txt"), "r")
fileLines = textFile.readlines()
keepTweets = []
keepHashtags = []
counter = 0

while((counter+1)<len(fileLines)):
    if(counter%1000==0):
        print "line %i of %i" %(counter, len(fileLines)) 
    textLine = fileLines[counter]
    htLine = fileLines[counter+1]
    if(textLine[0:5]=="text:"):
        wordHT, hasMultiEnglishHashtag = checkHashtags(htLine,dictionary)
        if(hasMultiEnglishHashtag):
            keepTweets.append(textLine)
            keepHashtags.append(wordHT)
    counter = counter + 1

print "Number of remaining tweets: ",len(keepTweets)     
with open(expanduser("~/tweetnet/data/englishHashtagTweet.pkl"), "wb") as file1:
    pickle.dump(keepTweets,file1, pickle.HIGHEST_PROTOCOL) 

with open(expanduser("~/tweetnet/data/englishHashtag.pkl"), "wb") as file2:
    pickle.dump(keepHashtags,file2,pickle.HIGHEST_PROTOCOL) 

