import cPickle as pickle
import numpy as np
#reduces all tweets to feature only characters in
#the second and fourth columns of the standard ascii table.

#This should be used if you are using an old version
#of the storm topology that does not do this online.

tweets = pickle.load(open("new_tweets_list_string.pkl","rb"))
embeddings = pickle.load(open("new_embeddings.pkl","rb"))

print "tweet array shape: ", len(tweets)
print "embeddings array shape: ", embeddings.shape
print "tweet array type: ", type(tweets[0])
print "embeddings array type: ", type(embeddings[0])


for i in range(len(tweets)):
    s=""
    for j in range(len(tweets[i])):
        asciiVal = ord(tweets[i][j])
        
        if(asciiVal>=32 and asciiVal<=63):
            s+=tweets[i][j]
        elif(asciiVal>=96 and asciiVal <= 127):
            s+=tweets[i][j]
        else:
            continue
    tweets[i]=s

pickle.dump(tweets, open( "preprocessed_new_tweets","wb"))

