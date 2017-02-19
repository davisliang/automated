import numpy as np
import cPickle as pickle
from os.path import expanduser

hashtag = pickle.load(open(expanduser("~/tweetnet/data/englishHashtag.pkl")))
dic = {}
for ht in hashtag:
    ht = ht.split(" ")
    if dic.get(ht[2]) == None:
        dic[ht[2]] = 1
    else:
        dic[ht[2]] += 1

pickle.dump(dic, open(expanduser("~/tweetnet/data/hashtagFreq.pkl"), "w"))
for threshold in range(1,1000):
    cnt = 0
    for k in dic.keys():
        if dic[k] >= threshold:
	    cnt += 1
    print "Threshold = ", threshold, "    # Hashtags= ", cnt*threshold

ht = dic.keys()
freq = dic.values()
idx = np.argsort(np.array(freq))

sortedHt = []
sortedFq = []
for i in idx:
    sortedHt.append(ht[i])
    sortedFq.append(freq[i])
