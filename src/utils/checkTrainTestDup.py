import cPickle as pickle
import numpy as np
import os
from os.path import expanduser

test_data = pickle.load(open(expanduser("~/tweetnet/data/test_data.pkl")))
train_data = pickle.load(open(expanduser("~/tweetnet/data/train_data.pkl")))

testX = test_data[0]
trainX = train_data[0]

idx = len(test_data)*np.random.rand(2000)

n = 0
cnt = 0
for i in idx:
    print n
    n += 1
    test_x = testX[int(i), :, :]
    for j in range(len(trainX)):
        if np.array_equal(test_x, trainX[j, :, :]):
            cnt += 1
            print "Dup"
            break
print cnt    
