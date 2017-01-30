from os.path import expanduser
import numpy
from numpy import shape
import cPickle as pickle

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.optimizers import SGD
from keras.layers import BatchNormalization

hashtags = pickle.load(open(expanduser("~/tweetnet/data/englishHashtag.pkl"),"rb"))
dictionary = pickle.load(open(expanduser("~/tweetnet/data/word2vec_dict.pkl"), "rb"))

train = numpy.zeros([len(hashtags),300])
test = numpy.zeros([len(hashtags),300])

for i in range(len(hashtags)):
    line = hashtags[i]
    listHashtag = line.split()
    train[i,:]=dictionary[listHashtag[1]]
    test[i,:]=dictionary[listHashtag[2]]

model = Sequential()

model.add(Dense(300, input_shape=(300,)))
model.add(Activation('tanh'))

model.add(Dense(300))
model.add(Activation('tanh'))

model.add(Dense(300))
model.add(Activation('softmax'))

optimizer = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='cosine_proximity', optimizer=optimizer)
model.fit(train, test, nb_epoch=50, batch_size=1, validation_split=0.1)



