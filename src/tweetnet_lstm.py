'''
Keras backend. LSTM Model.

Using standard default range:
Input: (65x1) 64 unique chars, 1 EOS char
Output: (65x1) 64 unique chars, 1 EOS char

'''
import pickle as pkl
import  numpy as np
import h5py
from loadData_lstm2 import loadData
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Activation
from keras.optimizers import RMSprop
from keras.optimizers import Adagrad
from keras.layers import Dropout
from keras.layers import BatchNormalization
from tweetGenerator_lstm import generateText
from keras.callbacks import ModelCheckpoint
from os.path import expanduser


#sequenceLength: sequence length (k in BPTTk)
sequenceLength = 40
print("Start loading data ...")
X, y, vocabLen, dictionary, tweetSequence, nextChar, tweets = loadData({},np.array([]), sequenceLength) 
print("Finished loading data")

loadWeights=False

#initialize some hyper-parameters
#inputSize: size of each input vector (default: 365x1)
inputSize = vocabLen
print vocabLen
#numHiddenFirst: size of first hidden layer
numHiddenFirst = 128
#seqPerSegment: sequences (of size sequenceLength) per mini-epoch. 
#Lowers maximum memory usage.
seqPerSegment = 10000

#X: [10000 (numTweets), 40 (sequenceLength), 65(inputSize)].
n_examples = len(X)
numSegments = np.ceil(n_examples/seqPerSegment).astype(int)
numEpochs=50
print('# of sequences per segments: ', seqPerSegment)
print('# of segments: ', numSegments)

#building cLSTM model
#print("\n")
print("Start building model ....")
model = Sequential()

model.add(LSTM(numHiddenFirst, input_shape=(sequenceLength, inputSize)))
#model.add(LSTM(numHiddenFirst))

#model.add(Dense(numHiddenFirst))
#model.add(Activation('relu'))
#model.add(BatchNormalization())

#model.add(Dense(numHiddenFirst))
#model.add(Activation('relu'))
#model.add(BatchNormalization())

model.add(Dense(vocabLen))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)

if(loadWeights==True):
    model.load_weights(expanduser("~/tweetnet/logs/intermediateWeights.hdf5"))


model.compile(loss='categorical_crossentropy', optimizer=optimizer)
print("Finished building model.")
#define file checkpoint
#filePath = expanduser("~/tweetnet/logs/intermediateWeights.hdf5")
#checkPoint = ModelCheckpoint(filePath, monitor='loss', verbose=1)
#callbacksList = [checkPoint]

#train on mini-epochs (sized seqPerSegment) to lower total RAM usage.
for epoch in range(numEpochs):
#    model.fit(X, y, nb_epoch=1, batch_size=128)
#    generateText(model, tweets, sequenceLength, vocabLen, dictionary)
    for seg in range(numSegments):
        print("\n")
        print "Segment: ", seg+1, "/", numSegments, " | Epoch: ", epoch, "/", numEpochs 
        model.fit(X[seg*seqPerSegment: (seg+1)*seqPerSegment], y[seg*seqPerSegment: (seg+1)*seqPerSegment], nb_epoch=1, batch_size=128)
        generateText(model, tweets, sequenceLength, vocabLen, dictionary)

        




