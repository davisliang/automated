'''
Keras backend. cLSTM Model.

Using standard default range:
Input: (365x1) 64 unique chars, 1 EOS char, 300 word2vec context
Output: (65x1) 64 unique chars, 1 EOS char

'''
import pickle as pkl
import  numpy as np
from loadData import loadData
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Activation
from keras.optimizers import RMSprop
from keras.layers import Dropout
from keras.layers import BatchNormalization

print("Start loading data ...")
data, dictLen, tweetLen = loadData({},np.array([])) 
# data shape = #tweets x 141 x inputSize(365)
print("Finished loading data")

#initialize some hyper-parameters
#inputSize: size of each input vector (default: 365x1)
inputSize = data.shape[2]
#sequenceLength: sequence length (k in BPTTk)
sequenceLength = 50
#numHiddenFirst: size of first hidden layer
numHiddenFirst = 512
#numTweets: total number of tweets in dataset
numTweets = data.shape[0]
#seqPerSegment: sequences (of size sequenceLength) per mini-epoch. 
#Lowers maximum memory usage.
seqPerSegment = 10000

X = []
y = []

#create input and target datasets from loaded data.
for i in range(numTweets):
    for j in range(0, int(tweetLen[i])-sequenceLength, 1):
        seq_in = data[i, j:j+sequenceLength, :]
        seq_out = data[i, j+sequenceLength, 0:dictLen]
        X.append(seq_in)
        y.append(seq_out)

#X: [10000 (numTweets), 40 (sequenceLength), 365(inputSize)].
n_examples = len(X)
numSegments = np.ceil(n_examples/seqPerSegment).astype(int)
print('# of sequences per segments: ', seqPerSegment)
print('# of segments: ', numSegments)

#building cLSTM model
print("\n")
print("Start building model ....")
model = Sequential()

model.add(LSTM(numHiddenFirst, input_shape=(sequenceLength, inputSize)))

model.add(Dense(numHiddenFirst))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Dense(dictLen))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

#train on mini-epochs (sized seqPerSegment) to lower total RAM usage.
for seg in range(numSegments):
    dataX = np.asarray(X[seg*seqPerSegment: (seg+1)*seqPerSegment])
    datay = np.asarray(y[seg*seqPerSegment: (seg+1)*seqPerSegment])
    print("Input shape: ", dataX.shape)
    print("Output shape: ", datay.shape)
    model.fit(dataX, datay, nb_epoch=20, batch_size=128)
