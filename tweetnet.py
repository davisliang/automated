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
from keras.optimizers import Adagrad
from keras.layers import Dropout
from keras.layers import BatchNormalization
from tweetGenerator import generateText
from keras.callbacks import ModelCheckpoint

print("Start loading data ...")
data, dictLen, tweetLen, dictionary = loadData({},np.array([])) 
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
seqPerSegment = 5000

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
numEpochs=50
#print('# of sequences per segments: ', seqPerSegment)
#print('# of segments: ', numSegments)

#building cLSTM model
#print("\n")
print("Start building model ....")
model = Sequential()

model.add(LSTM(numHiddenFirst, input_shape=(sequenceLength, inputSize), return_sequences=True))
model.add(LSTM(numHiddenFirst))

model.add(Dense(numHiddenFirst))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Dense(numHiddenFirst))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Dense(dictLen))
model.add(Activation('softmax'))

optimizer = Adagrad()
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
print("Finished building model.")
#define file checkpoint
filePath = "intermediateWeights.hdf5"
checkPoint = ModelCheckpoint(filePath, monitor='loss', verbose=1)
callbacksList = [checkPoint]

#train on mini-epochs (sized seqPerSegment) to lower total RAM usage.
for epoch in range(numEpochs):
    for seg in range(numSegments):
        print("\n")
        print "Segment: ", seg, "/", numSegments, " | Epoch: ", epoch, "/", numEpochs 
        dataX = np.asarray(X[seg*seqPerSegment: (seg+1)*seqPerSegment])
        datay = np.asarray(y[seg*seqPerSegment: (seg+1)*seqPerSegment])
        #print("Input shape: ", dataX.shape)
        #print("Output shape: ", datay.shape)
        model.fit(dataX, datay, nb_epoch=1, batch_size=128, callbacks=callbacksList)
        
        generateText(dictionary, data, dictLen, tweetLen, X, y, 
        inputSize, sequenceLength, numHiddenFirst, numTweets, seqPerSegment,
        n_examples, numSegments)
        




