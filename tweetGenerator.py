'''
Reads weight matrix from hdf5 file and generates text using seed.
'''
import numpy as np
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


def generateText(dictionary):

	print("Start loading data ...")
	data, dictLen, tweetLen, dictionary = loadData({},np.array([])) 
	# data shape = #tweets x 141 x inputSize(365)
	#initialize inverse dictionary to map integers to characterse
	inverseDictionary = {v: k for k, v in dictionary.iteritems()}
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

	#load the network weights
	fileName = "intermediateWeights.hdf5"
	model.load_weights(fileName)
	model.compile(loss='categorical_crossentropy', optimizer='adam')

	#initializing to random seed
	start = np.random.randint(0, len)
	pattern = X[start]
	print "seed: "
	print "\"", ' '.join([inverseDictionary[value] for value in pattern]), "\""
	#generate characters
	for i in range(140):
		x = numpy.reshape(pattern, (1, len(pattern), 1))
		x = x / float(dictLen)
		prediction = model.predict(x, verbose=0)
		index = numpy.argmax(prediction)
		result = inverseDictionary[index]
		seq_in = [inverseDictionary[value] for value in pattern]
		sys.stdout.write(result)
		pattern.append(index)
		pattern = pattern[1:len(pattern)]
	print "\nDone."