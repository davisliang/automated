'''
This code should not be used. Mega inefficient and a waste of RAM.
Will fix soon... :). Code calling this method from other classes
Have been commented out.
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


def generateText(dictionary, data, dictLen, tweetLen, X, y, 
	inputSize, sequenceLength, numHiddenFirst, numTweets, seqPerSegment,
	n_examples, numSegments):

	print("Start loading data ...")

	# data shape = #tweets x 141 x inputSize(365)
	#initialize inverse dictionary to map integers to characterse
	inverseDictionary = {v: k for k, v in dictionary.iteritems()}
	print("Finished loading data")

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
