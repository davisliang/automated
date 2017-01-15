'''
Reads weight matrix from hdf5 file and generates text using seed.
'''
import numpy as np
import pickle as pkl
import  numpy as np
from numpy import random
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
import sys

def sample(preds, temperature=1.0):

	preds = np.asarray(preds).astype('float64')
	preds = np.log(preds) / temperature
	exp_preds = np.exp(preds)
	preds = exp_preds / np.sum(exp_preds)
	preds = np.squeeze(preds)
	probas = np.random.multinomial(1, preds, 1)
	return np.argmax(probas)

def generateText(dictionary, data, dictLen, tweetLen, X, y, 
	inputSize, sequenceLength, numHiddenFirst, numTweets, seqPerSegment,
	n_examples, numSegments):

	# data shape = #tweets x 141 x inputSize(365)
	#initialize inverse dictionary to map integers to characterse
	inverseDictionary = {v: k for k, v in dictionary.iteritems()}

	#building cLSTM model
	print("\n")
	print("Generating Text... ")
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


	#load the network weights
	fileName = "intermediateWeights.hdf5"
	model.load_weights(fileName)
	model.compile(loss='categorical_crossentropy', optimizer='adam')

	#initializing to random seed
	seedTweet = np.random.randint(n_examples, size=1)

	contextVector=np.zeros(inputSize-dictLen)

	printSeed="SEED: "
	for c in range(sequenceLength):
		pattern = X[seedTweet][c,:]

		contextVector = pattern[dictLen:]

		counter = 0
		for i in range(dictLen):
			if(pattern[i] == 1):
				counter = i
				break
		
		if(counter>=(dictLen-1)):
			printSeed = printSeed + "<<EOS>>"
			continue;

		printSeed = printSeed + inverseDictionary[counter]

	print printSeed

	#generate characters
	charsGenerated=140
	temperatures = [0.001, 0.1, 0.25, 0.5, 1.0]
	for run in range(len(temperatures)+1):
		printResult = "GENERATED TEXT WITH TEMPERATURE "
		if run < len(temperatures):
			temperature = temperatures[run]
			printResult = printResult + str(temperature)+ ": "

		x = X[seedTweet][0:sequenceLength]
		inputVector = np.reshape(x,(1,len(x),len(x[0])))
		    
		for i in range(charsGenerated):

			prediction = model.predict(inputVector, verbose=0)
			if run < len(temperatures):
				rand_index = sample(prediction, temperature)
			else:
				rand_index = np.argmax(prediction)

			if(rand_index==(dictLen-1)):
				printResult = printResult + "<<EOS>>"
				break

			result = inverseDictionary[rand_index]
			printResult = printResult+result

			charVector=np.zeros(dictLen)
			charVector[rand_index]=1
			currInput = np.concatenate((charVector,contextVector))

			concatVector = np.reshape(currInput, (1,1,len(currInput)))

			inputVector=np.concatenate((inputVector,concatVector), axis=1)
			inputVector=inputVector[:,1:len(inputVector[0]),:]

		print printResult
