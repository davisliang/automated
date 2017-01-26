'''
Reads weight matrix from hdf5 file and generates text using seed.
'''
import numpy as np
import pickle as pkl
import  numpy as np
import h5py
from numpy import random
from loadData_lstm import loadData
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Activation
from keras.optimizers import RMSprop
from keras.optimizers import Adagrad
from keras.layers import Dropout
from keras.layers import BatchNormalization
from scipy.stats import rv_discrete
import sys
from os.path import expanduser

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generateText(model, tweets, sequenceLength, vocabLen, dictionary):
    start_index = random.randint(len(tweets))
    inverseDictionary = {v: k for k, v in dictionary.iteritems()}
    
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print("\n")
        print('----- diversity:', diversity)

        generated = ""
        seed = tweets[start_index][6:sequenceLength+6]
        print('----- Generating with seed: "' + seed + '"')
        generated += seed
        sys.stdout.write(generated)

        for i in range(140):
            x = np.zeros((1, sequenceLength, vocabLen))
            for j, ch in enumerate(seed):
                x[0, j, dictionary.get(ch)] = 1

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            
            if next_index == vocabLen - 1:
                next_char = "<EOS>"
                generated += next_char
                seed = seed[1:] + next_char
                sys.stdout.write(next_char)
                sys.stdout.flush()
                break
            
            else:
                next_char = inverseDictionary[next_index]
                generated += next_char
                seed = seed[1:] + next_char
                sys.stdout.write(next_char)
                sys.stdout.flush()
        print("\n")
