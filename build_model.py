import pickle as pkl
import  numpy as np
from load_data import load_data
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Activation
from keras.optimizers import RMSprop
from keras.layers import Dropout

print("Start loading data ...")
data, n_char, tweet_length = load_data() # data shape = #tweets x 141 x input_dim(365)
print("Finished loading data")

input_dim = data.shape[2]
input_length = 40
hidden1 = 256
n_tweets = data.shape[0]
n_windows = 10000

X = []
y = []
for i in range(n_tweets):
    for j in range(0, int(tweet_length[i])-input_length, 1):
        seq_in = data[i, j:j+input_length, :]
        seq_out = data[i, j+input_length, 0:n_char]
        X.append(seq_in)
        y.append(seq_out)

n_examples = len(X)
n_segments = np.ceil(n_examples/n_windows).astype(int)
print('# of sequences per segments: ', n_windows)
print('# of segments: ', n_segments)

print("\n")
print("Start building model ....")
model = Sequential()
model.add(LSTM(hidden1, input_shape=(input_length, input_dim)))
model.add(Dense(hidden1))
model.add(Dense(n_char))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

for seg in range(n_segments):
    dataX = np.asarray(X[seg*n_windows: (seg+1)*n_windows])
    datay = np.asarray(y[seg*n_windows: (seg+1)*n_windows])
    print("Input shape: ", dataX.shape)
    print("Output shape: ", datay.shape)
    model.fit(dataX, datay, nb_epoch=20, batch_size=128)
