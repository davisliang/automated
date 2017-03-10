from keras.layers import Input, Dense, Embedding, LSTM, merge
from keras.models import Model

# this returns a tensor
contextX, contexty, taskX, taskY = loadData();

text_input = Input(shape=(100,), dtype='float32', name='text_input')

lstm_body = lstm(32)(text_input)

lstm_context = lstm(32)(lstm_body)
fc_context = Dense(256)(lstm_context)
out_context = Dense(300)(fc_context)

lstm_task = lstm(32)(lstm_body)
fc_task = Dense(256)(lstm_task)
fc_out = Dense(300)(fc_task)


