import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd

# Prepare data
training = pd.read_csv("TrumpTweets.csv")
training = training['text']

tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                      lower=True)
tokenizer.fit_on_texts(training)

wordCount = len(tokenizer.word_index) + 1

input_sequences = []
for line in training:
	token_list = tokenizer.texts_to_sequences([line])[0]
	for i in range(1, len(token_list)):
		n_gram_sequence = token_list[:i+1]
		input_sequences.append(n_gram_sequence)

max_sequence_length = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre'))

# create predictors and label
xs = input_sequences[:,:-1]
labels = input_sequences[:,-1]
ys = tf.keras.utils.to_categorical(labels, num_classes=wordCount)

# Build model
model = tf.keras.models.Sequential([
    Embedding(wordCount, 80, input_length=max_sequence_length-1),
    LSTM(100, return_sequences=True),
    LSTM(50),
    Dropout(0.1),
    Dense(wordCount/20),
    Dense(wordCount, activation='softmax')
])

# Compile model
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(lr=0.001),
              metrics=['accuracy'])

# Train model
history = model.fit(xs, ys, epochs=100, verbose=1)

# Save a model using the HDF5 format
model.save("TrumpModel.h5")
