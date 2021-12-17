import tensorflow as tf
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
    tokenized_texts = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(tokenized_texts)):
        input_sequences.append(tokenized_texts[:i + 1])

max_sequence_length = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre'))

# Load model
loaded_model = tf.keras.models.load_model("TrumpModel.h5")

# Make a prediction
flag = True
next_words = 20
while (flag):

    input_text = input("Enter the starting text: ")
    if (input_text == '-1'):
        flag = False

    for i in range(next_words):
        tokenized_texts = tokenizer.texts_to_sequences([input_text])[0]
        tokenized_texts = pad_sequences([tokenized_texts], maxlen=max_sequence_length - 1, padding='pre')
        predicted = np.argmax(loaded_model.predict(tokenized_texts), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        input_text += " " + output_word

    print(input_text)
