# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from utils import get_data, preprocess, get_sorted_words, get_filtered_dict, get_words_dict, get_vectors

'''COLUMNS: id, keyword, location, text, target'''
train, test = get_data('./data/train.csv', './data/test.csv')

train.text = preprocess(train.text)

words_list_disaster = get_sorted_words(get_filtered_dict(get_words_dict(train.text[train.target == 1]), 5))

words_list_disaster = ['!!!&&&***&&&!!!'] + words_list_disaster

vectors, words_indexes = get_vectors(train.text, words_list_disaster, 17)
target = list(train.target)

batch_size = 64
epochs = 200
verbose = 1
length = 17

model = Sequential()

model.add(Dense(length, activation='relu', input_dim=length))
model.add(Dense(2 * length, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

x_train, y_train = np.array(words_indexes), np.array(target)

model.fit(
    x=np.array(x_train),
    y=np.array(y_train),
    batch_size=batch_size,
    epochs=epochs,
    verbose=verbose,
    shuffle=True
    # validation_data=(
    #     np.array(x_val),
    #     np.array(y_val)
    # )
)

# model.save(path)

# Summary: accuracy ~0.71, took base words only from disaster tweets, it increased accuracy.
