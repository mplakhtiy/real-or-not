# -*- coding: utf-8 -*-
import numpy as np
from tweets import TweetsVectorization, tweets_preprocessor
from models import KerasModels
from utils import draw_keras_graph
from data import data

# shuffle data
# data = data.sample(frac=1).reset_index(drop=True)

WORDS_REPUTATION_FILTER = 0
TRAIN_PERCENTAGE = 0.8
PREPROCESS_OPTRIONS = {
    'remove_links': True,
    'remove_users': True,
    'remove_hash': True,
    'unslang': True,
    'split_words': True,
    'stem': True,
    'remove_punctuations': True,
    'remove_numbers': True,
    'to_lower_case': True,
    'remove_stop_words': True,
    'remove_not_alpha': True,
    'join': False
}

x_train, y_train, x_val, y_val, words, vectors, max_vector_len = TweetsVectorization.get_prepared_data_based_on_words_indexes(
    tweets_preprocessor=tweets_preprocessor,
    tweets=data.text,
    target=data.target,
    preprocess_options=PREPROCESS_OPTRIONS,
    # tweets_for_words_base=data.text[data.target == 1],
    words_reputation_filter=WORDS_REPUTATION_FILTER,
    train_percentage=TRAIN_PERCENTAGE
)

BATCH_SIZE = 256
EPOCHS = 15
VERBOSE = 1
EMBEDING_DIM = 256
LSTM_UNITS = 128
INPUT_LENGTH = max_vector_len

EMBEDDING_OPTIONS = {
    'input_dim': len(words),
    'output_dim': EMBEDING_DIM,
    'input_length': INPUT_LENGTH
}

model = KerasModels.get_binary_classification_model(EMBEDDING_OPTIONS)
history = model.fit(
    x=np.array(x_train),
    y=np.array(y_train),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=VERBOSE,
    shuffle=True,
    validation_data=(
        np.array(x_val),
        np.array(y_val)
    )
)

# model = KerasModels.get_lstm_model(embedding_options, lstm_units)
# history = model.fit(
#     x=np.array(x_train),
#     y=np.array(y_train),
#     batch_size=batch_size,
#     epochs=epochs,
#     verbose=verbose,
#     shuffle=True,
#     validation_data=(
#         np.array(x_val),
#         np.array(y_val)
#     )
# )

# model = KerasModels.get_mlp_for_binary_classification_model(embedding_options, [64])
# history = model.fit(
#     x=np.array(x_train),
#     y=np.array(y_train),
#     batch_size=batch_size,
#     epochs=epochs,
#     verbose=verbose,
#     shuffle=True,
#     validation_data=(
#         np.array(x_val),
#         np.array(y_val)
#     )
# )

draw_keras_graph(history)
