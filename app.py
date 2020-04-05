# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from inits import tweets_preprocessor
from tweets_vectorization import TweetsVectorization
from models import Models

'''COLUMNS: id, keyword, location, text, target'''
data = pd.read_csv('./data/train.csv')

# data = data.sample(frac=1).reset_index(drop=True)

x_train, y_train, x_val, y_val, words, vectors, max_vector_len = TweetsVectorization.get_prepared_data(
    tweets_preprocessor=tweets_preprocessor,
    tweets=data.text,
    target=data.target,
    preprocess_options={
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
    },
    # tweets_for_words_base=data.text[data.target == 1],
    words_reputation_filter=0,
    train_percentage=0.8
)

batch_size = 256
epochs = 15
verbose = 1
embedding_dim = 256
lstm_units = 256
input_length = max_vector_len
embedding_options = {
    'input_dim': len(words),
    'output_dim': embedding_dim,
    'input_length': input_length
}

model = Models.get_binary_classification_model(embedding_options)
model.fit(
    x=np.array(x_train),
    y=np.array(y_train),
    batch_size=batch_size,
    epochs=epochs,
    verbose=verbose,
    shuffle=True,
    validation_data=(
        np.array(x_val),
        np.array(y_val)
    )
)

# model = Models.get_lstm_model(embedding_options, lstm_units)
# model.fit(
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

# model = Models.get_mlp_for_binary_classification_model(embedding_options, [64])
# model.fit(
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

# VECTORIZATION USING COUNTER
# vectorizer = CountVectorizer(analyzer='word', binary=True)
# vectorizer.fit(data.text)
#
# vectorized_data = vectorizer.transform(data.text).todense()
#
# train, test, train_target, test_target = train_test_split(
#     vectorized_data,
#     data.target,
#     test_size=0.2,
#     random_state=0
# )
