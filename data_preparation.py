# -*- coding: utf-8 -*-
from tweets import TweetsVectorization, tweets_preprocessor
from data import data
from utils import save_to_file

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

DESCRIPTION = 'All preprocess options are true. Vectors of words indexes from Vocabulary. Vocabulary is created base on true and false tweets. Vocabulary is padded with a start symbol.'
FILE_NAME = 'words_indexes_preprocess_all_true_with_start_symbol.json'

x_train, y_train, x_val, y_val, vocabulary, max_vector_len = TweetsVectorization.get_prepared_data_based_on_words_indexes(
    tweets_preprocessor=tweets_preprocessor,
    tweets=data.text,
    target=data.target,
    preprocess_options=PREPROCESS_OPTRIONS,
    # tweets_for_words_base=data.text[data.target == 1],
    words_reputation_filter=WORDS_REPUTATION_FILTER,
    train_percentage=TRAIN_PERCENTAGE,
    add_start_symbol=True
)

save_to_file(f'./data/prepared_data/{FILE_NAME}', {
    "description": DESCRIPTION,
    'preprocess_options': PREPROCESS_OPTRIONS,
    'x_train': x_train,
    'y_train': y_train,
    'x_val': x_val,
    'y_val': y_val,
    'vocabulary': vocabulary,
    'vocabulary_len': len(vocabulary),
    'max_vector_len': max_vector_len,
    'train_percentage': TRAIN_PERCENTAGE
})

# DESCRIPTION = 'All preprocess options are true. Vectors of words are based on Count Vectorizer.'
# FILE_NAME = 'count_vectorizer_preprocess_all_true.json'
# x_train, y_train, x_val, y_val = TweetsVectorization.get_prepared_data_based_on_count_vectorizer(
#     tweets_preprocessor=tweets_preprocessor,
#     tweets=data.text,
#     target=data.target,
#     preprocess_options=PREPROCESS_OPTRIONS,
#     train_percentage=TRAIN_PERCENTAGE
# )
#
# save_to_file(f'./data/prepared_data/{FILE_NAME}', {
#     "description": DESCRIPTION,
#     'preprocess_options': PREPROCESS_OPTRIONS,
#     'x_train': x_train.tolist(),
#     'y_train': y_train.tolist(),
#     'x_val': x_val.tolist(),
#     'y_val': y_val.tolist(),
#     'train_percentage': TRAIN_PERCENTAGE
# })
