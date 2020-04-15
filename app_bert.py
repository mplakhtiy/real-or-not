# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow_hub as hub
from bert_tokenization import FullTokenizer
from data import train_data as data, test_data_with_target as test_data
from tweets import Helpers, tweets_preprocessor
from models import Keras, TestDataCallback
from utils import log
import os


def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []

    for text in texts:
        text = tokenizer.tokenize(text)
        text = text[:max_len - 2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)
        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len

        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)

    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)


def build_model(bert_layer, max_len=512, lr=2e-6):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]
    out = Dense(1, activation='sigmoid')(clf_output)

    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])

    return model


DATA = {
    'VALIDATION_PERCENTAGE': 0.2,
    'PREPROCESS_OPTRIONS': {
        'add_link_flag': True,
        'add_user_flag': True,
        'add_hash_flag': True,
        'add_number_flag': True,
        'remove_links': True,
        'remove_users': True,
        'remove_hash': True,
        'unslang': True,
        'split_words': True,
        'stem': False,
        'remove_punctuations': True,
        'remove_numbers': True,
        'to_lower_case': True,
        'remove_stop_words': True,
        'remove_not_alpha': True,
        'join': True
    }
}

# data['preprocessed'] = tweets_preprocessor.preprocess(
#     data.text,
#     DATA['PREPROCESS_OPTRIONS']
# )

Helpers.coorrect_data(data)

# test_data['preprocessed'] = tweets_preprocessor.preprocess(
#     test_data.text,
#     DATA['PREPROCESS_OPTRIONS']
# )

module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"

bert_layer = hub.KerasLayer(module_url, trainable=True)

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()

do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

tokenizer = FullTokenizer(vocab_file, do_lower_case)

train_labels = data.target_relabeled.values
test_labels = test_data.target.values

for lr in [2e-6, 1e-4, 3e-5]:
    for l in [84, 100, 160]:
        for batch_size in [16, 32, 64]:
            p = f'./data/models/{lr}-{l}-{batch_size}'
            if not os.path.exists(p):
                os.makedirs(p)

            checkpoint = ModelCheckpoint(
                p + '/model-{epoch:03d}-{accuracy:03f}-{val_accuracy:03f}.h5',
                verbose=1,
                monitor='val_loss',
                save_best_only=True,
                mode='auto'
            )

            train_input = bert_encode(data.text.values, tokenizer, max_len=l)
            test_input = bert_encode(test_data.text.values, tokenizer, max_len=l)
            model = build_model(bert_layer, max_len=l, lr=lr)

            model.summary()

            test_data_callback = TestDataCallback(
                x_test=test_input,
                y_test=test_labels
            )

            history = model.fit(
                train_input, train_labels,
                validation_split=0.2,
                epochs=4,
                batch_size=batch_size,
                verbose=1,
                callbacks=[checkpoint, test_data_callback]
            )

            mode_history = history.history.copy()
            mode_history['test_loss'] = test_data_callback.loss
            mode_history['test_accuracy'] = test_data_callback.accuracy

            # Keras.draw_graph(mode_history)
            log(
                file='app_bert.py',
                model={'bert': module_url, 'batch_size': batch_size, 'lr': lr, 'length': l},
                model_history=mode_history
            )
