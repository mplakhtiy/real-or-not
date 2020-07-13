# -*- coding: utf-8 -*-
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow_hub as hub
from bert_tokenization import FullTokenizer
from data import train_data, test_data
from models import Keras, TestDataCallback
from sklearn.model_selection import StratifiedKFold
from tweets import Helpers, tweets_preprocessor
from utils import log_model
import os
import uuid
import gc

PREPROCESSING_ALGORITHMS = {
    'None': {},
    '1258a9d2-111e-4d4a-acda-852dd7ba3e88': {
        'add_link_flag': True, 'add_user_flag': True, 'add_hash_flag': True, 'add_number_flag': True,
        'add_keyword_flag': False, 'add_location_flag': False, 'remove_links': True, 'remove_users': True,
        'remove_hash': True, 'unslang': True, 'split_words': True, 'stem': False, 'remove_punctuations': True,
        'remove_numbers': True, 'to_lower_case': True, 'remove_stop_words': True, 'remove_not_alpha': True,
        'join': True},
    '60314ef9-271d-4865-a7db-6889b1670f59': {
        'add_link_flag': False, 'add_user_flag': True, 'add_hash_flag': True, 'add_number_flag': False,
        'add_keyword_flag': False, 'add_location_flag': False, 'remove_links': True, 'remove_users': True,
        'remove_hash': True, 'unslang': True, 'split_words': True, 'stem': False, 'remove_punctuations': True,
        'remove_numbers': True, 'to_lower_case': True, 'remove_stop_words': True, 'remove_not_alpha': True,
        'join': True},
    '4c2e484d-5cb8-4e3e-ba7b-679ae7a73fca': {
        'add_link_flag': True, 'add_user_flag': True, 'add_hash_flag': True, 'add_number_flag': True,
        'add_keyword_flag': True, 'add_location_flag': True, 'remove_links': True, 'remove_users': True,
        'remove_hash': True, 'unslang': True, 'split_words': True, 'stem': False, 'remove_punctuations': True,
        'remove_numbers': True, 'to_lower_case': True, 'remove_stop_words': True, 'remove_not_alpha': True,
        'join': True},
    '8b7db91c-c8bf-40f2-986a-83a659b63ba6': {
        'add_link_flag': True, 'add_user_flag': True, 'add_hash_flag': True, 'add_number_flag': False,
        'add_keyword_flag': False, 'add_location_flag': False, 'remove_links': True, 'remove_users': True,
        'remove_hash': True, 'unslang': True, 'split_words': True, 'stem': False, 'remove_punctuations': True,
        'remove_numbers': True, 'to_lower_case': True, 'remove_stop_words': True, 'remove_not_alpha': True,
        'join': True},
    '7bc816a1-25df-4649-8570-0012d0acd72a': {
        'add_link_flag': False, 'add_user_flag': False, 'add_hash_flag': False, 'add_number_flag': False,
        'add_keyword_flag': False, 'add_location_flag': False, 'remove_links': True, 'remove_users': True,
        'remove_hash': True, 'unslang': True, 'split_words': True, 'stem': False, 'remove_punctuations': True,
        'remove_numbers': True, 'to_lower_case': True, 'remove_stop_words': True, 'remove_not_alpha': True,
        'join': True},
    'a85c8435-6f23-4015-9e8c-19547222d6ce': {
        'add_link_flag': True, 'add_user_flag': True, 'add_hash_flag': True, 'add_number_flag': True,
        'add_keyword_flag': False, 'add_location_flag': False, 'remove_links': True, 'remove_users': True,
        'remove_hash': True, 'unslang': True, 'split_words': True, 'stem': False, 'remove_punctuations': False,
        'remove_numbers': True, 'to_lower_case': True, 'remove_stop_words': False, 'remove_not_alpha': False,
        'join': True},
    'b054e509-4f04-44f2-bcf9-14fa8af4eeed': {
        'add_link_flag': True, 'add_user_flag': True, 'add_hash_flag': True, 'add_number_flag': True,
        'add_keyword_flag': False, 'add_location_flag': False, 'remove_links': True, 'remove_users': True,
        'remove_hash': True, 'unslang': True, 'split_words': False, 'stem': False, 'remove_punctuations': False,
        'remove_numbers': True, 'to_lower_case': True, 'remove_stop_words': False, 'remove_not_alpha': False,
        'join': True},
    '2e359f0b-bfb9-4eda-b2a4-cd839c122de6': {
        'add_link_flag': False, 'add_user_flag': False, 'add_hash_flag': False, 'add_number_flag': False,
        'add_keyword_flag': False, 'add_location_flag': False, 'remove_links': True, 'remove_users': True,
        'remove_hash': True, 'unslang': True, 'split_words': False, 'stem': False, 'remove_punctuations': False,
        'remove_numbers': True, 'to_lower_case': True, 'remove_stop_words': False, 'remove_not_alpha': False,
        'join': True},
    '71bd09db-e104-462d-887a-74389438bb49': {
        'add_link_flag': False, 'add_user_flag': False, 'add_hash_flag': False, 'add_number_flag': False,
        'add_keyword_flag': False, 'add_location_flag': False, 'remove_links': True, 'remove_users': True,
        'remove_hash': True, 'unslang': True, 'split_words': False, 'stem': False, 'remove_punctuations': True,
        'remove_numbers': True, 'to_lower_case': True, 'remove_stop_words': False, 'remove_not_alpha': False,
        'join': True},
    'd3cc3c6e-10de-4b27-8712-8017da428e41': {
        'add_link_flag': True, 'add_user_flag': True, 'add_hash_flag': True, 'add_number_flag': True,
        'add_keyword_flag': False, 'add_location_flag': False, 'remove_links': True, 'remove_users': True,
        'remove_hash': True, 'unslang': True, 'split_words': False, 'stem': False, 'remove_punctuations': True,
        'remove_numbers': True, 'to_lower_case': True, 'remove_stop_words': False, 'remove_not_alpha': False,
        'join': True}
}

SEED = 7
KFOLD = 10

for algorithm_id, preprocessing_algorithm in PREPROCESSING_ALGORITHMS.items():
    MODEL = {
        'UUID': str(uuid.uuid4()),
        # 'BERT': 'bert_en_uncased_L-12_H-768_A-12',
        'BERT': 'bert_en_uncased_L-24_H-1024_A-16',
        'BERT_VERSION': 2,
        'BATCH_SIZE': 16,
        'EPOCHS': 4,
        'OPTIMIZER': 'adam',
        'LEARNING_RATE': 2e-6,
        'PREPROCESSING_ALGORITHM_UUID': algorithm_id,
        'PREPROCESSING_ALGORITHM': preprocessing_algorithm,
        'KFOLD_HISTORY': []
    }

    kfold = StratifiedKFold(n_splits=KFOLD, shuffle=True, random_state=SEED)

    print(algorithm_id)

    if algorithm_id != 'None':
        train_data['preprocessed'] = tweets_preprocessor.preprocess(
            train_data.text,
            preprocessing_algorithm,
            keywords=train_data.keyword,
            locations=train_data.location
        )

        test_data['preprocessed'] = tweets_preprocessor.preprocess(
            test_data.text,
            preprocessing_algorithm,
            keywords=test_data.keyword,
            locations=test_data.location
        )
    else:
        train_data['preprocessed'] = train_data.text
        test_data['preprocessed'] = test_data.text

    inputs = train_data['preprocessed']
    targets = train_data['target']

    k = 0

    for train, validation in kfold.split(inputs, targets):
        x_train = inputs[train]
        y_train = targets[train]

        x_val = inputs[validation]
        y_val = targets[validation]

        x_test = test_data.preprocessed
        y_test = test_data.target.values

        bert_layer = None
        vocab_file = None
        do_lower_case = None
        tokenizer = None

        gc.collect()

        bert_layer = hub.KerasLayer(f'https://tfhub.dev/tensorflow/{MODEL["BERT"]}/{MODEL["BERT_VERSION"]}',
                                    trainable=True)
        vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
        do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
        tokenizer = FullTokenizer(vocab_file, do_lower_case)

        gc.collect()

        x_train, INPUT_LENGTH = Helpers.get_bert_input(x_train, tokenizer)
        x_val = Helpers.get_bert_input(x_val, tokenizer, input_length=INPUT_LENGTH)
        x_test = Helpers.get_bert_input(x_test, tokenizer, input_length=INPUT_LENGTH)

        MODEL['INPUT_LENGTH'] = INPUT_LENGTH
        test_data_callback = None
        gc.collect()

        test_data_callback = TestDataCallback(
            x_test=x_test,
            y_test=y_test
        )

        model = None
        gc.collect()

        model = Keras.get_bert_model(
            bert_layer=bert_layer,
            input_length=INPUT_LENGTH,
            optimizer=MODEL['OPTIMIZER'],
            learning_rate=MODEL['LEARNING_RATE']
        )

        history = None
        gc.collect()

        history = model.fit(
            x_train, y_train,
            epochs=MODEL['EPOCHS'],
            batch_size=MODEL['BATCH_SIZE'],
            verbose=1,
            validation_data=(
                x_val,
                y_val
            ),
            callbacks=[test_data_callback]
        )

        gc.collect()

        model_history = history.history.copy()
        model_history['test_loss'] = test_data_callback.loss
        model_history['test_accuracy'] = test_data_callback.accuracy

        MODEL['KFOLD_HISTORY'].append(model_history)

        log_model(MODEL)

        os.system('spd-say "Experiment is finished"')
