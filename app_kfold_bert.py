# -*- coding: utf-8 -*-
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow_hub as hub
from bert_tokenization import FullTokenizer
from data import train_data, test_data
from models import Keras, TestDataCallback
from sklearn.model_selection import StratifiedKFold
from tweets import Helpers, tweets_preprocessor
from utils import log_model
from configs import get_preprocessing_algorithm
import uuid
import gc

PREPROCESSING_ALGORITHMS = get_preprocessing_algorithm(join=True)
PREPROCESSING_ALGORITHMS['None'] = {}

SEED = 7
KFOLD = 10

for algorithm_id, preprocessing_algorithm in PREPROCESSING_ALGORITHMS.items():
    MODEL = {
        'UUID': str(uuid.uuid4()),
        # 'BERT': 'bert_en_uncased_L-12_H-768_A-12',
        'BERT': 'bert_en_uncased_L-24_H-1024_A-16',
        'BERT_VERSION': 1,
        'BATCH_SIZE': 16,
        'EPOCHS': 3,
        'OPTIMIZER': 'adam',
        'LEARNING_RATE': 2e-6,
        'PREPROCESSING_ALGORITHM_UUID': algorithm_id,
        'PREPROCESSING_ALGORITHM': preprocessing_algorithm,
        'KFOLD_HISTORY': []
    }

    kfold = StratifiedKFold(n_splits=KFOLD, shuffle=True, random_state=SEED)

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
