# -*- coding: utf-8 -*-
import tensorflow_hub as hub
from bert_tokenization import FullTokenizer
from data import train_data, test_data
from models import Keras
from sklearn.model_selection import StratifiedKFold
from tweets import Helpers, tweets_preprocessor
from utils import log_model, get_from_file
from configs import get_preprocessing_algorithm
from multiprocessing import Process
import uuid
import numpy as np

PREPROCESSING_ALGORITHM_ID = '2e359f0b'
PREPROCESSING_ALGORITHM = get_preprocessing_algorithm(PREPROCESSING_ALGORITHM_ID, join=True)

failed_10000 = get_from_file('./v1/10000/10000-failed.json')['failed_indexes']
failed_7000 = get_from_file('./v1/7000/7000-failed.json')['failed_indexes']

SEED = 7
KFOLD = 10

MODEL_DICT = {
    'TYPE': 'BERT',
    'UUID': str(uuid.uuid4()),
    # 'BERT': 'bert_en_uncased_L-12_H-768_A-12',
    'BERT': 'bert_en_uncased_L-24_H-1024_A-16',
    'BERT_VERSION': 1,
    'BATCH_SIZE': 16,
    'EPOCHS': 3,
    'OPTIMIZER': 'adam',
    'LEARNING_RATE': 2e-6,
    'PREPROCESSING_ALGORITHM_UUID': PREPROCESSING_ALGORITHM_ID,
    'PREPROCESSING_ALGORITHM': PREPROCESSING_ALGORITHM,
    'KFOLD_HISTORY': []
}

kfold = StratifiedKFold(n_splits=KFOLD, shuffle=True, random_state=SEED)

train_data['preprocessed'] = tweets_preprocessor.preprocess(
    train_data.text,
    PREPROCESSING_ALGORITHM,
    keywords=train_data.keyword,
    locations=train_data.location
)

test_data['preprocessed'] = tweets_preprocessor.preprocess(
    test_data.text,
    PREPROCESSING_ALGORITHM,
    keywords=test_data.keyword,
    locations=test_data.location
)

# inputs = np.array(train_data['preprocessed'])
# targets = np.array(train_data['target'])
# x_f = inputs[failed_7000]
# y_f = targets[failed_7000]
# inputs = np.delete(inputs, failed_7000)
# targets = np.delete(targets, failed_7000)

inputs = np.array(train_data['preprocessed'])
targets = np.array(train_data['target'])

x_train_set = np.delete(inputs, failed_7000)
y_train_set = np.delete(targets, failed_7000)

inputs = inputs[failed_7000]
targets = targets[failed_7000]

x_test_set = test_data.preprocessed
y_test_set = test_data.target.values


def train_bert(train, validation, MODEL, index):
    x_train = inputs[train]
    y_train = targets[train]

    x_val = inputs[validation]
    y_val = targets[validation]

    bert_layer = hub.KerasLayer(f'https://tfhub.dev/tensorflow/{MODEL["BERT"]}/{MODEL["BERT_VERSION"]}',
                                trainable=True)
    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = FullTokenizer(vocab_file, do_lower_case)

    x_train, INPUT_LENGTH = Helpers.get_bert_input(x_train, tokenizer)
    x_val = Helpers.get_bert_input(x_val, tokenizer, input_length=INPUT_LENGTH)
    x_train_s = Helpers.get_bert_input(x_train_set, tokenizer, input_length=INPUT_LENGTH)
    x_test_s = Helpers.get_bert_input(x_test_set, tokenizer, input_length=INPUT_LENGTH)

    MODEL['INPUT_LENGTH'] = INPUT_LENGTH

    model = Keras.get_bert_model(
        bert_layer=bert_layer,
        input_length=INPUT_LENGTH,
        optimizer=MODEL['OPTIMIZER'],
        learning_rate=MODEL['LEARNING_RATE']
    )

    history = Keras.fit(model, (x_train, y_train, x_val, y_val, x_train_s, y_train_set, x_test_s, y_test_set), MODEL)

    MODEL['KFOLD_HISTORY'].append(history)

    log_model(MODEL, index)


k = 0
for train, validation in kfold.split(inputs, targets):
    process_train = Process(target=train_bert, args=(train, validation, MODEL_DICT, k))
    process_train.start()
    process_train.join()
    k += 1
