from tensorflow.keras.layers import Dense
import uuid
from tweets import Helpers, tweets_preprocessor
from models import Keras, Sklearn
from tensorflow.keras.models import load_model
from utils import get_glove_embeddings, save_to_file, ensure_path_exists, get_from_file, load_classifier
from data import train_data, test_data
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from configs import get_preprocessing_algorithm, get_model_config
from pathlib import Path
import numpy as np

saved_models_pathes = [str(lst) for lst in list(Path("./data-saved-models/").rglob("*.h5"))] + [str(lst) for lst in
                                                                                                list(Path(
                                                                                                    "./data-saved-models/").rglob(
                                                                                                    "*.pickle"))]

SEED = 42
USE_GLOVE = True

NETWORKS_KEYS = ['LSTM', 'LSTM_DROPOUT', 'BI_LSTM', 'FASTTEXT', 'RCNN', 'CNN', 'RNN', 'GRU']
CLASSIFIERS_KEYS = ['RIDGE', 'SVC', 'LOGISTIC_REGRESSION', 'SGD']
PREPROCESSING_ALGORITHM_IDS = [
    '1258a9d2',
    '60314ef9',
    '4c2e484d',
    '8b7db91c',
    '7bc816a1',
    'a85c8435',
    'b054e509',
    '2e359f0b',
    '71bd09db',
    'd3cc3c6e',
]

PREDICTIONS = {
    'glove-true': {},
    'glove-false': {},
    'classifiers': {},
}

STATICS = {}

# def flatten(lst):
#     return [sub_sub_l for sub_l in lst for sub_sub_l in sub_l]
#
#
# def get_probs(lst):
#     return [(d + 1) / 2 for d in lst]
#
#
# for folder in PREDICTIONS.keys():
#     is_classifier = folder == 'classifiers'
#
#     for key in NETWORKS_KEYS if not is_classifier else CLASSIFIERS_KEYS:
#         model_path = [x for x in saved_models_pathes if f'{folder}/{key}/' in x][0]
#         data = model_path.split('/')[-1].split('-')
#         preprocessing_algorithm_id = data[1]
#         preprocessing_algorithm = get_preprocessing_algorithm(preprocessing_algorithm_id, join=is_classifier)
#
#         train_data_preprocessed = tweets_preprocessor.preprocess(
#             train_data.text,
#             preprocessing_algorithm,
#             keywords=train_data.keyword,
#             locations=train_data.location
#         )
#
#         test_data_preprocessed = tweets_preprocessor.preprocess(
#             test_data.text,
#             preprocessing_algorithm,
#             keywords=test_data.keyword,
#             locations=test_data.location
#         )
#
#         train_inputs, val_inputs, train_targets, val_targets = train_test_split(
#             train_data_preprocessed,
#             train_data['target'],
#             test_size=0.3,
#             random_state=SEED
#         )
#
#         if not is_classifier:
#             keras_tokenizer = Tokenizer()
#
#             (x_train, x_val, x_test), input_dim, input_len = Helpers.get_model_inputs(
#                 (train_inputs, val_inputs, test_data_preprocessed),
#                 keras_tokenizer
#             )
#
#             y_train = train_targets
#             y_val = val_targets
#             y_test = test_data.target.values
#
#             model = load_model(model_path)
#
#             print(key)
#             print(f'a - {model.evaluate(x_train, y_train, verbose=0)}')
#             print(f'va - {model.evaluate(x_val, y_val, verbose=0)}')
#             print(f'ta - {model.evaluate(x_test, y_test, verbose=0)}')
#             print('----------------------------')
#
#             if key == NETWORKS_KEYS[0]:
#                 STATICS['y_train'] = y_train.tolist()
#                 STATICS['y_val'] = y_val.tolist()
#                 STATICS['y_test'] = y_test.tolist()
#
#             x_train_predictions = model.predict(x_train)
#             x_val_predictions = model.predict(x_val)
#             x_test_predictions = model.predict(x_test)
#
#             PREDICTIONS[folder][key] = {
#                 'x_train_predictions': flatten(x_train_predictions.tolist()),
#                 'x_val_predictions': flatten(x_val_predictions.tolist()),
#                 'x_test_predictions': flatten(x_test_predictions.tolist())
#             }
#         else:
#             vectorizer = Sklearn.VECTORIZERS[data[2]](**{
#                 'binary': True,
#                 'ngram_range': (int(data[3]), int(data[4]))
#             })
#
#             x_train = vectorizer.fit_transform(train_inputs).todense()
#             y_train = train_targets
#
#             x_val = vectorizer.transform(val_inputs).todense()
#             y_val = val_targets
#
#             x_test = vectorizer.transform(test_data_preprocessed).todense()
#             y_test = test_data.target.values
#
#             cls = load_classifier(model_path)
#
#             print(key)
#             # print(f'a - {cls.score(x_train, y_train)}')
#             # print(f'va - {cls.score(x_val, y_val)}')
#             # print(f'ta - {cls.score(x_test, y_test)}')
#             # print('----------------------------')
#
#             PREDICTIONS[folder][key] = {
#                 'x_train_predictions': cls.predict(x_train).tolist(),
#                 'x_val_predictions': cls.predict(x_val).tolist(),
#                 'x_test_predictions': cls.predict(x_test).tolist()
#             }
#
# save_to_file('./predictions.json', {**PREDICTIONS, **STATICS})

predictions = get_from_file('./predictions.json')

x_train = []
x_val = []
x_test = []

train_l = len(predictions['y_train'])
val_l = len(predictions['y_val'])
test_l = len(predictions['y_test'])


def fill(lst, lngth, prdctns, key):
    for i in range(lngth):
        lst.append([])
        for folder in ['glove-true', 'classifiers']:
            is_classifiers = folder == 'classifiers'

            for n_key in NETWORKS_KEYS if not is_classifiers else CLASSIFIERS_KEYS:
                lst[i].append(round(prdctns[folder][n_key][key][i]))
                # if prdctns[folder][n_key][key][i] > 0.6:
                #     lst[i].append(1)
                # else:
                #     lst[i].append(0)


fill(x_train, train_l, predictions, 'x_train_predictions')
fill(x_val, val_l, predictions, 'x_val_predictions')
fill(x_test, test_l, predictions, 'x_test_predictions')

y_train = predictions['y_train']
y_val = predictions['y_val']
y_test = predictions['y_test']

config = {
    'BATCH_SIZE': 8,
    'EPOCHS': 100,
    'OPTIMIZER': 'adam',
    'LEARNING_RATE': 1e-4,
    'INPUT_DIM': 12,
    'HIDDEN_LAYER': 12,
}
model = Keras.get_sequential_model(
    [
        Dense(config['INPUT_DIM'], input_dim=config['INPUT_DIM']),
        Dense(config['HIDDEN_LAYER']),
    ],
    config
)

Keras.fit(model, (x_val, y_val, x_test, y_test), config)
