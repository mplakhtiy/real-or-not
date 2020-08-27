from tweets import Helpers, tweets_preprocessor
from models import Sklearn
from tensorflow.keras.models import load_model
from utils import save_to_file, load_classifier
from data import train_data, test_data
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from configs import get_preprocessing_algorithm
from pathlib import Path
import numpy as np
from bert_tokenization import FullTokenizer
import tensorflow_hub as hub

saved_models_pathes = [str(lst) for lst in list(Path("./data-saved-models/").rglob("*.h5"))] + \
                      [str(lst) for lst in list(Path("./data-saved-models/").rglob("*.pickle"))]
SEED = 7

NETWORKS_KEYS = ['LSTM', 'LSTM_DROPOUT', 'BI_LSTM', 'LSTM_CNN', 'FASTTEXT', 'RCNN', 'CNN', 'RNN', 'GRU']
BERT_KEY = ['BERT']
BERT_MODEL = {
    'BERT': 'bert_en_uncased_L-24_H-1024_A-16',
    'BERT_VERSION': 1,
}
OMIT_BERT = True
CLASSIFIERS_KEYS = ['RIDGE', 'SVC', 'LOGISTIC_REGRESSION', 'SGD', 'DECISION_TREE', 'RANDOM_FOREST']
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
    # 'bert': {},
    # 'glove-true': {},
    'glove-false': {},
    'classifiers': {},
}

STATICS = {}


def flatten(lst):
    return [sub_sub_l for sub_l in lst for sub_sub_l in sub_l]


def get_probs(lst):
    return [np.exp(d) / (1 + np.exp(d)) for d in lst]


correct_targets_saved = False

for folder in PREDICTIONS.keys():
    is_classifier = folder == 'classifiers'
    is_bert = folder == 'bert'
    keys = None

    if is_bert and OMIT_BERT:
        continue

    if is_classifier:
        keys = CLASSIFIERS_KEYS
    elif is_bert:
        keys = BERT_KEY
    else:
        keys = NETWORKS_KEYS

    for key in keys:
        try:
            model_path = [x for x in saved_models_pathes if f'{folder}/{key}/' in x and f'SEED-{SEED}' in x][0]
        except:
            continue

        data = model_path.split('/')[-1].split('-')
        preprocessing_algorithm_id = data[1]
        preprocessing_algorithm = get_preprocessing_algorithm(
            preprocessing_algorithm_id,
            join=(is_classifier or is_bert)
        )

        train_data_preprocessed = tweets_preprocessor.preprocess(
            train_data.text,
            preprocessing_algorithm,
            keywords=train_data.keyword,
            locations=train_data.location
        )

        test_data_preprocessed = tweets_preprocessor.preprocess(
            test_data.text,
            preprocessing_algorithm,
            keywords=test_data.keyword,
            locations=test_data.location
        )

        train_inputs, val_inputs, train_targets, val_targets = train_test_split(
            train_data_preprocessed,
            train_data['target'],
            test_size=0.3,
            random_state=SEED
        )

        y_train = np.asarray(train_targets)
        y_val = np.asarray(val_targets)
        y_test = np.asarray(test_data.target.values)

        if not correct_targets_saved:
            STATICS['y_train'] = y_train.tolist()
            STATICS['y_val'] = y_val.tolist()
            STATICS['y_test'] = y_test.tolist()
            correct_targets_saved = True

        if is_bert:
            x_train = np.asarray(train_inputs)
            x_val = np.asarray(val_inputs)
            x_test = np.asarray(test_data_preprocessed)

            bert_layer = hub.KerasLayer(
                f'https://tfhub.dev/tensorflow/{BERT_MODEL["BERT"]}/{BERT_MODEL["BERT_VERSION"]}',
                trainable=True
            )

            vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
            do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
            tokenizer = FullTokenizer(vocab_file, do_lower_case)

            x_train, INPUT_LENGTH = Helpers.get_bert_input(x_train, tokenizer)
            x_val = Helpers.get_bert_input(x_val, tokenizer, input_length=INPUT_LENGTH)
            x_test = Helpers.get_bert_input(x_test, tokenizer, input_length=INPUT_LENGTH)

            model = load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})

            print(key)
            # print(f'a - {model.evaluate(x_train, y_train, verbose=0)}')
            # print(f'va - {model.evaluate(x_val, y_val, verbose=0)}')
            # print(f'ta - {model.evaluate(x_test, y_test, verbose=1)}')
            # print('----------------------------')

            x_train_predictions = model.predict(x_train)
            x_val_predictions = model.predict(x_val)
            x_test_predictions = model.predict(x_test)

            PREDICTIONS[folder][f'{key}-{preprocessing_algorithm_id}'] = {
                'x_train': flatten(x_train_predictions.tolist()),
                'x_val': flatten(x_val_predictions.tolist()),
                'x_test': flatten(x_test_predictions.tolist()),
            }

            save_to_file('./predictions_bert_backup.json', {**PREDICTIONS, **STATICS})

        elif not is_classifier:
            keras_tokenizer = Tokenizer()

            (x_train, x_val, x_test), input_dim, input_len = Helpers.get_model_inputs(
                (train_inputs, val_inputs, test_data_preprocessed),
                keras_tokenizer,
            )

            model = load_model(model_path)

            print(key)
            print(model_path)
            # print(f'a - {model.evaluate(x_train, y_train, verbose=0)}')
            # print(f'va - {model.evaluate(x_val, y_val, verbose=0)}')
            # print(f'ta - {model.evaluate(x_test, y_test, verbose=0)}')
            print('----------------------------')

            x_train_predictions = model.predict(x_train)
            x_val_predictions = model.predict(x_val)
            x_test_predictions = model.predict(x_test)

            PREDICTIONS[folder][f'{key}-{preprocessing_algorithm_id}'] = {
                'x_train': flatten(x_train_predictions.tolist()),
                'x_val': flatten(x_val_predictions.tolist()),
                'x_test': flatten(x_test_predictions.tolist()),
            }
        else:
            vectorizer = Sklearn.VECTORIZERS[data[2]](**{
                'binary': True,
                'ngram_range': (int(data[3]), int(data[4]))
            })

            x_train = vectorizer.fit_transform(train_inputs).todense()
            x_val = vectorizer.transform(val_inputs).todense()
            x_test = vectorizer.transform(test_data_preprocessed).todense()

            cls = load_classifier(model_path)

            print(key)
            # print(f'a - {cls.score(x_train, y_train)}')
            # print(f'va - {cls.score(x_val, y_val)}')
            # print(f'ta - {cls.score(x_test, y_test)}')
            # print('----------------------------')

            PREDICTIONS[folder][f'{key}-{preprocessing_algorithm_id}'] = {
                'x_train': cls.predict(x_train).tolist(),
                'x_val':  cls.predict(x_val).tolist(),
                'x_test':  cls.predict(x_test).tolist(),
            }

save_to_file('./predictions_v_5.json', {**PREDICTIONS, **STATICS})
