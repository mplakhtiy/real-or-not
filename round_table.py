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
from bert_tokenization import FullTokenizer
import tensorflow_hub as hub

saved_models_pathes = [str(lst) for lst in list(Path("./data-saved-models/").rglob("*.h5"))] + [str(lst) for lst in
                                                                                                list(Path(
                                                                                                    "./data-saved-models/").rglob(
                                                                                                    "*.pickle"))]
SEED = 7

NETWORKS_KEYS = ['LSTM', 'LSTM_DROPOUT', 'BI_LSTM', 'FASTTEXT', 'RCNN', 'CNN', 'RNN', 'GRU']
BERT_KEY = ['BERT']
BERT_MODEL = {
    'BERT': 'bert_en_uncased_L-24_H-1024_A-16',
    'BERT_VERSION': 1,
}
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

predictions = get_from_file('./predictions_v_4.json')

x_train = []
x_val = []
x_test = []

train_l = len(predictions['y_train'])
val_l = len(predictions['y_val'])
test_l = len(predictions['y_test'])


def fill(lst, lngth, prdctns, key):
    for i in range(lngth):
        lst.append([])
        for folder in ['bert', 'glove-true']:
            is_classifier = folder == 'classifiers'
            is_bert = folder == 'bert'

            keys = None

            if is_classifier:
                keys = CLASSIFIERS_KEYS
            elif is_bert:
                keys = BERT_KEY
            else:
                keys = NETWORKS_KEYS

            for n_key in keys:
                try:
                    value = prdctns[folder][n_key][key][i]
                    if value > 0.75:
                        lst[i].append(1)
                    elif value > 0.25:
                        lst[i].append(round(value, 6))
                    else:
                        lst[i].append(0)
                except:
                    pass


fill(x_train, train_l, predictions, 'x_train')
fill(x_val, val_l, predictions, 'x_val')
fill(x_test, test_l, predictions, 'x_test')

y_train = predictions['y_train']
y_val = predictions['y_val']
y_test = predictions['y_test']

x_train, x_val, y_train, y_val = train_test_split(
    x_val,
    y_val,
    test_size=0.3,
    random_state=42
)

config = {
    'BATCH_SIZE': 32,
    'EPOCHS': 1000,
    'OPTIMIZER': 'adam',
    'LEARNING_RATE': 1e-4,
    'INPUT_DIM': 9,
    'HIDDEN_LAYER': 18,
    'HIDDEN_LAYER_1': 9
}
model = Keras.get_sequential_model(
    [
        Dense(config['INPUT_DIM'], input_dim=config['INPUT_DIM']),
        Dense(config['HIDDEN_LAYER']),
        Dense(config['HIDDEN_LAYER_1']),
    ],
    config
)

history = Keras.fit(model, (x_train, y_train, x_val, y_val, x_test, y_test), config)

save_to_file('./round_table_history.json', history)
