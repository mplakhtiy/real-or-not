# -*- coding: utf-8 -*-
# Implementation of Hierarchical Attentional Networks for Document Classification
# http://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf
# https://github.com/arunarn2/HierarchicalAttentionNetworks
# requirements: python3.6, tensorflow 1.4.0, keras 2.0.8 !!!

from keras.callbacks import Callback, ModelCheckpoint
from keras import backend as K
from keras.models import Model
from keras import initializers
from keras.engine.topology import Layer
from keras.optimizers import Adam, RMSprop
from keras.layers import Dense, Input
from keras.layers import Embedding, GRU, Bidirectional, TimeDistributed
from utils import get_glove_embeddings, log_model, ensure_path_exists
from data import train_data, test_data
from tweets import Helpers, tweets_preprocessor
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from configs import get_preprocessing_algorithm
import numpy as np
import uuid

PREDICTIONS = {
    'x_train': [],
    'x_val': [],
    'x_test': [],
}


class TestDataCallback(Callback):
    def __init__(self, x_test, y_test, x_train, x_val):
        super().__init__()
        self.accuracy = []
        self.loss = []
        self.x_train = x_train
        self.x_val = x_val
        self.x_test = x_test
        self.y_test = y_test

    def on_epoch_end(self, epoch, logs=None):
        score = self.model.evaluate(self.x_test, self.y_test, verbose=1)
        self.loss.append(score[0])
        PREDICTIONS['x_train'].append([v[0] for v in self.model.predict(self.x_train).tolist()])
        PREDICTIONS['x_val'].append([v[0] for v in self.model.predict(self.x_val).tolist()])
        PREDICTIONS['x_test'].append([v[0] for v in self.model.predict(self.x_test).tolist()])
        self.accuracy.append(score[1])


# class defining the custom attention layer
class HierarchicalAttentionNetwork(Layer):
    def __init__(self, attention_dim):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(HierarchicalAttentionNetwork, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim,)))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(HierarchicalAttentionNetwork, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))

        ait = K.exp(K.squeeze(K.dot(uit, self.u), -1))

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        weighted_input = x * K.expand_dims(ait)
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


OPTIMIZERS = {
    'adam': Adam,
    'rmsprop': RMSprop
}

DEFAULTS = {
    'ACTIVATION': 'sigmoid',
    'OPTIMIZER': 'adam',
    'LEARNING_RATE': 1e-4,
}


def get_han_model(config):
    embedding_layer = Embedding(**config['EMBEDDING_OPTIONS'], mask_zero=True)

    sentence_input = Input(shape=(config['EMBEDDING_OPTIONS']['input_length'],), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    lstm_word = Bidirectional(GRU(config['GRU_UNITS'], return_sequences=True))(embedded_sequences)
    attn_word = HierarchicalAttentionNetwork(config['ATTN_UNITS'])(lstm_word)
    sentenceEncoder = Model(sentence_input, attn_word)

    review_input = Input(shape=(1, config['EMBEDDING_OPTIONS']['input_length']), dtype='int32')
    review_encoder = TimeDistributed(sentenceEncoder)(review_input)
    lstm_sentence = Bidirectional(GRU(config['GRU_UNITS'], return_sequences=True))(review_encoder)
    attn_sentence = HierarchicalAttentionNetwork(config['ATTN_UNITS'])(lstm_sentence)
    preds = Dense(
        1,
        activation=config.get('ACTIVATION', DEFAULTS['ACTIVATION'])
    )(attn_sentence)

    model = Model(review_input, preds)

    model.compile(
        optimizer=OPTIMIZERS[
            config.get('OPTIMIZER', DEFAULTS['OPTIMIZER'])
        ](
            lr=config.get(
                'LEARNING_RATE',
                DEFAULTS['LEARNING_RATE']
            )
        ),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    # print("model fitting - Hierachical attention network")
    # model.fit(x_train, y_train, validation_data=(x_val, y_val), nb_epoch=10, batch_size=100)


def fit(model, data, config):
    is_with_test_data = len(data) == 6

    if is_with_test_data:
        x_train, y_train, x_val, y_val, x_test, y_test = data
    else:
        x_train, y_train, x_val, y_val = data

    callbacks = []

    if is_with_test_data:
        test_data_callback = TestDataCallback(
            x_test=x_test,
            y_test=y_test,
            x_train=x_train,
            x_val=x_val
        )
        callbacks.append(test_data_callback)

    if config.get('DIR') is not None and config.get('PREFIX') is not None:
        suffix = '-e{epoch:03d}-a{acc:03f}-va{val_acc:03f}-ta.hdf5'
        callbacks.append(ModelCheckpoint(
            config['DIR'] + config['PREFIX'] + suffix,
            verbose=1,
            monitor='val_loss',
            save_best_only=True,
            mode='auto'
        ))

    history = model.fit(
        x=x_train, y=y_train,
        batch_size=config['BATCH_SIZE'],
        epochs=config['EPOCHS'],
        verbose=1,
        validation_data=(
            x_val,
            y_val
        ),
        callbacks=callbacks
    )

    model_history = history.history.copy()
    model_history['test_loss'] = test_data_callback.loss
    model_history['test_accuracy'] = test_data_callback.accuracy
    model_history = {
        k: [round(float(v), 6) for v in data] for k, data in model_history.items()
    }

    return model_history


HAN_CONFIG = {
    'TYPE': 'HAN',
    'BATCH_SIZE': 32,
    'EPOCHS': 5,
    'OPTIMIZER': 'adam',
    'LEARNING_RATE': 1e-4,
    'EMBEDDING_OPTIONS': {
        'output_dim': 256,
    },
    'GRU_UNITS': 128,
    'ATTN_UNITS': 128,
}

TRAIN_UUID = str(uuid.uuid4())

SEED = 7

USE_GLOVE = False

NETWORK_KEY = 'HAN'
PREPROCESSING_ALGORITHM_ID = 'a85c8435'

PREPROCESSING_ALGORITHM = get_preprocessing_algorithm(PREPROCESSING_ALGORITHM_ID, join=True)

CONFIG = HAN_CONFIG.copy()

if USE_GLOVE:
    CONFIG['GLOVE'] = {
        'SIZE': 200
    }
    GLOVE = f'glove.twitter.27B.{CONFIG["GLOVE"]["SIZE"]}d.txt'
    GLOVE_FILE_PATH = f'./data/glove/{GLOVE}'
    GLOVE_EMBEDDINGS = get_glove_embeddings(GLOVE_FILE_PATH)

CONFIG['TRAIN_UUID'] = TRAIN_UUID
CONFIG['UUID'] = str(uuid.uuid4())
CONFIG['PREPROCESSING_ALGORITHM'] = PREPROCESSING_ALGORITHM
CONFIG['PREPROCESSING_ALGORITHM_UUID'] = PREPROCESSING_ALGORITHM_ID
CONFIG['HISTORY'] = None
# CONFIG['DIR'] = f'./data-saved-models/glove-true/{NETWORK_KEY}/'
# ensure_path_exists(CONFIG['DIR'])
# CONFIG['PREFIX'] = f'{NETWORK_KEY}-{PREPROCESSING_ALGORITHM_ID}-SEED-{SEED}'

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

train_inputs, val_inputs, train_targets, val_targets = train_test_split(
    train_data['preprocessed'],
    train_data['target'],
    test_size=0.3,
    random_state=SEED
)

keras_tokenizer = Tokenizer()

(x_train, x_val, x_test), input_dim, input_len = Helpers.get_model_inputs(
    (train_inputs, val_inputs, test_data.preprocessed),
    keras_tokenizer
)
y_train = train_targets
y_val = val_targets
y_test = test_data.target.values

x_train = np.array([[v] for v in x_train])
x_val = np.array([[v] for v in x_val])
x_test = np.array([[v] for v in x_test])

CONFIG['EMBEDDING_OPTIONS']['input_dim'] = input_dim
CONFIG['EMBEDDING_OPTIONS']['input_length'] = input_len

if USE_GLOVE:
    Helpers.with_glove_embedding_options(CONFIG, keras_tokenizer, GLOVE_EMBEDDINGS)

model = get_han_model(CONFIG)

history = fit(model, (x_train, y_train, x_val, y_val, x_test, y_test), CONFIG)

CONFIG['HISTORY'] = history
loss_index = history['val_loss'].index(min(history['val_loss']))

CONFIG['PREDICTIONS'] = {
    'x_train': PREDICTIONS['x_train'][loss_index],
    'x_val': PREDICTIONS['x_val'][loss_index],
    'x_test': PREDICTIONS['x_test'][loss_index],
}

log_model(CONFIG)
print('VAL_ACC: {} ; TEST ACC: {}'.format(history['val_acc'][loss_index], history['test_accuracy'][loss_index]))
