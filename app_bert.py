# -*- coding: utf-8 -*-
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow_hub as hub
from bert_tokenization import FullTokenizer
from data import train, validation, test
from tweets import Helpers
from models import Keras, TestDataCallback
from utils import log
import os

MODEL = {
    # 'BERT': 'bert_en_uncased_L-12_H-768_A-12',
    'BERT': 'bert_en_uncased_L-24_H-1024_A-16',
    'BERT_VERSION': 2,
    'BATCH_SIZE': 16,
    'EPOCHS': 3,
    'VERBOSE': 1,
    'OPTIMIZER': 'adam',
    'LEARNING_RATE': 2e-6,
    'SHUFFLE': True,
    'VALIDATION_PERCENTAGE': 0.2
}

bert_layer = hub.KerasLayer(f'https://tfhub.dev/tensorflow/{MODEL["BERT"]}/{MODEL["BERT_VERSION"]}', trainable=True)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = FullTokenizer(vocab_file, do_lower_case)

x_train, INPUT_LENGTH = Helpers.get_bert_input(train.text.values, tokenizer)
x_val = Helpers.get_bert_input(validation.text.values, tokenizer, input_length=INPUT_LENGTH)
x_test = Helpers.get_bert_input(test.text.values, tokenizer, input_length=INPUT_LENGTH)

y_train = train.target.values
y_val = validation.target.values
y_test = test.target.values

MODEL['INPUT_LENGTH'] = INPUT_LENGTH

MODEL_SAVE_PATH = f'./data/models/bert/{MODEL["BERT"]}/{MODEL["BATCH_SIZE"]}-{MODEL["LEARNING_RATE"]}-{INPUT_LENGTH}'

if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)

test_data_callback = TestDataCallback(
    x_test=x_test,
    y_test=y_test
)

checkpoint = ModelCheckpoint(
    MODEL_SAVE_PATH + '/model-{epoch:03d}-{accuracy:03f}-{val_accuracy:03f}.h5',
    verbose=MODEL['VERBOSE'],
    monitor='val_loss',
    # save_best_only=True,
    mode='auto'
)

model = Keras.get_bert_model(
    bert_layer=bert_layer,
    input_length=INPUT_LENGTH,
    optimizer=MODEL['OPTIMIZER'],
    learning_rate=MODEL['LEARNING_RATE']
)

model.summary()

history = model.fit(
    x_train, y_train,
    epochs=MODEL['EPOCHS'],
    batch_size=MODEL['BATCH_SIZE'],
    verbose=MODEL['VERBOSE'],
    validation_data=(
        x_val,
        y_val
    ),
    callbacks=[checkpoint, test_data_callback]
)

model_history = history.history.copy()
model_history['test_loss'] = test_data_callback.loss
model_history['test_accuracy'] = test_data_callback.accuracy

log(
    target='bert',
    model=MODEL,
    model_history=model_history,
    model_config=model.get_config(),
)

os.system('spd-say "Experiment is finished"')

Keras.draw_graph(model_history)
