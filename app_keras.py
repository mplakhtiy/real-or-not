# -*- coding: utf-8 -*-
import numpy as np
from models import KerasModels
from utils import draw_keras_graph, get_prepared_data_from_file, get_from_file

PREPARED_DATA_FILE_PATH = './data/prepared_data/words_indexes_preprocess_all_true.json'

x_train, y_train, x_val, y_val, data = get_prepared_data_from_file(PREPARED_DATA_FILE_PATH)

BATCH_SIZE = 256
EPOCHS = 15
VERBOSE = 1
EMBEDING_DIM = 256
LSTM_UNITS = 128
INPUT_LENGTH = data.get('max_vector_len')
EMBEDDING_OPTIONS = {
    'input_dim': data.get('vocabulary_len'),
    'output_dim': EMBEDING_DIM,
    'input_length': INPUT_LENGTH
}

model = KerasModels.get_binary_classification_model(EMBEDDING_OPTIONS)
history = model.fit(
    x=np.array(x_train),
    y=np.array(y_train),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=VERBOSE,
    shuffle=True,
    validation_data=(
        np.array(x_val),
        np.array(y_val)
    )
)

# model = KerasModels.get_lstm_model(embedding_options, lstm_units)
# history = model.fit(
#     x=np.array(x_train),
#     y=np.array(y_train),
#     batch_size=batch_size,
#     epochs=epochs,
#     verbose=verbose,
#     shuffle=True,
#     validation_data=(
#         np.array(x_val),
#         np.array(y_val)
#     )
# )

# model = KerasModels.get_mlp_for_binary_classification_model(embedding_options, [64])
# history = model.fit(
#     x=np.array(x_train),
#     y=np.array(y_train),
#     batch_size=batch_size,
#     epochs=epochs,
#     verbose=verbose,
#     shuffle=True,
#     validation_data=(
#         np.array(x_val),
#         np.array(y_val)
#     )
# )

draw_keras_graph(history)
