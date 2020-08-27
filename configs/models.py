"""
Example of Model Config
MODEL = {
    'ACTIVATION': 'sigmoid',
    'OPTIMIZER': 'adam',
    'LEARNING_RATE': 1e-4,
    'BATCH_SIZE': 16,
    'EPOCHS': 12,
    'LSTM_UNITS': 128,
    'DROPOUT': 0.2,
    'DENSE_UNITS': 128,
    'CONV_FILTERS': 128,
    'CONV_KERNEL_SIZE': 5,
    'MAX_POOLING_POOL_SIZE': 4,
    'GRU_UNITS': 128,
    'EMBEDDING_OPTIONS': {
        'input_dim': 1000,
        'output_dim': 256,
        'input_length': 100
    },
    'TYPE': 'CNN'
}
"""
GLOVE_CONFIGS = {
    'LSTM': {
        'TYPE': 'LSTM',
        'BATCH_SIZE': 32,
        'EPOCHS': 20,
        'OPTIMIZER': 'adam',
        'LEARNING_RATE': 1e-4,
        'EMBEDDING_OPTIONS': {
            'output_dim': 200,
        },
        'LSTM_UNITS': 100,
    },
    'LSTM_DROPOUT': {
        'TYPE': 'LSTM_DROPOUT',
        'BATCH_SIZE': 32,
        'EPOCHS': 50,
        'OPTIMIZER': 'adam',
        'LEARNING_RATE': 1e-4,
        'EMBEDDING_OPTIONS': {
            'output_dim': 200,
        },
        'LSTM_UNITS': 100,
        'DROPOUT': 0.2,
    },
    'BI_LSTM': {
        'TYPE': 'BI_LSTM',
        'BATCH_SIZE': 32,
        'EPOCHS': 20,
        'OPTIMIZER': 'adam',
        'LEARNING_RATE': 1e-4,
        'EMBEDDING_OPTIONS': {
            'output_dim': 200,
        },
        'LSTM_UNITS': 100,
        'DROPOUT': 0.2,
    },
    'LSTM_CNN': {
        'TYPE': 'LSTM_CNN',
        'BATCH_SIZE': 32,
        'EPOCHS': 7,
        'OPTIMIZER': 'adam',
        'LEARNING_RATE': 1e-4,
        'EMBEDDING_OPTIONS': {
            'output_dim': 200,
        },
        'CONV_FILTERS': 100,
        'CONV_KERNEL_SIZE': 5,
        'LSTM_UNITS': 100,
    },
    'FASTTEXT': {
        'TYPE': 'FASTTEXT',
        'BATCH_SIZE': 8,
        'EPOCHS': 150,
        'OPTIMIZER': 'adam',
        'LEARNING_RATE': 1e-4,
        'EMBEDDING_OPTIONS': {
            'output_dim': 200,
        },
        'DENSE_UNITS': 200,
    },
    'RCNN': {
        'TYPE': 'RCNN',
        'BATCH_SIZE': 32,
        'EPOCHS': 15,
        'OPTIMIZER': 'adam',
        'LEARNING_RATE': 1e-4,
        'EMBEDDING_OPTIONS': {
            'output_dim': 200,
        },
        'CONV_FILTERS': 100,
        'CONV_KERNEL_SIZE': 5,
        'MAX_POOLING_POOL_SIZE': 4,
        'LSTM_UNITS': 100,
        'DROPOUT': 0.2,
    },
    'CNN': {
        'TYPE': 'CNN',
        'BATCH_SIZE': 32,
        'EPOCHS': 20,
        'OPTIMIZER': 'adam',
        'LEARNING_RATE': 1e-4,
        'EMBEDDING_OPTIONS': {
            'output_dim': 200,
        },
        'CONV_FILTERS': 100,
        'CONV_KERNEL_SIZE': 5,
        'DENSE_UNITS': 100,
        'DROPOUT': 0.2,
    },
    'RNN': {
        'TYPE': 'RNN',
        'BATCH_SIZE': 32,
        'EPOCHS': 20,
        'OPTIMIZER': 'adam',
        'LEARNING_RATE': 1e-4,
        'EMBEDDING_OPTIONS': {
            'output_dim': 200,
        },
        'LSTM_UNITS': 100,
        'DENSE_UNITS': 100,
    },
    'GRU': {
        'TYPE': 'GRU',
        'BATCH_SIZE': 32,
        'EPOCHS': 20,
        'OPTIMIZER': 'adam',
        'LEARNING_RATE': 1e-4,
        'EMBEDDING_OPTIONS': {
            'output_dim': 200,
        },
        'GRU_UNITS': 100,
        'DENSE_UNITS': 100,
        'DROPOUT': 0.2,
    },
}
CONFIGS = {
    'LSTM': {
        'TYPE': 'LSTM',
        'BATCH_SIZE': 128,
        'EPOCHS': 10,
        'OPTIMIZER': 'adam',
        'LEARNING_RATE': 1e-4,
        'EMBEDDING_OPTIONS': {
            'output_dim': 256,
        },
        'LSTM_UNITS': 128,
    },
    'LSTM_DROPOUT': {
        'TYPE': 'LSTM_DROPOUT',
        'BATCH_SIZE': 64,
        'EPOCHS': 15,
        'OPTIMIZER': 'adam',
        'LEARNING_RATE': 1e-4,
        'EMBEDDING_OPTIONS': {
            'output_dim': 256,
        },
        'LSTM_UNITS': 128,
        'DROPOUT': 0.2,
    },
    'BI_LSTM': {
        'TYPE': 'BI_LSTM',
        'BATCH_SIZE': 128,
        'EPOCHS': 10,
        'OPTIMIZER': 'adam',
        'LEARNING_RATE': 1e-4,
        'EMBEDDING_OPTIONS': {
            'output_dim': 256,
        },
        'LSTM_UNITS': 128,
        'DROPOUT': 0.2,
    },
    'LSTM_CNN': {
        'TYPE': 'LSTM_CNN',
        'BATCH_SIZE': 32,
        'EPOCHS': 7,
        'OPTIMIZER': 'adam',
        'LEARNING_RATE': 1e-4,
        'EMBEDDING_OPTIONS': {
            'output_dim': 256,
        },
        'CONV_FILTERS': 128,
        'CONV_KERNEL_SIZE': 5,
        'LSTM_UNITS': 128,
    },
    'FASTTEXT': {
        'TYPE': 'FASTTEXT',
        'BATCH_SIZE': 64,
        'EPOCHS': 30,
        'OPTIMIZER': 'adam',
        'LEARNING_RATE': 1e-4,
        'EMBEDDING_OPTIONS': {
            'output_dim': 256,
        },
        'DENSE_UNITS': 128,
    },
    'RCNN': {
        'TYPE': 'RCNN',
        'BATCH_SIZE': 128,
        'EPOCHS': 15,
        'OPTIMIZER': 'adam',
        'LEARNING_RATE': 1e-4,
        'EMBEDDING_OPTIONS': {
            'output_dim': 256,
        },
        'CONV_FILTERS': 128,
        'CONV_KERNEL_SIZE': 5,
        'MAX_POOLING_POOL_SIZE': 4,
        'LSTM_UNITS': 128,
        'DROPOUT': 0.2,
    },
    'CNN': {
        'TYPE': 'CNN',
        'BATCH_SIZE': 64,
        'EPOCHS': 15,
        'OPTIMIZER': 'adam',
        'LEARNING_RATE': 1e-4,
        'EMBEDDING_OPTIONS': {
            'output_dim': 256,
        },
        'CONV_FILTERS': 128,
        'CONV_KERNEL_SIZE': 5,
        'DENSE_UNITS': 128,
        'DROPOUT': 0.2,
    },
    'RNN': {
        'TYPE': 'RNN',
        'BATCH_SIZE': 64,
        'EPOCHS': 15,
        'OPTIMIZER': 'adam',
        'LEARNING_RATE': 1e-4,
        'EMBEDDING_OPTIONS': {
            'output_dim': 256,
        },
        'LSTM_UNITS': 128,
        'DENSE_UNITS': 128,
    },
    'GRU': {
        'TYPE': 'GRU',
        'BATCH_SIZE': 128,
        'EPOCHS': 15,
        'OPTIMIZER': 'adam',
        'LEARNING_RATE': 1e-4,
        'EMBEDDING_OPTIONS': {
            'output_dim': 256,
        },
        'GRU_UNITS': 128,
        'DENSE_UNITS': 128,
        'DROPOUT': 0.2,
    }
}


def get_model_config(model_type, glove=True):
    if glove:
        return GLOVE_CONFIGS[model_type].copy()
    else:
        return CONFIGS[model_type].copy()
