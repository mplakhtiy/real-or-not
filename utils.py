import json
import matplotlib.pyplot as plt
from datetime import datetime
import os


def plot(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()


def draw_keras_graph(history):
    plot(history, "accuracy")
    plot(history, 'loss')


def save_to_file(file_path, data):
    with open(file_path, 'w', encoding='utf8') as file:
        file.write(json.dumps(data, ensure_ascii=False))


def get_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def get_prepared_data_from_file(file_path):
    data = get_from_file(file_path)

    return data.get('x_train'), data.get('y_train'), data.get('x_val'), data.get('y_val'), data


def log(
        preprocess_options=None,
        model_history=None,
        model_config=None,
        words_reputation_filter=None,
        train_percentage=None,
        batch_size=None,
        epochs=None,
        embedding_dim=None,
        vocabulary_len=None,
        add_start_symbol=None,
        input_len=None,
        optimizer=None,
        tweets_for_vocabulary_base=None,
        shuffle_data=None,
        file_path=f'./logs/log-{datetime.now().date()}.json'
):
    if not os.path.exists(file_path):
        save_to_file(file_path, {})

    log_data = get_from_file(file_path)

    log_data[str(datetime.now())] = {
        'data': {
            'preprocess_options': preprocess_options,
            'words_reputation_filter': words_reputation_filter,
            'add_start_symbol': add_start_symbol,
            'train_percentage': train_percentage,
            'vocabulary_len': vocabulary_len,
            'tweets_for_vocabulary_base': tweets_for_vocabulary_base,
            'shuffle_data': shuffle_data
        },
        'model': {
            'config': model_config,
            'val_accuracy': [float(a) for a in model_history['val_accuracy']],
            'epochs': epochs,
            'embedding_dim': embedding_dim,
            'input_len': input_len,
            'batch_size': batch_size,
            'optimizer': optimizer,
        },
    }

    save_to_file(file_path, log_data)
