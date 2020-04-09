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
        data=None,
        vocabulary=None,
        model=None,
        model_history=None,
        model_config=None,
        test_performance=None,
        file_path=f'./logs/log-{datetime.now().date()}.json'
):
    if not os.path.exists(file_path):
        save_to_file(file_path, {})

    log_data = get_from_file(file_path)

    current_log = {
        'data': data,
        'model': model,
        'vocabulary': vocabulary,
    }

    current_log['model']['config'] = model_config
    current_log['model']['val_accuracy'] = [float(a) for a in model_history['val_accuracy']]
    current_log['model']['test_performance'] = test_performance

    log_data[str(datetime.now())] = current_log

    save_to_file(file_path, log_data)
