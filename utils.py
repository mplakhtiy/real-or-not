import json
import matplotlib.pyplot as plt


def plot(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()


def save_to_file(file_path, data):
    with open(file_path, 'w', encoding='utf8') as file:
        file.write(json.dumps(data, ensure_ascii=False))


def draw_keras_graph(history):
    plot(history, "accuracy")
    plot(history, 'loss')


def get_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def get_prepared_data_from_file(file_path):
    data = get_from_file(file_path)

    return data.get('x_train'), data.get('y_train'), data.get('x_val'), data.get('y_val'), data
