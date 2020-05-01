import json
from datetime import datetime
import os
import numpy as np
import pickle


def json_dumper(obj):
    try:
        return obj.toJSON()
    except:
        return json.dumps({'class_name': type(obj).__name__})


def save_to_file(file_path, data):
    with open(file_path, 'w', encoding='utf8') as file:
        file.write(json.dumps(data, ensure_ascii=False, default=json_dumper))


def get_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def ensure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_classifier(file_path, classifier):
    pickle.dump(classifier, open(file_path, 'wb'))


def log(
        target=None,
        data=None,
        model=None,
        model_history=None,
        model_config=None,
        classifier=None,
        vectorizer=None,
):
    file_path = f'./logs/{target}/log-{datetime.now().date()}.json'

    if not os.path.exists(file_path):
        save_to_file(file_path, {})

    log_data = get_from_file(file_path)

    current_log = {}

    if target is not None:
        current_log['target'] = target

    if data is not None:
        current_log['data'] = data

    if model is not None:
        current_log['model'] = model

        if model_config is not None:
            current_log['model']['config'] = model_config

        current_log['model']['history'] = {k: [round(float(v), 5) for v in data] for k, data in model_history.items()}

    if classifier is not None:
        current_log['classifier'] = classifier

    if vectorizer is not None:
        current_log['vectorizer'] = vectorizer

    log_data[str(datetime.now())] = current_log

    save_to_file(file_path, log_data)


def get_glove_embeddings(file_path):
    embeddings_dict = {}

    with open(file_path, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector.tolist()

    return embeddings_dict
