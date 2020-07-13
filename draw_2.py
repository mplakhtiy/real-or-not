from models.keras import Keras
from utils import get_from_file, save_to_file
import plotly.express as px
import pandas as pd


def get_average_history(kfold_history):
    k = len(kfold_history)
    epochs = len(kfold_history[0]['accuracy'])
    average_history = {
        "loss": [0] * epochs,
        "accuracy": [0] * epochs,
        "val_loss": [0] * epochs,
        "val_accuracy": [0] * epochs,
        "test_loss": [0] * epochs,
        "test_accuracy": [0] * epochs,
    }

    for history in kfold_history:
        for key in average_history:
            average_history[key] = [x + (y / k) for x, y in zip(average_history[key], history[key])]

    for key in average_history:
        average_history[key] = [round(v, 7) for v in average_history[key]]

    return average_history


def draw_kfold_history(log):
    line_types = ['loss', 'accuracy', 'val_loss', 'val_accuracy', 'test_loss', 'test_accuracy']

    df = {
        'Line Type': [],
        'Epoch': [],
        'Accuracy/Loss Value': [],
        'Fold': []
    }

    for k, fold_history in enumerate(log['KFOLD_HISTORY']):
        for line_type in line_types:
            for index, value in enumerate(fold_history[line_type]):
                df['Fold'].append(k)
                df['Line Type'].append(line_type)
                df['Epoch'].append(index)
                df['Accuracy/Loss Value'].append(value)

    fig = px.line(pd.DataFrame(df), x="Epoch", y="Accuracy/Loss Value", color="Line Type", line_group="Fold",
                  hover_name="Line Type")
    fig.update_layout(title=f'Model: {log["TYPE"]}, Preprocessing Algorithm: {log["PREPROCESSING_ALGORITHM_UUID"]}')
    fig.show()


def draw_history(history, title):
    line_types = ['loss', 'accuracy', 'val_loss', 'val_accuracy', 'test_loss', 'test_accuracy']

    df = {
        'Line Type': [],
        'Epoch': [],
        'Accuracy/Loss Value': [],
    }

    for line_type in line_types:
        for index, value in enumerate(history[line_type]):
            df['Line Type'].append(line_type)
            df['Epoch'].append(index)
            df['Accuracy/Loss Value'].append(value)

    fig = px.line(pd.DataFrame(df), x="Epoch", y="Accuracy/Loss Value", color="Line Type", line_group="Line Type",
                  hover_name="Line Type")
    if title is not None:
        fig.update_layout(title=title)
    fig.show()


def draw_comparison(history_dict, title):
    line_types = ['loss', 'accuracy', 'val_loss', 'val_accuracy', 'test_loss', 'test_accuracy']

    df = {
        'Preprocessing Algorithm': [],
        'Line Type': [],
        'Epoch': [],
        'Accuracy/Loss Value': [],
    }

    for algorithm, history in history_dict.items():
        for line_type in line_types:
            for index, value in enumerate(history[line_type]):
                df['Preprocessing Algorithm'].append(algorithm)
                df['Line Type'].append(line_type)
                df['Epoch'].append(index)
                df['Accuracy/Loss Value'].append(value)

    fig = px.line(pd.DataFrame(df), x="Epoch", y="Accuracy/Loss Value", color="Preprocessing Algorithm",
                  line_group="Line Type",
                  hover_name="Line Type")
    if title is not None:
        fig.update_layout(title=title)
    fig.show()


def get_highest_history(kfold_history):
    highest_history = {
        'fold': None,
        'epoch': None,
        'test_score': 0,
        'val_score': None,
    }

    for fold, fold_history in enumerate(kfold_history):
        for epoch, test_score in enumerate(fold_history['test_accuracy']):
            if test_score > highest_history['test_score']:
                highest_history['fold'] = fold
                highest_history['epoch'] = epoch
                highest_history['test_score'] = test_score
                highest_history['val_score'] = fold_history['val_accuracy'][epoch]

    return highest_history


# logs = get_from_file('./logs/keras/2020-05-12.json')
logs = get_from_file('./logs/keras-glove/2020-07-12.json')
result = {
    'model': [],
    'glove': [],
    'optimizer': [],
    'algorithm': [],
    'fold': [],
    'epoch': [],
    'val_score': [],
    'test_score': [],
}

for log_path in ['./logs/keras/2020-05-07.json','./logs/keras/2020-05-12.json', './logs/keras/2020-07-11.json', './logs/keras-glove/2020-07-06.json', './logs/keras-glove/2020-07-09.json',
                 './logs/keras-glove/2020-07-10.json', './logs/keras-glove/2020-07-11.json',
                 './logs/keras-glove/2020-07-12.json']:
    for log_id, log in get_from_file(log_path).items():
        result['model'].append(log["TYPE"])
        result['algorithm'].append(log["PREPROCESSING_ALGORITHM_UUID"][:8])
        if 'GLOVE' in log:
            result['glove'].append(True)
        else:
            result['glove'].append(False)
        result['optimizer'].append(log['OPTIMIZER'])
        h = get_highest_history(log['KFOLD_HISTORY'])
        result['fold'].append(h['fold'])
        result['epoch'].append(h['epoch'])
        result['val_score'].append(h['val_score'])
        result['test_score'].append(h['test_score'])




# MODELS_AVERAGES_HISTORY = {}
#
# for log_id, log in logs.items():
#     if not MODELS_AVERAGES_HISTORY.get(log["TYPE"]):
#         MODELS_AVERAGES_HISTORY[log["TYPE"]] = {}
#     MODELS_AVERAGES_HISTORY[log["TYPE"]][log['PREPROCESSING_ALGORITHM_UUID']] = get_average_history(
#         log['KFOLD_HISTORY'])

# save_to_file('averages.json', MODELS_AVERAGES_HISTORY)

# result = {
#     'Model': [],
#     'Algorithm ID': []
# }
#
# for model_type, data in MODELS_AVERAGES_HISTORY.items():
#     result = {
#         'Model': [],
#         'Algorithm ID': []
#     }
#
#     for algorithm_id, history in data.items():
#         result['Model'].append(model_type)
#         result['Algorithm ID'].append(algorithm_id[:8])
#
#         for key in history:
#             for i, v in enumerate(history[key]):
#                 try:
#                     result[f'{key}_{i + 1}'].append(v)
#                 except:
#                     result[f'{key}_{i + 1}'] = [v]
#
#     a = {}
#     for k, v in result.items():
#         a[k] = pd.Series(v)
#     a = pd.DataFrame(a)
#     a.to_csv(f'{model_type}-averages.csv', index=False)
# save_to_file('averages.json', DATA)

# for model_type, history in MODELS_AVERAGES_HISTORY.items():
#     draw_comparison(history, f'Model: {model_type}')

    a = {}
    for k, v in result.items():
        a[k] = pd.Series(v)
    a = pd.DataFrame(a)
    a.to_csv(f'highest.csv', index=False)