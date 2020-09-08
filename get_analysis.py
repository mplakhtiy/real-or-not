from utils import get_from_file, save_to_file
from pathlib import Path
from data import train_data, test_data
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd

folder = '7000'

logs = [str(lst) for lst in list(Path(f"./logs/bert/{folder}").rglob("*.json"))] + \
       [str(lst) for lst in list(Path(f"./logs/keras/{folder}").rglob("*.json"))] + \
       [str(lst) for lst in list(Path(f"./logs/keras-glove/{folder}").rglob("*.json"))] + \
       [str(lst) for lst in list(Path(f"./logs/han/{folder}").rglob("*.json"))] + \
       [str(lst) for lst in list(Path(f"./logs/han-glove/{folder}").rglob("*.json"))] + \
       [str(lst) for lst in list(Path(f"./logs/sklearn/{folder}").rglob("*.json"))]


def get_best_epoch(history):
    best_loss_index = history['val_loss'].index(min(history['val_loss']))
    temp_acc = 0
    temp_loss = 99999
    best_acc_index = 0

    for i in range(best_loss_index + 1):
        if temp_acc <= history['val_accuracy'][i] and history['val_loss'][i] <= temp_loss:
            temp_acc = history['val_accuracy'][i]
            temp_loss = history['val_loss'][i]
            best_acc_index = i

    return best_acc_index, best_loss_index


new_log = {}

for log_path in logs:
    log = list(get_from_file(log_path).values())[0]
    kfold_history = []

    if '/sklearn' in log_path:
        new_log[log_path] = log

    else:
        for fold_history in log['KFOLD_HISTORY']:
            best_acc_epoch, best_loss_epoch = get_best_epoch(fold_history)
            new_fold = {
                "loss": fold_history['loss'],
                "accuracy": fold_history['accuracy'],
                "val_loss": fold_history['val_loss'],
                "val_accuracy": fold_history['val_accuracy'],
                "test_loss": fold_history['test_loss'],
                "test_accuracy": fold_history['test_accuracy'],
                "best_acc_epoch": best_acc_epoch,
                "best_loss_epoch": best_loss_epoch,
                "val_predictions": fold_history['val_predictions'][best_acc_epoch],
                "test_predictions": fold_history['test_predictions'][best_acc_epoch]
            }

            kfold_history.append(new_fold)

            log_copy = log.copy()
            del log_copy['KFOLD_HISTORY']
            log_copy['KFOLD_HISTORY'] = kfold_history

            new_log[log_path] = log_copy

save_to_file(f'./{folder}-10fcv-27m.json', new_log)

SEED = 7
KFOLD = 10

kfold = StratifiedKFold(n_splits=KFOLD, shuffle=True, random_state=SEED)

# inputs = np.concatenate([train_data['text'], test_data.text])
# targets = np.concatenate([train_data['target'], test_data.target])
# inputs = train_data['text']
# targets = train_data['target']
inputs = test_data['text']
targets = test_data['target']

result_csv = {
    'index': [],
    'fold': [],
    'text': [],
    'target': [],
    'BERT-2e359f0b': [],
    'LSTM-GloVe-8b7db91c': [],
    'LSTM_DROPOUT-GloVe-8b7db91c': [],
    'BI_LSTM-GloVe-8b7db91c': [],
    'LSTM_CNN-GloVe-8b7db91c': [],
    'FASTTEXT-GloVe-8b7db91c': [],
    'RCNN-GloVe-71bd09db': [],
    'CNN-GloVe-d3cc3c6e': [],
    'RNN-GloVe-7bc816a1': [],
    'GRU-GloVe-8b7db91c': [],
    'HAN-GloVe-a85c8435': [],
    'LSTM-71bd09db': [],
    'LSTM_DROPOUT-d3cc3c6e': [],
    'BI_LSTM-8b7db91c': [],
    'LSTM_CNN-8b7db91c': [],
    'FASTTEXT-b054e509': [],
    'RCNN-7bc816a1': [],
    'CNN-b054e509': [],
    'RNN-1258a9d2': [],
    'GRU-b054e509': [],
    'HAN-a85c8435': [],
    'RIDGE-2e359f0b': [],
    'SVC-4c2e484d': [],
    'LOGISTIC_REGRESSION-8b7db91c': [],
    'SGD-2e359f0b': [],
    'DECISION_TREE-7bc816a1': [],
    'RANDOM_FOREST-60314ef9': []
}

for key in new_log:
    if '/sklearn' in key:
        csv_key = f'{new_log[key]["CLASSIFIER"]["TYPE"]}-{new_log[key]["PREPROCESSING_ALGORITHM_UUID"]}'
    else:
        if 'GLOVE' in new_log[key]:
            csv_key = f'{new_log[key]["TYPE"]}-GloVe-{new_log[key]["PREPROCESSING_ALGORITHM_UUID"]}'
        else:
            csv_key = f'{new_log[key]["TYPE"]}-{new_log[key]["PREPROCESSING_ALGORITHM_UUID"]}'
    for fold in new_log[key]['KFOLD_HISTORY']:
        result_csv[csv_key] = result_csv[csv_key] + fold['test_predictions']
        # result_csv[csv_key].append('*')

# for train, validation in kfold.split(inputs, targets):
#     result_csv['index'] = result_csv['index'] + validation.tolist()
#     result_csv['text'] = result_csv['text'] + inputs[validation].tolist()
#     result_csv['target'] = result_csv['target'] + targets[validation].tolist()
#     # result_csv['index'].append('*')
#     # result_csv['text'].append('*')
#     # result_csv['target'].append('*')

for i in range(10):
    result_csv['index'] = result_csv['index'] + list(range(test_data.text.size))
    result_csv['fold'] = result_csv['fold'] + [i] * test_data.text.size
    result_csv['text'] = result_csv['text'] + test_data.text.tolist()
    result_csv['target'] = result_csv['target'] + test_data.target.tolist()

csv_file = {}
for k, v in result_csv.items():
    csv_file[k] = pd.Series(v)
a = pd.DataFrame(csv_file)
a.to_csv(f'./{folder}-10fcv-27m-TDS.csv', index=True)
