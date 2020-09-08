from utils import get_from_file, save_to_file
from pathlib import Path
from data import train_data, test_data
from sklearn.model_selection import StratifiedKFold
import numpy as np

folder = '10000'

logs = [str(lst) for lst in list(Path(f"./logs/bert/{folder}").rglob("*.json"))] + \
       [str(lst) for lst in list(Path(f"./logs/keras/{folder}").rglob("*.json"))] + \
       [str(lst) for lst in list(Path(f"./logs/keras-glove/{folder}").rglob("*.json"))] + \
       [str(lst) for lst in list(Path(f"./logs/han/{folder}").rglob("*.json"))] + \
       [str(lst) for lst in list(Path(f"./logs/han-glove/{folder}").rglob("*.json"))] + \
       [str(lst) for lst in list(Path(f"./logs/sklearn/{folder}").rglob("*.json"))]


def get_best_epoch(history):
    best_loss_index = history['val_loss'].index(min(history['val_loss']))
    temp_acc = 0
    temp_loss = 0
    best_acc_index = 0

    for i in range(best_loss_index + 1):
        if temp_acc <= history['val_accuracy'][i] and temp_loss <= history['val_loss'][i]:
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
                "best_acc_epoch": best_acc_epoch,
                "best_loss_epoch": best_loss_epoch,
                "val_predictions": fold_history['val_predictions'][best_acc_epoch]
            }

            kfold_history.append(new_fold)

            log_copy = log.copy()
            del log_copy['KFOLD_HISTORY']
            log_copy['KFOLD_HISTORY'] = kfold_history

            new_log[log_path] = log_copy

save_to_file(f'./{folder}-analysis.json', new_log)

# SEED = 7
# KFOLD = 10
#
# kfold = StratifiedKFold(n_splits=KFOLD, shuffle=True, random_state=SEED)
#
# inputs = np.concatenate([train_data['text'], test_data.text])
# targets = np.concatenate([train_data['target'], test_data.target])
#
#
#
# for train, validation in kfold.split(inputs, targets):
#     x_train = inputs[train]
#     y_train = targets[train]
#     x_val = inputs[validation]
#     y_val = targets[validation]
