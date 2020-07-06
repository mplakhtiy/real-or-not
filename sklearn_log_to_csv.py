from os import listdir
from os.path import isfile, join
from utils import get_from_file, save_to_file
import pandas as pd

dir_path = './logs/sklearn/'

logs = [join(dir_path, f) for f in listdir(dir_path) if isfile(join(dir_path, f))]

DATA = {

}

K = 10

result = {
    'Train Id': [],
    'Classifier': [],
    'Vectorizer': [],
    'Algorithm ID': [],
    'Train Score': [],
    'Validation Score': [],
    'Test Score': [],
}

for log_path in logs:
    log = get_from_file(log_path)

    for log_value in log.values():
        item = {}
        accumulated = {
            "train_score": 0,
            "val_score": 0,
            "test_score": 0
        }

        for h in log_value['KFOLD_HISTORY']:
            accumulated['train_score'] += h['train_score'] / K
            accumulated['val_score'] += h['val_score'] / K
            accumulated['test_score'] += h['test_score'] / K

        item['CLASSIFIER'] = log_value['CLASSIFIER']
        item['VECTORIZER'] = log_value['VECTORIZER']
        item['SCORE'] = accumulated

        try:
            DATA[log_value['PREPROCESSING_ALGORITHM_UUID']].append(item)
        except:
            DATA[log_value['PREPROCESSING_ALGORITHM_UUID']] = [item]

        result['Train Id'].append(log_value['UUID'][:8])
        result['Classifier'].append(log_value['CLASSIFIER']['TYPE'])
        result['Vectorizer'].append(
            log_value['VECTORIZER']['TYPE'] + str(log_value['VECTORIZER']['OPTIONS'].get('ngram_range', [1, 1])))
        result['Algorithm ID'].append(log_value['PREPROCESSING_ALGORITHM_UUID'][:8])
        result['Train Score'].append(accumulated['train_score'])
        result['Validation Score'].append(accumulated['val_score'])
        result['Test Score'].append(accumulated['test_score'])

a = {}
for k, v in result.items():
    a[k] = pd.Series(v)
a = pd.DataFrame(a)
a.to_csv('sklearn_res.csv', index=False)
save_to_file('averages.json', DATA)
