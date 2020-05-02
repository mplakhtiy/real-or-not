import os
import pandas as pd
from utils import get_from_file

from os import listdir
from os.path import isfile, join

LOGS_DIR = './logs/sklearn/'

logs_files = [f for f in listdir(LOGS_DIR) if isfile(join(LOGS_DIR, f))]

VAL_SIZE = 1523
TEST_SIZE = 3263
TOTAL_SIZE = VAL_SIZE + TEST_SIZE
MAPPING = {
    "add_link_flag": 'Link Flag',
    "add_user_flag": 'User Flag',
    "add_hash_flag": 'Hash Flag',
    "add_number_flag": 'Numb Flag',
    "add_keyword_flag": 'Keyw Flag',
    "add_location_flag": 'Lctn Flag',
    "remove_links": 'R Links',
    "remove_users": 'R Users',
    "remove_hash": 'R Hash',
    "unslang": 'Unslang',
    "split_words": 'Split W',
    "stem": 'Stem',
    "remove_punctuations": 'R Punct',
    "remove_numbers": 'R Numb',
    "to_lower_case": 'Lower',
    "remove_stop_words": 'R StpWrds',
    "remove_not_alpha": 'R NotAlpha',
    "join": 'Join'
}

result = {
    'Classifier': [""],
    'Vectorizer': [""],
    'Validation Acc': [VAL_SIZE],
    'Test Acc': [TEST_SIZE],
    'Total': [TOTAL_SIZE],
    'Total Acc': [TOTAL_SIZE],
    'Link Flag': [""],
    'User Flag': [""],
    'Hash Flag': [""],
    'Numb Flag': [""],
    'Keyw Flag': [""],
    'Lctn Flag': [""],
    'R Links': [""],
    'R Users': [""],
    'R Hash': [""],
    'Unslang': [""],
    'Split W': [""],
    'Stem': [""],
    'R Punct': [""],
    'R Numb': [""],
    'Lower': [""],
    'R StpWrds': [""],
    'R NotAlpha': [""],
    'Join': [""],
    'Python Algo': [""],
    'time': [""]
}

for logs_file in logs_files:
    logs = get_from_file(LOGS_DIR + logs_file)
    for key, log in logs.items():
        result['Classifier'].append(log['classifier']['classifier']['TYPE'])
        result['Vectorizer'].append(
            log['vectorizer']['TYPE'] + str(log['vectorizer']['OPTIONS'].get('ngram_range', [1, 1])))
        result['Validation Acc'].append(log['classifier']['val_score'])
        result['Test Acc'].append(log['classifier']['test_score'])
        result['Total'].append(log['classifier']['val_score'] * VAL_SIZE + log['classifier']['test_score'] * TEST_SIZE)
        result['Total Acc'].append(
            (log['classifier']['val_score'] * VAL_SIZE + log['classifier']['test_score'] * TEST_SIZE) / TOTAL_SIZE)
        result['time'].append(key)
        result['Python Algo'].append(str(log['data']['PREPROCESS_OPTIONS']))
        for k, v in log['data']['PREPROCESS_OPTIONS'].items():
            result[MAPPING[k]].append(v)

a = {}
for k, v in result.items():
    a[k] = pd.Series(v)
a = pd.DataFrame(a)
a.to_csv('res.csv', index=False)
