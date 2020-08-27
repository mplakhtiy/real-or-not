import pandas as pd

data = pd.read_csv('./res.csv')

data_dict = data.to_dict()

models = ['BERT-8b7db91c', 'LSTM-GloVe-8b7db91c', 'LSTM_DROPOUT-GloVe-8b7db91c', 'BI_LSTM-GloVe-8b7db91c',
          'FASTTEXT-GloVe-8b7db91c', 'RCNN-GloVe-8b7db91c', 'CNN-GloVe-8b7db91c', 'RNN-GloVe-8b7db91c',
          'GRU-GloVe-8b7db91c',
          'HAN-GloVe-8b7db91c', 'HAN-GloVe-a85c8435', 'LSTM-71bd09db', 'LSTM_DROPOUT-d3cc3c6e', 'BI_LSTM-8b7db91c',
          'LSTM_CNN-8b7db91c', 'FASTTEXT-b054e509', 'RCNN-7bc816a1', 'CNN-b054e509', 'RNN-1258a9d2', 'GRU-b054e509',
          'HAN-a85c8435', 'RIDGE-2e359f0b', 'SVC-4c2e484d', 'LOGISTIC_REGRESSION-8b7db91c', 'SGD-2e359f0b',
          'DECISION_TREE-7bc816a1', 'RANDOM_FOREST-60314ef9']

result = {}
for k in data_dict:
    result[k] = []

for index in data_dict['id']:
    has_diff = False

    l = []
    for model_key in models:
        if round(data_dict[model_key][index]) != data_dict['target'][index]:
            l.append(True)
            has_diff = True
            # break

    # if has_diff:
    if len(l) != len(models) and len(l) != 0:
        result['id'].append(data_dict['id'][index])
        result['keyword'].append(data_dict['keyword'][index])
        result['location'].append(data_dict['location'][index])
        result['text'].append(data_dict['text'][index])
        result['target'].append(data_dict['target'][index])
        result['Correct Prediction'].append(data_dict['Correct Prediction'][index])
        for key in models:
            result[key].append(data_dict[key][index])

a = {}
for k, v in result.items():
    a[k] = pd.Series(v)
a = pd.DataFrame(a)
a.to_csv(f'filtered.csv', index=False)
