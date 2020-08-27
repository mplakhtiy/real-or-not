import pandas as pd


data = pd.read_csv('./res.csv')

data_dict= data.to_dict()


models = ['BERT', 'LSTM', 'LSTM_DROPOUT', 'BI_LSTM', 'FASTTEXT', 'RCNN', 'CNN', 'RNN', 'GRU']
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
    if len(l) != len(models) and len(l) !=0:
        result['id'].append(data_dict['id'][index])
        result['keyword'].append(data_dict['keyword'][index])
        result['location'].append(data_dict['location'][index])
        result['text'].append(data_dict['text'][index])
        result['target'].append(data_dict['target'][index])
        result['Control Correct Prediction'].append(data_dict['Control Correct Prediction'][index])
        for key in models:
            result[key].append(data_dict[key][index])


a = {}
for k, v in result.items():
    a[k] = pd.Series(v)
a = pd.DataFrame(a)
a.to_csv(f'filtered_random.csv', index=False)
