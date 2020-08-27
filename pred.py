from utils import get_from_file
import pandas as pd

predictions = get_from_file('./predictions_v_5.json')

keys = ['glove-false', 'classifiers']

csv_res = {
    'Correct Prediction': None,
}

for key in keys:
    for k, v in predictions[key].items():
        csv_res[k] = v['x_test']

csv_res['Correct Prediction'] = predictions['y_test']

a = {}
for k, v in csv_res.items():
    a[k] = pd.Series(v)
a = pd.DataFrame(a)
a.to_csv(f'pred_comp.csv', index=False)
