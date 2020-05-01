# -*- coding: utf-8 -*-
from tweets import Helpers
from sklearn.model_selection import train_test_split
from data import original_train_data as data
import pandas as pd

Helpers.correct_data(data)

id_t, id_val, keyword, keyword_val, location, location_val, text, text_val, target, target_val = train_test_split(
    data.id.values,
    data.keyword.values,
    data.location.values,
    data.text.values,
    data['target_relabeled'].values,
    test_size=0.2
)

train = pd.DataFrame({
    'id': pd.Series(id_t),
    'keyword': pd.Series(keyword),
    'location': pd.Series(location),
    'text': pd.Series(text),
    'target': pd.Series(target)
})

validation = pd.DataFrame({
    'id': pd.Series(id_val),
    'keyword': pd.Series(keyword_val),
    'location': pd.Series(location_val),
    'text': pd.Series(text_val),
    'target': pd.Series(target_val)
})

train.to_csv('./data/train.csv', index=False)
validation.to_csv('./data/validation.csv', index=False)
