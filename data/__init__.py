import pandas as pd

original_train_data = pd.read_csv('./data/original/train.csv')

original_test_data = pd.read_csv('./data/original/test.csv')

original_test_data_with_target = pd.read_csv('./data/original/test_with_target.csv')

train = pd.read_csv('./data/train.csv')

validation = pd.read_csv('./data/validation.csv')

test = pd.read_csv('./data/test.csv')
