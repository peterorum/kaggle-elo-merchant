# initial lbg of just train
# local 3.835
# kaggle
# minimize score

import os
import sys  # noqa
from time import time
from pprint import pprint  # noqa
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

pd.options.display.float_format = '{:.4f}'.format
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)

is_kaggle = os.environ['HOME'] == '/tmp'

zipext = ''  # if is_kaggle else '.zip'


def evaluate():

    lgb_model = lgb.LGBMRegressor(nthread=4, n_jobs=-1, verbose=-1)

    x_train = train.drop([target, unique_id], axis=1)
    y_train = train[target]

    x_test = test[x_train.columns]

    lgb_model.fit(x_train, y_train)

    train_predictions = lgb_model.predict(x_train)
    test_predictions = lgb_model.predict(x_test)

    train_score = np.sqrt(mean_squared_error(train_predictions, y_train))

    return test_predictions, train_score

# -------- categorical data


def get_categorical_data():
    categorical_cols = [col for col in train.columns if train[col].dtype == 'object']
    categorical_cols.remove(unique_id)

    label_encode_categorical_cols = categorical_cols

    # label encode (convert to integer)
    for col in label_encode_categorical_cols:
        lbl = LabelEncoder()
        lbl.fit(list(train[col].values.astype('str')) + list(test[col].values.astype('str')))
        train[col] = lbl.transform(list(train[col].values.astype('str')))
        test[col] = lbl.transform(list(test[col].values.astype('str')))

    return train, test


# -------- main


start_time = time()

unique_id = 'card_id'
target = 'target'

# load data

train = pd.read_csv(f'../input/train.csv{zipext}')
test = pd.read_csv(f'../input/test.csv{zipext}')

# categorical data

train, test = get_categorical_data()

# ----------

test_predictions, train_score = evaluate()

print('score', train_score)

test[target] = test_predictions

predictions = test[[unique_id, target]]

predictions.to_csv('submission.csv', index=False)

print(f'{((time() - start_time) / 60):.0f} mins\a')
