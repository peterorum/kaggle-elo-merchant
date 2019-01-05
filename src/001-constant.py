# baseline: constant value
# local score  3.852
# kaggle score 3.931

import sys  # noqa
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from time import time

import os

is_kaggle = os.environ['HOME'] == '/tmp'

zipext = ''  # if is_kaggle else '.zip'

# load data
train = pd.read_csv(f'../input/train.csv{zipext}')
test = pd.read_csv(f'../input/test.csv{zipext}')

# -------- main

start_time = time()

target = 'target'
unique_id = 'card_id'

result = -0.5

train['predicted'] = result

score = np.sqrt(mean_squared_error(train[target], train.predicted))
print('score', score)

test[target] = result

# print(test.head())
# print(test.describe())

predictions = test[[unique_id, target]]

predictions.to_csv('submission.csv', index=False)

print('%.0f mins' % ((time() - start_time) / 60))
