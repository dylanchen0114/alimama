# -*- coding: utf-8 -*-

"""
@author: Dylan Chen

"""

import datetime
import gc
from glob import glob

import lightgbm as lgb
import numpy as np
import pandas as pd
from tqdm import tqdm


def read_pickles(path, col=None):
    if col is None:
        df = pd.concat([pd.read_pickle(f) for f in tqdm(sorted(glob(path+'/*')))])
    else:
        df = pd.concat([pd.read_pickle(f)[col] for f in tqdm(sorted(glob(path+'/*')))])
    return df


train = read_pickles('../processed/train_valid/train_x').sort_index()
train_y = pd.read_pickle('../processed/train_valid/train_y.p')

valid = pd.read_pickle('../processed/train_valid/valid_x.p')
valid_y = pd.read_pickle('../processed/train_valid/valid_y.p')

# feat_importance = pd.read_csv('../feat_importance.csv')
# feature_name = feat_importance['name'].values
# feature_importance = feat_importance['importance'].values
#
# drop_col = feature_name[feature_importance < 10].tolist()

train_data = lgb.Dataset(train, label=train_y)
valid_data = lgb.Dataset(valid, label=valid_y, reference=train_data)

feat_cnt = valid.shape[1]

del train, valid
gc.collect()

num_rounds = 5000
params = {
    'boosting_type': 'gbdt',  # np.random.choice(['dart', 'gbdt']),
    'objective': 'binary',
    'metric': ['binary_logloss'],
    'max_bin': 256,

    'learning_rate': 0.02,

    'num_leaves': 100,
    'max_depth': 15,
    'min_data_in_leaf': 1000,

    'feature_fraction': 0.6,
    'bagging_fraction': 0.6,
    'bagging_freq': 1,

    'lambda_l1': 0,
    'lambda_l2': 0,
    'min_gain_to_split': 0.0,
    'min_sum_hessian_in_leaf': 0,

    'verbose': 1,
    'is_training_metric': 'True'
}


evals_result = {}

gbm = lgb.train(params, train_data, num_boost_round=num_rounds, valid_sets=[train_data, valid_data],
                valid_names=['train', 'valid'], evals_result=evals_result,
                early_stopping_rounds=150, verbose_eval=50)

bst_round = np.argmin(evals_result['valid']['binary_logloss'])

trn_loss = evals_result['train']['binary_logloss'][bst_round]
val_loss = evals_result['valid']['binary_logloss'][bst_round]

print('Best Round: %d' % bst_round)
print('Training loss: %.5f, Validation loss: %.5f' % (trn_loss, val_loss))


feature_importance = pd.DataFrame({'name': gbm.feature_name(), 'importance': gbm.feature_importance()}).sort_values(
    by='importance', ascending=False)
feature_importance.to_csv('../feat_importance_valid.csv', index=False)

res = '%s,%s,%d,%s,%.4f,%d,%d,%d,%.4f,%.4f,%d,%.4e,%.4e,%.4e,%.4e,%.4e,%s,%.5f,%.5f\n' % \
          (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
           'LightGBM_with_timestamp_context_prob', feat_cnt, params['boosting_type'],
           params['learning_rate'], params['num_leaves'], params['max_depth'],
           params['min_data_in_leaf'], params['feature_fraction'], params['bagging_fraction'],
           params['bagging_freq'], params['lambda_l1'], params['lambda_l2'], params['min_gain_to_split'],
           params['min_sum_hessian_in_leaf'], 0.0, bst_round+1, trn_loss, val_loss)
f = open('../lgb_record.csv', 'a')
f.write(res)
f.close()

