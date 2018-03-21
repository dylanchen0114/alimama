# -*- coding: utf-8 -*-

"""
@author: Dylan Chen

"""

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


def to_pickles(df, path, split_size=3):
    """
    path = '../output/mydf'

    wirte '../output/mydf/0.p'
          '../output/mydf/1.p'
          '../output/mydf/2.p'

    """

    for i in tqdm(range(split_size)):
        df.ix[df.index % split_size == i].to_pickle(path + '/{}.p'.format(i))

    return

# train = pd.read_pickle('../processed/train_test/train_id_processed.p')
# # test = pd.read_pickle('../processed/train_test/test_id_processed.p')
#
# train_feats = read_pickles('../features/all/train/')
# train_feats.drop(['context_date_day', 'context_date'], axis=1)
# # test_feats = pd.read_csv('../features/all/test_concat_all.csv')
#
# train = train.merge(train_feats, how='left', on='instance_id')
#
# valid = train[train.context_date_day == 24].copy()
# train = train[train.context_date_day < 24].copy()
#
# drop_columns = ['instance_id', 'context_id', 'context_date',
#                 'context_date_day', 'context_date_hour',
#                 'item_id', 'item_brand_id', 'item_city_id', 'user_id', 'shop_id']
# train.drop(drop_columns, axis=1, inplace=True)
# valid.drop(drop_columns, axis=1, inplace=True)
#
# embedding_features = ['user_gender_id', 'user_occupation_id']
#
# for col in embedding_features:
#     train[col] = train[col].astype('category')
#     valid[col] = valid[col].astype('category')
#
#
# train_y = train['is_trade'].reset_index(drop=True).copy()
# train = train.reset_index(drop=True).drop('is_trade', axis=1)
#
# valid_y = valid['is_trade'].reset_index(drop=True).copy()
# valid = valid.reset_index(drop=True).drop('is_trade', axis=1)
#
# train_y.to_pickle('../processed/train_test/train_y.p')
# to_pickles(df=train, path='../processed/train_test/train_x/')
#
# valid_y.to_pickle('../processed/train_test/valid_y.p')
# valid.to_pickle('../processed/train_test/valid_x.p')


#####################################################
#               Model Training
#####################################################

train = read_pickles('../processed/train_test/train_x')
train_y = pd.read_pickle('../processed/train_test/train_y.p')

valid = pd.read_pickle('../processed/train_test/valid_x.p')
valid_y = pd.read_pickle('../processed/train_test/valid_y.p')

# feat_importance = pd.read_csv('../feat_importance.csv')
# feature_name = feat_importance['name'].values
# feature_importance = feat_importance['importance'].values
#
# drop_col = feature_name[feature_importance < 10].tolist()

train.drop(['context_timestamp', 'item_category_1'], axis=1, inplace=True)
valid.drop(['context_timestamp', 'item_category_1'], axis=1, inplace=True)

train_data = lgb.Dataset(train, label=train_y)
valid_data = lgb.Dataset(valid, label=valid_y, reference=train_data)

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
                early_stopping_rounds=250, verbose_eval=50)

bst_round = np.argmin(evals_result['valid']['binary_logloss'])

# trn_auc = evals_result['train']['auc'][bst_round]
trn_loss = evals_result['train']['binary_logloss'][bst_round]

# val_auc = evals_result['valid']['auc'][bst_round]
val_loss = evals_result['valid']['binary_logloss'][bst_round]

print('Best Round: %d' % bst_round)
print('Training loss: %.5f, Validation loss: %.5f' % (trn_loss, val_loss))
# print('Training AUC : %.5f, Validation AUC : %.5f' % (trn_auc, val_auc))

feature_importance = pd.DataFrame({'name': gbm.feature_name(), 'importance': gbm.feature_importance()}).sort_values(
    by='importance', ascending=False)
feature_importance.to_csv('../feat_importance.csv', index=False)

# 0.08704
