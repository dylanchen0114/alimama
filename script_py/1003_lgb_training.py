# -*- coding: utf-8 -*-

"""
@author: Dylan Chen

"""

from glob import glob

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss
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


train = pd.read_pickle('../processed/train_test/train_id_processed.p')
test = pd.read_pickle('../processed/train_test/test_id_processed.p')

train_feats = pd.read_csv('../features/all/train_concat_all.csv')
test_feats = pd.read_csv('../features/all/test_concat_all.csv')

train = train.merge(train_feats, how='left', on='instance_id')
test = test.merge(test_feats, how='left', on='instance_id')

test_id = test['instance_id']
drop_columns = ['instance_id', 'context_id', 'context_date',
                'context_date_day', 'context_date_hour',
                'item_id', 'item_brand_id', 'item_city_id', 'user_id', 'shop_id']
train.drop(drop_columns, axis=1, inplace=True)
test.drop(drop_columns, axis=1, inplace=True)

embedding_features = ['user_gender_id', 'user_occupation_id']

for col in embedding_features:
    train[col] = train[col].astype('category')
    test[col] = test[col].astype('category')


train_y = train['is_trade'].reset_index(drop=True).copy()
train = train.reset_index(drop=True).drop('is_trade', axis=1)

#####################################################
#               Model Training
#####################################################

train.drop(['context_timestamp'], axis=1, inplace=True)
test.drop(['context_timestamp'], axis=1, inplace=True)

train_data = lgb.Dataset(train, label=train_y)

num_rounds = 299

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


gbm = lgb.train(params, train_data, num_boost_round=num_rounds, valid_sets=[train_data],
                early_stopping_rounds=250, verbose_eval=50)

pred_train = gbm.predict(train, num_iteration=256)
trn_loss = log_loss(y_pred=pred_train, y_true=train_y)
trn_auc = roc_auc_score(y_score=pred_train, y_true=train_y)

print('Training loss: %.5f, Training AUC: %.5f' % (trn_loss, trn_auc))

feature_importance = pd.DataFrame({'name': gbm.feature_name(), 'importance': gbm.feature_importance()}).sort_values(
    by='importance', ascending=False)
feature_importance.to_csv('../full_train_feat_importance.csv', index=False)

pred_test = gbm.predict(test)
test_sub = pd.DataFrame({'instance_id': test_id, 'predicted_score': pred_test})

test_sub.to_csv('../lgb_%.5f_logloss.txt' % trn_loss, index=False, sep=' ', line_terminator='\n')







# num_rounds = 299
#
# params = {
#     'boosting_type': 'gbdt',  # np.random.choice(['dart', 'gbdt']),
#     'objective': 'binary',
#     'metric': ['binary_logloss'],
#     'max_bin': 256,
#
#     'learning_rate': 0.02,
#
#     'num_leaves': 100,
#     'max_depth': 15,
#     'min_data_in_leaf': 1000,
#
#     'feature_fraction': 0.6,
#     'bagging_fraction': 0.6,
#     'bagging_freq': 1,
#
#     'lambda_l1': 0,
#     'lambda_l2': 0,
#     'min_gain_to_split': 0.0,
#     'min_sum_hessian_in_leaf': 0,
#
#     'verbose': 1,
#     'is_training_metric': 'True'
# }