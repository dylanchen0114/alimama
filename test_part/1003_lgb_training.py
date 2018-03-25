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


train = read_pickles('../processed/train_test/train_x').sort_index()  # 保持train原来的顺序,和y对应
train_y = pd.read_pickle('../processed/train_test/train_y.p')

test = pd.read_pickle('../processed/train_test/test_x.p')
test_id = pd.read_pickle('../processed/train_test/test_id.p')


train_data = lgb.Dataset(train, label=train_y)

num_rounds = 575

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

pred_train = gbm.predict(train)
trn_loss = log_loss(y_pred=pred_train, y_true=train_y)
trn_auc = roc_auc_score(y_score=pred_train, y_true=train_y)

print('Training loss: %.5f, Training AUC: %.5f' % (trn_loss, trn_auc))

feature_importance = pd.DataFrame({'name': gbm.feature_name(), 'importance': gbm.feature_importance()}).sort_values(
    by='importance', ascending=False)
feature_importance.to_csv('../full_train_feat_importance.csv', index=False)

pred_test = gbm.predict(test)
test_sub = pd.DataFrame({'instance_id': test_id, 'predicted_score': pred_test})

test_sub.to_csv('../lgb_%.5f_logloss.txt' % trn_loss, index=False, sep=' ', line_terminator='\n')
