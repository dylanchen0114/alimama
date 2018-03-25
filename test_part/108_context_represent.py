# -*- coding: utf-8 -*-

"""
@author: Dylan Chen

重要：同一个context_timestamp下，有多次曝光的，转化率??

"""

import pandas as pd
import numpy as np


train = pd.read_pickle('../processed/train_test/train_id_processed.p')
test = pd.read_pickle('../processed/train_test/test_id_processed.p')

concat_predict = pd.read_pickle('../processed/concat_predict_item_category_property.p')
concat_predict.replace(-1, np.nan, inplace=True)


# category and property count
pred_cnt = concat_predict.groupby('instance_id')['predict_item_category', 'predict_item_property'].nunique()
pred_cnt.columns = ['pred_cate_cnt', 'pred_prop_cnt']

pred_cnt = pred_cnt.reset_index()


# same time exposure
cols = ['user_id', 'context_timestamp', 'instance_id']
concat = train[cols].append(test[cols])

same_expo = concat.groupby(['user_id', 'context_timestamp'])['instance_id'].nunique().to_frame()
same_expo.columns = ['same_time_expo_cnt']

same_expo = same_expo.reset_index()

train_feats = train[['user_id', 'context_timestamp', 'instance_id']].\
    merge(same_expo, how='left', on=['user_id', 'context_timestamp']).\
    merge(pred_cnt, how='left', on=['instance_id'])

test_feats = test[['user_id', 'context_timestamp', 'instance_id']].\
    merge(same_expo, how='left', on=['user_id', 'context_timestamp']).\
    merge(pred_cnt, how='left', on=['instance_id'])


# saving
train_feats.drop(['context_timestamp', 'user_id'], axis=1).\
    to_pickle('../features/train_test/train_feature_108.p')
test_feats.drop(['context_timestamp', 'user_id'], axis=1).\
    to_pickle('../features/train_test/test_feature_108.p')
