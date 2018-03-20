# -*- coding: utf-8 -*-

"""
@author: Dylan Chen

"""

import numpy as np
import pandas as pd

train = pd.read_pickle('../processed/train_test/train_id_processed.p')
test = pd.read_pickle('../processed/train_test/test_id_processed.p')

concat_category = pd.read_pickle('../processed/concat_item_category.p')
concat_property = pd.read_pickle('../processed/concat_item_property.p')

col = ['instance_id', 'user_id', 'context_timestamp']
concat = train[col].append(test[col])

concat.sort_values(['user_id', 'context_timestamp'], inplace=True)

concat = concat.merge(concat_category[['instance_id', 'item_category_1']], how='left', on='instance_id')

concat['t-1_context_timestamp_cate'] = concat.groupby(['user_id', 'item_category_1'])['context_timestamp'].shift(1)

concat['time_diff_last_expose_this_cate'] = np.log1p(concat['context_timestamp'] - concat['t-1_context_timestamp_cate'])

train_feat = concat[concat.instance_id.isin(train.instance_id)].copy()
test_feat = concat[concat.instance_id.isin(test.instance_id)].copy()

train_feat[['instance_id', 'time_diff_last_expose_this_cate']].to_pickle('../features/train_feature_502.p')
test_feat[['instance_id', 'time_diff_last_expose_this_cate']].to_pickle('../features/test_feature_502.p')

