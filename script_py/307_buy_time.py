# -*- coding: utf-8 -*-

"""
@author: Dylan Chen

"""

import numpy as np
import pandas as pd


def timezone(s):
    if s < 6:
        return 'midnight'
    elif s < 12:
        return 'morning'
    elif s < 18:
        return 'noon'
    else:
        return 'night'


train = pd.read_pickle('../processed/train_test/train_id_processed.p')
test = pd.read_pickle('../processed/train_test/test_id_processed.p')

cols = ['instance_id', 'item_id', 'context_date_hour', 'context_date']
concat = train[cols].append(test[cols])


# expose hour
exp_hour = concat.groupby(['item_id', 'context_date_hour']).instance_id.nunique().to_frame()
exp_hour.columns = ['item_hour_cnt']
exp_hour = exp_hour.reset_index()

exp_hour['item_hour_cnt_ratio'] = exp_hour.item_hour_cnt / exp_hour.groupby('item_id').item_hour_cnt.transform(np.sum)


# expose timezone
concat['context_hour'] = concat['context_date'].map(lambda x: str(x)[11:13]).astype(int)
concat['context_timezone'] = concat['context_hour'].map(timezone)

exp_tz = concat.groupby(['item_id', 'context_timezone']).instance_id.nunique().to_frame()
exp_tz.columns = ['item_timezone_cnt']
exp_tz = exp_tz.reset_index()

exp_tz['item_timezone_cnt_ratio'] = exp_tz.item_timezone_cnt / \
                                    exp_tz.groupby('item_id').item_timezone_cnt.transform(np.sum)

features = concat.\
    merge(exp_hour, how='left', on=['item_id', 'context_date_hour']).\
    merge(exp_tz, how='left', on=['item_id', 'context_timezone'])

# saving
feature_columns = [col for col in list(features) if
                   col.endswith(('cnt', 'ratio'))]

features[:len(train)][['instance_id'] + feature_columns].to_pickle('../features/train_feature_307.p')
features[len(train):][['instance_id'] + feature_columns].to_pickle('../features/test_feature_307.p')


