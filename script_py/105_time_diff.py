# -*- coding: utf-8 -*-

"""
@author: Dylan Chen

短时间内的多次曝光，转化率低；
距离上次曝光一段时间后的转化率似乎很高？

timestamp variance ?

"""

import numpy as np
import pandas as pd

train = pd.read_pickle('../processed/train_test/train_id_processed.p')
test = pd.read_pickle('../processed/train_test/test_id_processed.p')

col = ['instance_id', 'user_id', 'context_timestamp']
concat = train[col].append(test[col])

concat.sort_values(['user_id', 'context_timestamp'], inplace=True)

concat['t-1_context_timestamp'] = concat.groupby('user_id')['context_timestamp'].shift(1)

concat['time_diff_last_query'] = np.log1p(concat['context_timestamp'] - concat['t-1_context_timestamp'])


train_feat = concat[concat.instance_id.isin(train.instance_id)].copy()
test_feat = concat[concat.instance_id.isin(test.instance_id)].copy()


# user_mean = train.groupby(by='user_id').mean()['context_timestamp'].to_dict()
# train_feat['user_timestamp_mean'] = train_feat['user_id'].apply(lambda x: user_mean.get(x, np.nan))
# test_feat['user_timestamp_mean'] = test_feat['user_id'].apply(lambda x: user_mean.get(x, np.nan))
#
# user_std = train.groupby(by='user_id').std()['context_timestamp'].to_dict()
# train_feat['user_timestamp_std'] = train_feat['user_id'].apply(lambda x: user_std.get(x, np.nan))
# test_feat['user_timestamp_std'] = test_feat['user_id'].apply(lambda x: user_std.get(x, np.nan))


train_feat[['instance_id', 'time_diff_last_query']].to_pickle('../features/train_feature_105.p')
test_feat[['instance_id', 'time_diff_last_query']].to_pickle('../features/test_feature_105.p')
