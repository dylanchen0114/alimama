# -*- coding: utf-8 -*-

"""
@author: Dylan Chen

"""

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


train = pd.read_pickle('../processed/train_valid/train_id_processed.p')
test = pd.read_pickle('../processed/train_valid/test_id_processed.p')


train['context_hour'] = train['context_date'].map(lambda x: str(x)[11:13]).astype(int)
test['context_hour'] = test['context_date'].map(lambda x: str(x)[11:13]).astype(int)

train['context_timezone'] = train['context_hour'].map(timezone)
test['context_timezone'] = test['context_hour'].map(timezone)


timezone_freq = pd.crosstab(train.user_id, train.context_timezone).add_prefix('user_timezone_freq_')
timezone_freq_ = pd.crosstab(train.user_id, train.context_timezone, normalize='index').\
    add_prefix('user_timezone_freq_norm_')

result = pd.concat([timezone_freq, timezone_freq_], axis=1)

result = result.reset_index().copy()


train = train.merge(result, how='left', on='user_id')
test = test.merge(result, how='left', on='user_id')

# ================================================
#                saving
# ================================================

feature_columns = [col for col in list(train) if
                   col.endswith(('midnight', 'morning', 'noon', 'night'))]

train[['instance_id'] + feature_columns].to_pickle('../features/train_valid/train_feature_104.p')
test[['instance_id'] + feature_columns].to_pickle('../features/train_valid/test_feature_104.p')


