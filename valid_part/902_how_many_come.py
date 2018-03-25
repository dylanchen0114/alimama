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


train = pd.read_pickle('../processed/train_valid/train_id_processed.p')
test = pd.read_pickle('../processed/train_valid/test_id_processed.p')

cols = ['instance_id', 'context_date_hour', 'context_date']
concat = train[cols + ['is_trade']].append(test[cols])


# how many expo by hour
exp_hour = concat.groupby(['context_date_hour']).instance_id.nunique().to_frame()
exp_hour.columns = ['how_many_hour_cnt']
exp_hour = exp_hour.reset_index()

exp_hour['how_many_hour_cnt_ratio'] = exp_hour.how_many_hour_cnt / np.sum(exp_hour.how_many_hour_cnt)


# expose timezone
concat['context_hour'] = concat['context_date'].map(lambda x: str(x)[11:13]).astype(int)
concat['context_timezone'] = concat['context_hour'].map(timezone)

exp_tz = concat.groupby(['context_timezone']).instance_id.nunique().to_frame()
exp_tz.columns = ['how_many_timezone_cnt']
exp_tz = exp_tz.reset_index()

exp_tz['how_many_timezone_cnt_ratio'] = exp_tz.how_many_timezone_cnt / np.sum(exp_tz.how_many_timezone_cnt)


# convert rate
hour_sum = concat[pd.notnull(concat.is_trade)].groupby(['context_date_hour']).is_trade.sum().to_frame()
hour_sum.columns = ['hour_cnvt_1']

hour_size = concat[pd.notnull(concat.is_trade)].groupby(['context_date_hour']).is_trade.size().to_frame()
hour_size.columns = ['hour_cnvt_all']

hour_ctr = pd.concat([hour_sum, hour_size], axis=1)
hour_ctr['hour_cnvt_rate'] = hour_ctr.hour_cnvt_1 / hour_ctr.hour_cnvt_all

hour_ctr = hour_ctr.reset_index()


tz_sum = concat[pd.notnull(concat.is_trade)].groupby(['context_timezone']).is_trade.sum().to_frame()
tz_sum.columns = ['timezone_cnvt_1']

tz_size = concat[pd.notnull(concat.is_trade)].groupby(['context_timezone']).is_trade.size().to_frame()
tz_size.columns = ['timezone_cnvt_all']

tz_ctr = pd.concat([tz_sum, tz_size], axis=1)
tz_ctr['timezone_cnvt_rate'] = tz_ctr.timezone_cnvt_1 / tz_ctr.timezone_cnvt_all

tz_ctr = tz_ctr.reset_index()


features = concat.\
    merge(exp_hour, how='left', on=['context_date_hour']).\
    merge(exp_tz, how='left', on=['context_timezone']).\
    merge(hour_ctr, how='left', on=['context_date_hour']).\
    merge(tz_ctr, how='left', on=['context_timezone'])

# saving
feature_columns = [col for col in list(features) if
                   col.endswith(('_1', '_all', '_rate', 'cnt', 'ratio'))]

features[:len(train)][['instance_id'] + feature_columns].\
    to_pickle('../features/train_valid/train_feature_902.p')
features[len(train):][['instance_id'] + feature_columns].\
    to_pickle('../features/train_valid/test_feature_902.p')


