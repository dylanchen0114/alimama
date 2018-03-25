# -*- coding: utf-8 -*-

"""
@author: Dylan Chen

"""

import random

import pandas as pd

train = pd.read_pickle('../processed/train_test/train_id_processed.p')


# choose the last day in train as validation data
valid = train[train.context_date_day == 24].reset_index(drop=True).copy()

# 尝试对最后天按小时将行抽样
random.seed(114)

valid_sample = []
for hour in sorted(valid.context_date_hour.unique()):
    users_list = valid.loc[valid['context_date_hour'] == hour, 'user_id'].unique()
    sample_users = random.sample(list(users_list), int(len(users_list) * 0.3))
    valid_sample += sample_users

train = train[train.context_date_day < 24].append(valid[~valid.user_id.isin(valid_sample)]).copy()
valid = valid[valid.user_id.isin(valid_sample)].copy()

train = train.reset_index(drop=True)
valid = valid.reset_index(drop=True)

train.to_pickle('../processed/train_valid/train_id_processed.p')
valid.to_pickle('../processed/train_valid/test_id_processed.p')
