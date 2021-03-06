# -*- coding: utf-8 -*-

"""
@author: Dylan Chen

"""

from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

train = pd.read_pickle('../processed/train_valid/train_id_processed.p')
test = pd.read_pickle('../processed/train_valid/test_id_processed.p')

concat = train.append(test)
concat = concat.sort_values('context_date')

user_list = concat.user_id.values


# till_now count
user_dict = defaultdict(lambda: 0)
user_till_now_cnt = np.zeros(len(concat))


for i in tqdm(range(len(concat))):
    user_till_now_cnt[i] = user_dict[user_list[i]]
    user_dict[user_list[i]] += 1

concat['user_till_now_cnt'] = user_till_now_cnt

train_feat = train[['instance_id']].merge(concat[['instance_id', 'user_till_now_cnt']], how='left', on='instance_id')
test_feat = test[['instance_id']].merge(concat[['instance_id', 'user_till_now_cnt']], how='left', on='instance_id')

train_feat.to_pickle('../features/train_valid/train_feature_103.p')
test_feat.to_pickle('../features/train_valid/test_feature_103.p')
