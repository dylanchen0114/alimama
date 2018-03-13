# -*- coding: utf-8 -*-

"""
@author: Dylan Chen

"""

from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

train = pd.read_pickle('../processed/train_test/train_id_processed.p')
test = pd.read_pickle('../processed/train_test/test_id_processed.p')

concat = train.append(test)
user_list = concat.user_id.values


# till_now count
user_dict = defaultdict(lambda: 0)
user_till_now_cnt = np.zeros(len(concat))


for i in tqdm(range(len(concat))):
    user_till_now_cnt[i] = user_dict[user_list[i]]
    user_dict[user_list[i]] += 1

concat['user_till_now_cnt'] = user_till_now_cnt


concat[:len(train)][['instance_id', 'user_till_now_cnt']].to_pickle('../features/train_feature_103.p')
concat[len(train):][['instance_id', 'user_till_now_cnt']].to_pickle('../features/test_feature_103.p')
