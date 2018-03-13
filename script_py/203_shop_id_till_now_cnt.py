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
shop_list = concat.shop_id.values


# till_now count
shop_dict = defaultdict(lambda: 0)
shop_till_now_cnt = np.zeros(len(concat))


for i in tqdm(range(len(concat))):
    shop_till_now_cnt[i] = shop_dict[shop_list[i]]
    shop_dict[shop_list[i]] += 1

concat['shop_till_now_cnt'] = shop_till_now_cnt


concat[:len(train)][['instance_id', 'shop_till_now_cnt']].to_pickle('../features/train_feature_203.p')
concat[len(train):][['instance_id', 'shop_till_now_cnt']].to_pickle('../features/test_feature_203.p')