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

item_list = concat.item_id.values
brand_list = concat.item_brand_id.values


# till_now count
item_dict = defaultdict(lambda: 0)
item_till_now_cnt = np.zeros(len(concat))

brand_dict = defaultdict(lambda: 0)
brand_till_now_cnt = np.zeros(len(concat))


for i in tqdm(range(len(concat))):
    item_till_now_cnt[i] = item_dict[item_list[i]]
    item_dict[item_list[i]] += 1

    brand_till_now_cnt[i] = brand_dict[brand_list[i]]
    brand_dict[brand_list[i]] += 1


concat['item_till_now_cnt'] = item_till_now_cnt
concat['brand_till_now_cnt'] = brand_till_now_cnt


concat[:len(train)][['instance_id', 'item_till_now_cnt', 'brand_till_now_cnt']].\
    to_pickle('../features/train_feature_305.p')
concat[len(train):][['instance_id', 'item_till_now_cnt', 'brand_till_now_cnt']].\
    to_pickle('../features/test_feature_305.p')
