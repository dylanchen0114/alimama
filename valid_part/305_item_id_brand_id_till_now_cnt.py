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

train_feat = train[['instance_id']].merge(concat[['instance_id', 'item_till_now_cnt', 'brand_till_now_cnt']], how='left'
                                          , on='instance_id')
test_feat = test[['instance_id']].merge(concat[['instance_id', 'item_till_now_cnt', 'brand_till_now_cnt']], how='left'
                                        , on='instance_id')

train_feat.to_pickle('../features/train_valid/train_feature_305.p')
test_feat.to_pickle('../features/train_valid/test_feature_305.p')

