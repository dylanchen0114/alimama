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

shop_list = concat.shop_id.values

# till_now count
shop_dict = defaultdict(lambda: 0)
shop_till_now_cnt = np.zeros(len(concat))


for i in tqdm(range(len(concat))):
    shop_till_now_cnt[i] = shop_dict[shop_list[i]]
    shop_dict[shop_list[i]] += 1

concat['shop_till_now_cnt'] = shop_till_now_cnt


train_feat = train[['instance_id']].merge(concat[['instance_id', 'shop_till_now_cnt']], how='left', on='instance_id')
test_feat = test[['instance_id']].merge(concat[['instance_id', 'shop_till_now_cnt']], how='left', on='instance_id')

train_feat.to_pickle('../features/train_valid/train_feature_203.p')
test_feat.to_pickle('../features/train_valid/test_feature_203.p')
