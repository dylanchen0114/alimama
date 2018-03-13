# -*- coding: utf-8 -*-

"""
@author: Dylan Chen

"""

from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm


concat_category = pd.read_pickle('../processed/concat_item_category.p')
concat_property = pd.read_pickle('../processed/concat_item_property.p')

category_list = concat_category.item_category_1.values
prop_list = concat_property.item_property.values


# till_now count
category_dict = defaultdict(lambda: 0)
category_till_now_cnt = np.zeros(len(concat_category))

prop_dict = defaultdict(lambda: 0)
prop_till_now_cnt = np.zeros(len(concat_property))


for i in tqdm(range(len(concat_category))):
    category_till_now_cnt[i] = category_dict[category_list[i]]
    category_dict[category_list[i]] += 1


for i in tqdm(range(len(concat_property))):
    prop_till_now_cnt[i] = prop_dict[prop_list[i]]
    prop_dict[prop_list[i]] += 1


concat_category['category_till_now_cnt'] = category_till_now_cnt
concat_property['prop_till_now_cnt'] = prop_till_now_cnt


concat_category.to_pickle('../features/concat_cate_till_now_cnt_feature_306.p')
concat_property.to_pickle('../features/concat_prop_till_now_cnt_feature_306.p')
