# -*- coding: utf-8 -*-

"""
@author: Dylan Chen

"""

import gc

import numpy as np
import pandas as pd
from tqdm import tqdm

train = pd.read_pickle('../processed/train_valid/train_id_processed.p')
test = pd.read_pickle('../processed/train_valid/test_id_processed.p')

concat_category = pd.read_pickle('../processed/concat_item_category.p')
concat_property = pd.read_pickle('../processed/concat_item_property.p')


train.replace(-1, np.nan, inplace=True)
test.replace(-1, np.nan, inplace=True)

# ================================================
#               basic count
# ================================================

group_keys = ['user_id', 'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']

cnt_cols = ['instance_id', 'item_id', 'item_brand_id', 'item_city_id', 'shop_id']
stat_cols = ['item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level',
             'shop_review_num_level', 'shop_review_positive_rate', 'shop_star_level', 'shop_score_service',
             'shop_score_delivery', 'shop_score_description', 'context_page_id']
item_prop = ['item_property_{}'.format(i) for i in range(100)]


concat = train.append(test)

for col in group_keys:
    concat[col] = concat[col].fillna(-1)

print('Concat shape:', concat.shape)

for grp in tqdm(group_keys):
    cnt_result = concat.groupby(grp)[cnt_cols].nunique()
    cnt_result = cnt_result.add_prefix(grp + '_').add_suffix('_cnt')

    statistics_results = []
    for col in stat_cols:
        result = concat.groupby(grp)[col].agg({
            '{}_{}_min'.format(grp, col): 'min',
            '{}_{}_median'.format(grp, col): 'median',
            '{}_{}_mean'.format(grp, col): 'mean',
            '{}_{}_max'.format(grp, col): 'max',
            '{}_{}_std'.format(grp, col): 'std',
            '{}_{}_skew'.format(grp, col): 'skew'})
        statistics_results.append(result)
    statistics_results = pd.concat(statistics_results, axis=1)

    # count category and property
    tmp_category = concat[['instance_id', grp]].merge(concat_category, how='left', on='instance_id')
    category_cnt = tmp_category.groupby(grp)['item_category_1'].nunique().to_frame()
    category_cnt.columns = ['{}_item_category_1_cnt'.format(grp)]

    tmp_property = concat[['instance_id', grp]].merge(concat_property, how='left', on='instance_id')
    property_cnt = tmp_property.groupby(grp)['item_property'].nunique().to_frame()
    property_cnt.columns = ['{}_item_property_cnt'.format(grp)]

    del tmp_category, tmp_property
    gc.collect()

    results = pd.concat([cnt_result, statistics_results, category_cnt, property_cnt], axis=1).reset_index()
    concat = concat.merge(results, how='left', on=grp)

gender_cnt = concat.groupby(['user_gender_id']).user_id.nunique().reset_index()
gender_cnt.columns = ['user_gender_id', 'user_gender_id_user_id_cnt']
concat = concat.merge(gender_cnt, how='left', on='user_gender_id')

age_cnt = concat.groupby(['user_age_level']).user_id.nunique().reset_index()
age_cnt.columns = ['user_age_level', 'user_age_level_user_id_cnt']
concat = concat.merge(age_cnt, how='left', on='user_age_level')

occupation_cnt = concat.groupby(['user_occupation_id']).user_id.nunique().reset_index()
occupation_cnt.columns = ['user_occupation_id', 'user_occupation_id_user_id_cnt']
concat = concat.merge(occupation_cnt, how='left', on='user_occupation_id')

print('Finish basic count, Concat shape:', concat.shape)

# ================================================
#                saving
# ================================================

feature_columns = [col for col in list(concat) if
                   col.endswith(('cnt', 'min', 'median', 'mean', 'max', 'std', 'skew'))]

concat[:len(train)][['instance_id'] + feature_columns].\
    to_pickle('../features/train_valid/train_feature_101.p')
concat[len(train):][['instance_id'] + feature_columns].\
    to_pickle('../features/train_valid/test_feature_101.p')
