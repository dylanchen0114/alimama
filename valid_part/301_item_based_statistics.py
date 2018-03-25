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

group_keys = ['item_id', 'item_brand_id', 'item_city_id',
              'item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level']

cnt_cols = ['user_occupation_id', 'user_id', 'instance_id', 'shop_id']
stat_cols = ['user_age_level', 'user_star_level', 'context_page_id',
             'shop_review_num_level', 'shop_review_positive_rate', 'shop_star_level',
             'shop_score_service', 'shop_score_delivery', 'shop_score_description']

concat = train.append(test)

for col in group_keys:
    concat[col] = concat[col].fillna(-1)

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


brand_cnt = concat.groupby(['item_brand_id']).item_id.nunique().reset_index()
brand_cnt.columns = ['item_brand_id', 'item_brand_id_item_id_cnt']
concat = concat.merge(brand_cnt, how='left', on='item_brand_id')

city_cnt = concat.groupby(['item_city_id']).item_id.nunique().reset_index()
city_cnt.columns = ['item_city_id', 'item_city_id_item_id_cnt']
concat = concat.merge(city_cnt, how='left', on='item_city_id')

price_cnt = concat.groupby(['item_price_level']).item_id.nunique().reset_index()
price_cnt.columns = ['item_price_level', 'item_price_level_item_id_cnt']
concat = concat.merge(price_cnt, how='left', on='item_price_level')

collected_cnt = concat.groupby(['item_collected_level']).item_id.nunique().reset_index()
collected_cnt.columns = ['item_collected_level', 'item_collected_level_item_id_cnt']
concat = concat.merge(collected_cnt, how='left', on='item_collected_level')

pv_cnt = concat.groupby(['item_pv_level']).item_id.nunique().reset_index()
pv_cnt.columns = ['item_pv_level', 'item_pv_level_item_id_cnt']
concat = concat.merge(pv_cnt, how='left', on='item_pv_level')

print('Finish basic count, Concat shape:', concat.shape)


# ================================================
#                saving
# ================================================

feature_columns = [col for col in list(concat) if
                   col.endswith(('cnt', 'min', 'median', 'mean', 'max', 'std', 'skew'))]

concat[:len(train)][['instance_id'] + feature_columns].\
    to_pickle('../features/train_valid/train_feature_301.p')
concat[len(train):][['instance_id'] + feature_columns].\
    to_pickle('../features/train_valid/test_feature_301.p')
