# -*- coding: utf-8 -*-

"""
@author: Dylan Chen

statistics by item category and property  

"""

import gc

import numpy as np
import pandas as pd

train = pd.read_pickle('../processed/train_test/train_id_processed.p')
test = pd.read_pickle('../processed/train_test/test_id_processed.p')

concat_category = pd.read_pickle('../processed/concat_item_category.p')
concat_property = pd.read_pickle('../processed/concat_item_property.p')

cnt_cols = ['instance_id', 'item_brand_id', 'user_id', 'shop_id', 'context_page_id']
stat_cols = ['item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level',
             'user_age_level', 'user_star_level', 'context_page_id',
             'shop_review_num_level', 'shop_review_positive_rate',
             'shop_star_level', 'shop_score_service', 'shop_score_delivery', 'shop_score_description']


concat = train.append(test)

concat.replace(-1, np.nan, inplace=True)


# ================================================
#               category part
# ================================================

tmp_category = concat_category.merge(concat[cnt_cols + stat_cols], how='left', on='instance_id')

cate_cnt = tmp_category.groupby('item_category_1')[cnt_cols].nunique().\
    add_prefix('item_category_1' + '_').add_suffix('_cnt')

statistics_results = []
for col in stat_cols:
    result = tmp_category.groupby('item_category_1')[col].agg({
        '{}_{}_min'.format('item_category_1', col): 'min',
        '{}_{}_median'.format('item_category_1', col): 'median',
        '{}_{}_mean'.format('item_category_1', col): 'mean',
        '{}_{}_max'.format('item_category_1', col): 'max',
        '{}_{}_std'.format('item_category_1', col): 'std',
        '{}_{}_skew'.format('item_category_1', col): 'skew'})
    statistics_results.append(result)
statistics_results = pd.concat(statistics_results, axis=1)

del tmp_category
gc.collect()

cate_cnt_prop = concat_category.merge(concat_property, how='left', on='instance_id')
cate_cnt_prop = cate_cnt_prop.groupby('item_category_1')['item_property'].nunique().to_frame()
cate_cnt_prop.columns = ['item_category_1_item_property_cnt']

result1 = pd.concat([cate_cnt, statistics_results, cate_cnt_prop], axis=1)


# ================================================
#               property part
# ================================================

tmp_property = concat_property.merge(concat[cnt_cols + stat_cols], how='left', on='instance_id')

prop_cnt = tmp_property.groupby('item_property')[cnt_cols].nunique().\
    add_prefix('item_property' + '_').add_suffix('_cnt')

statistics_results = []
for col in stat_cols:
    result = tmp_property.groupby('item_property')[col].agg({
        '{}_{}_min'.format('item_property', col): 'min',
        '{}_{}_median'.format('item_property', col): 'median',
        '{}_{}_mean'.format('item_property', col): 'mean',
        '{}_{}_max'.format('item_property', col): 'max',
        '{}_{}_std'.format('item_property', col): 'std',
        '{}_{}_skew'.format('item_property', col): 'skew'})
    statistics_results.append(result)
statistics_results = pd.concat(statistics_results, axis=1)

del tmp_property
gc.collect()

cate_cnt_prop = concat_category.merge(concat_property, how='left', on='instance_id')
cate_cnt_prop = cate_cnt_prop.groupby('item_property')['item_category_1'].nunique().to_frame()
cate_cnt_prop.columns = ['item_property_item_category_1_cnt']

result2 = pd.concat([prop_cnt, statistics_results, cate_cnt_prop], axis=1)

result1.to_pickle('../features/concat_category_feature_303.p')
result2.to_pickle('../features/concat_property_feature_303.p')
