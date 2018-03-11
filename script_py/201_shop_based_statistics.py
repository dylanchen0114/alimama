# -*- coding: utf-8 -*-

"""
@author: Dylan Chen

"""

import gc

import numpy as np
import pandas as pd
from tqdm import tqdm

train = pd.read_pickle('../processed/train_test/train_id_processed.p')
test = pd.read_pickle('../processed/train_test/test_id_processed.p')

concat_category = pd.read_pickle('../processed/concat_item_category.p')
concat_property = pd.read_pickle('../processed/concat_item_property.p')


train.replace(-1, np.nan, inplace=True)
test.replace(-1, np.nan, inplace=True)


concat = train.append(test)

# ================================================
#               basic count
# ================================================

# TODO: transform other shop features to bins, and calculate the following features
group_keys = ['shop_id', 'shop_review_num_level', 'shop_star_level']

cnt_cols = ['user_occupation_id', 'user_id', 'instance_id', 'item_id', 'item_brand_id', 'item_city_id']
stat_cols = ['item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level',
             'user_age_level', 'user_star_level',
             'context_page_id']

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


review_cnt = concat.groupby(['shop_review_num_level']).shop_id.nunique().reset_index()
review_cnt.columns = ['shop_review_num_level', 'shop_review_num_level_shop_id_cnt']
concat = concat.merge(review_cnt, how='left', on='shop_review_num_level')

star_level_cnt = concat.groupby(['shop_star_level']).shop_id.nunique().reset_index()
star_level_cnt.columns = ['shop_star_level', 'shop_star_level_shop_id_cnt']
concat = concat.merge(star_level_cnt, how='left', on='shop_star_level')

print('Finish basic count, Concat shape:', concat.shape)


# ================================================
#                saving
# ================================================

feature_columns = [col for col in list(concat) if
                   col.endswith(('cnt', 'min', 'median', 'mean', 'max', 'std', 'skew'))]

concat[:len(train)][['instance_id'] + feature_columns].to_pickle('../features/train_feature_201.p')
concat[len(train):][['instance_id'] + feature_columns].to_pickle('../features/test_feature_201.p')
