# -*- coding: utf-8 -*-

"""
@author: Dylan Chen

"""

import numpy as np
import pandas as pd

train = pd.read_pickle('../processed/train_test/train_id_processed.p')
test = pd.read_pickle('../processed/train_test/test_id_processed.p')

train.replace(-1, np.nan, inplace=True)
test.replace(-1, np.nan, inplace=True)

train_category = pd.read_pickle('../processed/train_category_with_instance_id.p')
test_category = pd.read_pickle('../processed/test_category_with_instance_id.p')

train_category.replace('missing', np.nan, inplace=True)
test_category.replace('missing', np.nan, inplace=True)

train = pd.DataFrame(pd.concat([train, train_category.drop('instance_id', axis=1)], axis=1))
test = pd.DataFrame(pd.concat([test, test_category.drop('instance_id', axis=1)], axis=1))


# ================================================
#               basic count
# ================================================

group_keys = ['user_id', 'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']

cnt_cols = ['instance_id', 'item_id', 'item_brand_id', 'item_city_id', 'shop_id', 'item_category_1']
stat_cols = ['item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level',
             'shop_review_num_level', 'shop_review_positive_rate', 'shop_star_level', 'shop_score_service',
             'shop_score_delivery', 'shop_score_description', 'context_page_id']
item_prop = ['item_property_{}'.format(i) for i in range(100)]

concat = train.append(test)
print('Concat shape:', concat.shape)

for grp in group_keys:
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

    category_tmp = concat[[grp] + ['item_property_{}'.format(i) for i in range(100)]]
    category_tmp = category_tmp.sort_values([grp]).set_index(grp).stack().reset_index()
    category_tmp.columns = [grp, 'item_level', 'item_property']
    cnt_result['{}_property_cnt'.format(grp)] = category_tmp.groupby(grp)['item_property'].nunique()

    results = pd.concat([cnt_result, statistics_results], axis=1).reset_index()
    concat = concat.merge(results, how='left', on=grp)

gender_cnt = concat.groupby(['user_gender_id']).user_id.nunique().to_frame()
gender_cnt.columns = ['user_gender_id_user_id_cnt']
concat = concat.merge(gender_cnt, how='left', on='user_gender_id')

age_cnt = concat.groupby(['user_age_level']).user_id.nunique().to_frame()
age_cnt.columns = ['user_age_level_user_id_cnt']
concat = concat.merge(age_cnt, how='left', on='user_age_level')

occupation_cnt = concat.groupby(['user_occupation_id']).user_id.nunique().to_frame()
occupation_cnt.columns = ['user_occupation_id_user_id_cnt']
concat = concat.merge(occupation_cnt, how='left', on='user_occupation_id')

print('Finish basic count, Concat shape:', concat.shape)

# ================================================
#                saving
# ================================================

feature_columns = [col for col in list(concat) if
                   col.endswith(('cnt', 'min', 'median', 'mean', 'max', 'std', 'skew'))]

concat[:len(train)][['instance_id'] + feature_columns].to_pickle('../features/train_test/train_feature_101.p')
concat[len(train):][['instance_id'] + feature_columns].to_pickle('../features/train_test/test_feature_101.p')
