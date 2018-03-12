# -*- coding: utf-8 -*-

"""
@author: Dylan Chen

"""

import pandas as pd
from function_utils import BayesianSmoothing
from tqdm import tqdm

columns = ['instance_id', 'item_id', 'item_brand_id', 'item_city_id',
           'item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level',
           'context_date_day']
train = pd.read_pickle('../processed/train_test/train_id_processed.p')[columns + ['is_trade']]
test = pd.read_pickle('../processed/train_test/test_id_processed.p')[columns]

for feat_1 in ['item_city_id', 'item_price_level', 'item_sales_level',
               'item_collected_level', 'item_pv_level', 'item_id', 'item_brand_id']:

    print(feat_1)

    res = pd.DataFrame()
    temp = train[[feat_1, 'context_date_day', 'is_trade']]

    for day in tqdm(range(18, 26)):
        count = temp.groupby([feat_1]).apply(lambda x: x['is_trade'][(x['context_date_day'] < day).values].count()).\
            reset_index(name=feat_1 + '_all')
        count1 = temp.groupby([feat_1]).apply(lambda x: x['is_trade'][(x['context_date_day'] < day).values].sum()).\
            reset_index(name=feat_1 + '_1')
        count[feat_1 + '_1'] = count1[feat_1 + '_1']
        # TODO: should handle first day conversion count and sum ?
        count.fillna(value=0, inplace=True)
        count['context_date_day'] = day
        res = res.append(count, ignore_index=True)

    # only smooth item_id and item_brand_id here
    if feat_1 == 'item_id':
        print('smoothing item_id')
        bs_item = BayesianSmoothing(1, 1)
        bs_item.update(res[feat_1 + '_all'].values, res[feat_1 + '_1'].values, 1000, 0.001)
        res[feat_1 + '_smooth'] = (res[feat_1 + '_1'] + bs_item.alpha) / \
                                  (res[feat_1 + '_all'] + bs_item.alpha + bs_item.beta)

    if feat_1 == 'item_brand_id':
        print('smoothing item_brand_id')
        bs_brand = BayesianSmoothing(1, 1)
        bs_brand.update(res[feat_1 + '_all'].values, res[feat_1 + '_1'].values, 1000, 0.001)
        res[feat_1 + '_smooth'] = (res[feat_1 + '_1'] + bs_brand.alpha) / \
                                  (res[feat_1 + '_all'] + bs_brand.alpha + bs_brand.beta)

    # all features conversion rate
    res[feat_1 + '_rate'] = res[feat_1 + '_1'] / res[feat_1 + '_all']

    train = train.merge(res, how='left', on=[feat_1, 'context_date_day'])
    test = test.merge(res, how='left', on=[feat_1, 'context_date_day'])

    if feat_1 == 'item_id':
        train['item_id_smooth'] = train['item_id_smooth'].fillna(value=bs_item.alpha / (bs_item.alpha + bs_item.beta))
        test['item_id_smooth'] = test['item_id_smooth'].fillna(value=bs_item.alpha / (bs_item.alpha + bs_item.beta))

    if feat_1 == 'item_brand_id':
        train['item_brand_id_smooth'] = train['item_brand_id_smooth'].fillna(value=bs_brand.alpha / (bs_brand.alpha + bs_brand.beta))
        test['item_brand_id_smooth'] = test['item_brand_id_smooth'].fillna(value=bs_brand.alpha / (bs_brand.alpha + bs_brand.beta))

    train[feat_1 + '_rate'] = train[feat_1 + '_rate'].fillna(value=0)
    test[feat_1 + '_rate'] = test[feat_1 + '_rate'].fillna(value=0)


# ================================================
#                saving
# ================================================

feature_columns = [col for col in list(train) if
                   col.endswith(('_1', '_all', '_smooth', '_rate'))]

train[['instance_id'] + feature_columns].to_pickle('../features/train_feature_302.p')
test[['instance_id'] + feature_columns].to_pickle('../features/test_feature_302.p')
