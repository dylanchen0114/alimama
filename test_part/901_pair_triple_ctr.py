# -*- coding: utf-8 -*-

"""
@author: Dylan Chen

"""

import pandas as pd
from tqdm import tqdm

train = pd.read_pickle('../processed/train_test/train_id_processed.p')
test = pd.read_pickle('../processed/train_test/test_id_processed.p')

concat_category = pd.read_pickle('../processed/concat_item_category.p')

train = train.merge(concat_category, how='left', on='instance_id')
test = test.merge(concat_category, how='left', on='instance_id')


for feat_1, feat_2 in [('item_id', 'user_gender_id'),
                       ('shop_id', 'item_id'), ('shop_id', 'item_brand_id'),
                       ('shop_id', 'user_gender_id'),
                       ('user_id', 'item_category_1'), ('shop_id', 'item_category_1'),
                       ('user_age_level', 'item_category_1'), ('user_gender_id', 'item_category_1'),
                       ('user_id', 'item_id'), ('user_id', 'item_brand_id')]:

    print(feat_1, feat_2)

    res = pd.DataFrame()
    temp = train[[feat_1, feat_2, 'context_date_day', 'is_trade']]

    for day in tqdm(range(18, 26)):
        count = temp.groupby([feat_1, feat_2]).apply(lambda x: x['is_trade'][(x['context_date_day'] < day).values].count()).\
            reset_index(name=feat_1 + '_' + feat_2 + '_all')
        count1 = temp.groupby([feat_1, feat_2]).apply(lambda x: x['is_trade'][(x['context_date_day'] < day).values].sum()). \
            reset_index(name=feat_1 + '_' + feat_2 + '_1')
        count[feat_1 + '_' + feat_2 + '_1'] = count1[feat_1 + '_' + feat_2 + '_1']
        # TODO: should handle first day conversion count and sum ?
        count.fillna(value=0, inplace=True)
        count['context_date_day'] = day
        res = res.append(count, ignore_index=True)
    print(feat_1, feat_2, ' over')

    # all features conversion rate
    # res[feat_1 + '_rate'] = res[feat_1 + '_1'] / res[feat_1 + '_all']

    train = train.merge(res, how='left', on=[feat_1, feat_2,  'context_date_day'])
    test = test.merge(res, how='left', on=[feat_1, feat_2, 'context_date_day'])

    # train[feat_1 + '_rate'] = train[feat_1 + '_rate'].fillna(value=0)
    # test[feat_1 + '_rate'] = test[feat_1 + '_rate'].fillna(value=0)


# ================================================
#                saving
# ================================================

feature_columns = [col for col in list(train) if
                   col.endswith(('_1', '_all'))]

train[['instance_id'] + feature_columns].to_pickle('../features/train_test/train_feature_901.p')
test[['instance_id'] + feature_columns].to_pickle('../features/train_test/test_feature_901.p')
