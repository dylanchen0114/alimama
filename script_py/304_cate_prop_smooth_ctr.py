# -*- coding: utf-8 -*-

"""
@author: Dylan Chen

"""

import pandas as pd
from function_utils import BayesianSmoothing
from tqdm import tqdm

columns = ['instance_id', 'context_date_day']
train = pd.read_pickle('../processed/train_test/train_id_processed.p')[columns + ['is_trade']]
test = pd.read_pickle('../processed/train_test/test_id_processed.p')[columns]

concat_category = pd.read_pickle('../processed/concat_item_category.p')
concat_property = pd.read_pickle('../processed/concat_item_property.p')


# ================================================
#               category part
# ================================================

tmp_category = train.merge(concat_category, how='left', on='instance_id')

res = pd.DataFrame()
temp = tmp_category[['item_category_1', 'context_date_day', 'is_trade']]

for day in tqdm(range(18, 26)):
    count = temp.groupby(['item_category_1']).apply(lambda x: x['is_trade'][(x['context_date_day'] < day).values].count()).\
        reset_index(name='item_category_1' + '_all')
    count1 = temp.groupby(['item_category_1']).apply(lambda x: x['is_trade'][(x['context_date_day'] < day).values].sum()).\
        reset_index(name='item_category_1' + '_1')
    count['item_category_1' + '_1'] = count1['item_category_1' + '_1']

    # TODO: should handle first day conversion count and sum ?
    count.fillna(value=0, inplace=True)
    count['context_date_day'] = day
    res = res.append(count, ignore_index=True)

print('smoothing category_id')
bs = BayesianSmoothing(1, 1)
bs.update(res['item_category_1' + '_all'].values, res['item_category_1' + '_1'].values, 1000, 0.001)
res['item_category_1' + '_smooth'] = (res['item_category_1' + '_1'] + bs.alpha) / \
                                     (res['item_category_1' + '_all'] + bs.alpha + bs.beta)

# conversion rate
res['item_category_1' + '_rate'] = res['item_category_1' + '_1'] / res['item_category_1' + '_all']

res.to_pickle('../features/concat_cate_smt_ctr_feature_304.p')


# ================================================
#               property part
# ================================================

tmp_property = train.merge(concat_property, how='left', on='instance_id')

res = pd.DataFrame()
temp = tmp_property[['item_property', 'context_date_day', 'is_trade']]

for day in tqdm(range(18, 26)):
    count = temp.groupby(['item_property']).apply(lambda x: x['is_trade'][(x['context_date_day'] < day).values].count()).\
        reset_index(name='item_property' + '_all')
    count1 = temp.groupby(['item_property']).apply(lambda x: x['is_trade'][(x['context_date_day'] < day).values].sum()).\
        reset_index(name='item_property' + '_1')
    count['item_property' + '_1'] = count1['item_property' + '_1']

    # TODO: should handle first day conversion count and sum ?
    count.fillna(value=0, inplace=True)
    count['context_date_day'] = day
    res = res.append(count, ignore_index=True)

print('smoothing item_property')
bs = BayesianSmoothing(1, 1)
bs.update(res['item_property' + '_all'].values, res['item_property' + '_1'].values, 1000, 0.001)
res['item_property' + '_smooth'] = (res['item_property' + '_1'] + bs.alpha) / \
                                   (res['item_property' + '_all'] + bs.alpha + bs.beta)

# conversion rate
res['item_property' + '_rate'] = res['item_property' + '_1'] / res['item_property' + '_all']

res.to_pickle('../features/concat_prop_smt_ctr_feature_304.p')