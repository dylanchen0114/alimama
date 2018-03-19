# -*- coding: utf-8 -*-

"""
@author: Dylan Chen

"""

from glob import glob

import pandas as pd
from tqdm import tqdm


def read_pickles(path, index):

    df = pd.concat([pd.read_pickle(f).set_index(index) for f in
                    tqdm(sorted(glob(path)))], axis=1)

    return df


train = pd.read_pickle('../processed/train_test/train_id_processed.p')
test = pd.read_pickle('../processed/train_test/test_id_processed.p')

concat_category = pd.read_pickle('../processed/concat_item_category.p')
concat_property = pd.read_pickle('../processed/concat_item_property.p')

# concat features
train_feats = read_pickles('../features/train_feature_*.p', 'instance_id').reset_index()
test_feats = read_pickles('../features/test_feature_*.p', 'instance_id').reset_index()


# feature 303
tmp_feat = pd.read_pickle('../features/concat_category_feature_303.p').reset_index()
tmp_feat = concat_category.drop(['item_category_0', 'item_category_2'], axis=1).\
    merge(tmp_feat, how='left', on='item_category_1').\
    drop(['item_category_1'], axis=1)

train_feats = train_feats.merge(tmp_feat, how='left', on='instance_id')
test_feats = test_feats.merge(tmp_feat, how='left', on='instance_id')

# category feature 306
tmp_feat = pd.read_pickle('../features/concat_cate_till_now_cnt_feature_306.p')

train_feats = train_feats.merge(tmp_feat.drop(['item_category_0', 'item_category_1', 'item_category_2'], axis=1),
                                how='left', on='instance_id')
test_feats = test_feats.merge(tmp_feat.drop(['item_category_0', 'item_category_1', 'item_category_2'], axis=1),
                              how='left', on='instance_id')

# property feature 306
tmp_feat = pd.read_pickle('../features/concat_prop_till_now_cnt_feature_306.p')
tmp_feat = tmp_feat.groupby('instance_id')['prop_till_now_cnt'].\
    agg({'mean', 'max', 'min', 'median'}).\
    add_prefix('prop_till_now_cnt_').\
    reset_index()

train_feats = train_feats.merge(tmp_feat, how='left', on='instance_id')
test_feats = test_feats.merge(tmp_feat, how='left', on='instance_id')


# category feature 304
tmp_feat = pd.read_pickle('../features/concat_cate_smt_ctr_feature_304.p')

concat = train[['instance_id', 'context_date_day']].append(test[['instance_id', 'context_date_day']])
tmp = concat[['instance_id', 'context_date_day']].merge(concat_category[['instance_id', 'item_category_1']],
                                                        how='left', on='instance_id')

tmp_feat = tmp.merge(tmp_feat, how='left', on=['item_category_1', 'context_date_day'])

train_feats = train_feats.merge(tmp_feat.drop(['context_date_day', 'item_category_1'], axis=1),
                                how='left', on='instance_id')
test_feats = test_feats.merge(tmp_feat.drop(['context_date_day', 'item_category_1'], axis=1),
                              how='left', on='instance_id')


# property feature 304
tmp_feat = pd.read_pickle('../features/concat_prop_smt_ctr_feature_304.p')

concat = train[['instance_id', 'context_date_day']].append(test[['instance_id', 'context_date_day']])
tmp = concat[['instance_id', 'context_date_day']].merge(concat_property, how='left', on='instance_id')

tmp_feat = tmp.merge(tmp_feat, how='left', on=['item_property', 'context_date_day'])


tmp_feat = tmp_feat.groupby(['instance_id', 'context_date_day'])['item_property_all', 'item_property_1',
                                                                 'item_property_smooth'].\
    agg({'mean', 'min', 'max', 'median'})
tmp_feat.columns = ['_'.join(col).strip() for col in tmp_feat.columns.values]
tmp_feat = tmp_feat.reset_index()

train_feats = train_feats.merge(tmp_feat.drop('context_date_day', axis=1), how='left', on='instance_id')
test_feats = test_feats.merge(tmp_feat.drop('context_date_day', axis=1), how='left', on='instance_id')


# saving
train_feats.to_csv('../features/all/train_concat_all.csv', index=None)
test_feats.to_csv('../features/all/test_concat_all.csv', index=None)
