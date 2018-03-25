# -*- coding: utf-8 -*-

"""
@author: Dylan Chen

"""

from glob import glob

import pandas as pd
from tqdm import tqdm


def read_pickles(path, col=None):
    if col is None:
        df = pd.concat([pd.read_pickle(f) for f in tqdm(sorted(glob(path+'/*')))])
    else:
        df = pd.concat([pd.read_pickle(f)[col] for f in tqdm(sorted(glob(path+'/*')))])
    return df


def to_pickles(df, path, split_size=3):
    """
    path = '../output/mydf'

    wirte '../output/mydf/0.p'
          '../output/mydf/1.p'
          '../output/mydf/2.p'

    """

    for i in tqdm(range(split_size)):
        df.ix[df.index % split_size == i].to_pickle(path + '/{}.p'.format(i))


train = pd.read_pickle('../processed/train_test/train_id_processed.p')
test = pd.read_pickle('../processed/train_test/test_id_processed.p')

train_feats = read_pickles('../features/all/train_test/train/')
train_feats.drop(['context_date_day', 'context_date'], axis=1, inplace=True)
test_feats = read_pickles('../features/all/train_test/test/')
test_feats.drop(['context_date_day', 'context_date'], axis=1, inplace=True)

train = train.merge(train_feats, how='left', on='instance_id')
test = test.merge(test_feats, how='left', on='instance_id')

test_id = test['instance_id']
drop_columns = ['instance_id', 'context_id', 'context_date',
                'context_date_day', 'context_date_hour', 'context_timestamp', 'item_category_1',
                'item_id', 'item_brand_id', 'item_city_id', 'user_id', 'shop_id']
train.drop(drop_columns, axis=1, inplace=True)
test.drop(drop_columns, axis=1, inplace=True)

embedding_features = ['user_gender_id', 'user_occupation_id']

for col in embedding_features:
    train[col] = train[col].astype('category')
    test[col] = test[col].astype('category')

train_y = train['is_trade'].reset_index(drop=True).copy()
train = train.reset_index(drop=True).drop('is_trade', axis=1)

train_y.to_pickle('../processed/train_test/train_y.p')
to_pickles(df=train, path='../processed/train_test/train_x/')

test_id.to_pickle('../processed/train_test/test_id.p')
test.to_pickle('../processed/train_test/test_x.p')
