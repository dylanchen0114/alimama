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

    return

train = pd.read_pickle('../processed/train_valid/train_id_processed.p')
test = pd.read_pickle('../processed/train_valid/test_id_processed.p')

train_feats = read_pickles('../features/all/train_valid/train/')
test_feats = read_pickles('../features/all/train_valid/test/')

train_feats.drop(['context_date_day', 'context_date'], axis=1, inplace=True)
test_feats.drop(['context_date_day', 'context_date'], axis=1, inplace=True)


train = train.merge(train_feats, how='left', on='instance_id')
test = test.merge(test_feats, how='left', on='instance_id')


drop_columns = ['instance_id', 'context_id', 'context_date',
                'context_date_day', 'context_date_hour', 'context_timestamp',
                'item_category_1', 'item_id', 'item_brand_id', 'item_city_id',
                'user_id', 'shop_id']

train.drop(drop_columns, axis=1, inplace=True)
test.drop(drop_columns, axis=1, inplace=True)

embedding_features = ['user_gender_id', 'user_occupation_id']

for col in embedding_features:
    train[col] = train[col].astype('category')
    test[col] = test[col].astype('category')


train_y = train['is_trade'].reset_index(drop=True).copy()
train = train.reset_index(drop=True).drop('is_trade', axis=1)

valid_y = test['is_trade'].reset_index(drop=True).copy()
valid = test.reset_index(drop=True).drop('is_trade', axis=1)

train_y.to_pickle('../processed/train_valid/train_y.p')
to_pickles(df=train, path='../processed/train_valid/train_x/')

valid_y.to_pickle('../processed/train_valid/valid_y.p')
valid.to_pickle('../processed/train_valid/valid_x.p')
