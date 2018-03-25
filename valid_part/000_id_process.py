# -*- coding: utf-8 -*-

"""
@author: Dylan Chen

"""

from datetime import datetime as dt

import pandas as pd
from sklearn.preprocessing import LabelEncoder


train = pd.read_csv('../input/round1_ijcai_18_train_20180301.txt', delimiter=' ')
test = pd.read_csv('../input/round1_ijcai_18_test_a_20180301.txt', delimiter=' ')

train = train.drop_duplicates('instance_id').copy()

instance_encoder = LabelEncoder()
instance_encoder.fit(train.instance_id)

train['instance_id'] = instance_encoder.transform(train.instance_id)

# ================================================
#             item pre-processing
# ================================================

# item id
item_encoder = LabelEncoder()
item_encoder.fit(train.item_id.append(test.item_id))

train['item_id'] = item_encoder.transform(train.item_id)
test['item_id'] = item_encoder.transform(test.item_id)

# item category
train_item_category = train.item_category_list.str.split(';', expand=True).add_prefix('item_category_')
test_item_category = test.item_category_list.str.split(';', expand=True).add_prefix('item_category_')

train_item_category.fillna('missing', inplace=True)
test_item_category.fillna('missing', inplace=True)

train.drop('item_category_list', axis=1, inplace=True)
test.drop('item_category_list', axis=1, inplace=True)

# # train 与 test中的类目一致，只有一种一级类目，从属的二级类目有13种，三级类目较少，只有两种
# print(train_item_category['item_category_0'].nunique())
# print(train_item_category['item_category_1'].nunique())

# item property
train_item_property = train['item_property_list'].str.split(';', expand=True).add_prefix('item_property_')
test_item_property = test['item_property_list'].str.split(';', expand=True).add_prefix('item_property_')

train_item_property.fillna('missing', inplace=True)
test_item_property.fillna('missing', inplace=True)

train.drop('item_property_list', axis=1, inplace=True)
test.drop('item_property_list', axis=1, inplace=True)

# item_brand_id and city id
for col in ['item_brand_id', 'item_city_id']:
    col_encoder = LabelEncoder()
    col_encoder.fit(train[col].append(test[col]))

    train[col] = col_encoder.transform(train[col])
    test[col] = col_encoder.transform(test[col])


# ================================================
#             user pre-processing
# ================================================

for col in ['user_id', 'user_occupation_id']:
    col_encoder = LabelEncoder()
    col_encoder.fit(train[col].append(test[col]))

    train[col] = col_encoder.transform(train[col])
    test[col] = col_encoder.transform(test[col])

print('train_user %i' % train['user_id'].nunique(), 'test_user %i' % test['user_id'].nunique())
print('intersection_user_count: ', len(set(test['user_id']).intersection((train['user_id']))))  # user_id 交集较少


# ================================================
#             shop pre-processing
# ================================================

for col in ['shop_id']:
    col_encoder = LabelEncoder()
    col_encoder.fit(train[col].append(test[col]))

    train[col] = col_encoder.transform(train[col])
    test[col] = col_encoder.transform(test[col])

print('train_shop %i' % train['shop_id'].nunique(), 'test_shop %i' % test['shop_id'].nunique())
print('intersection_shop_count: ', len(set(test['shop_id']).intersection((train['shop_id']))))


# ================================================
#             context pre-processing
# ================================================

# create day and hour
train['context_date'] = train['context_timestamp'].map(dt.fromtimestamp)
test['context_date'] = test['context_timestamp'].map(dt.fromtimestamp)

train['context_date_day'] = train['context_date'].map(lambda x: int(str(x)[8:10]))
test['context_date_day'] = test['context_date'].map(lambda x: int(str(x)[8:10]))

train['context_date_hour'] = train['context_date'].map(lambda x: int(str(x)[11:13]))
test['context_date_hour'] = test['context_date'].map(lambda x: int(str(x)[11:13]))

# split predict category
train_context_category = train['predict_category_property'].str.split(';', expand=True).\
    add_prefix('predict_context_category_')
test_context_category = test['predict_category_property'].str.split(';', expand=True).\
    add_prefix('predict_context_category_')

train_context_category.fillna('missing', inplace=True)
test_context_category.fillna('missing', inplace=True)

# concat category, property and predict_context_category
train_category_with_instance_id = pd.concat([train.instance_id, train_item_category,
                                             train_item_property, train_context_category], axis=1)
test_category_with_instance_id = pd.concat([test.instance_id, test_item_category,
                                            test_item_property, test_context_category], axis=1)

train_category_with_instance_id.to_pickle('../processed/train_category_with_instance_id.p')
test_category_with_instance_id.to_pickle('../processed/test_category_with_instance_id.p')

train.drop('predict_category_property', axis=1, inplace=True)
test.drop('predict_category_property', axis=1, inplace=True)


# ================================================
#                    saving
# ================================================

train.to_pickle('../processed/train_test/train_id_processed.p')
test.to_pickle('../processed/train_test/test_id_processed.p')
