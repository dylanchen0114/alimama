# -*- coding: utf-8 -*-

"""
@author: Dylan Chen

"""

import gc

import numpy as np
import pandas as pd

train_cate = pd.read_pickle('../processed/train_category_with_instance_id.p')
test_cate = pd.read_pickle('../processed/test_category_with_instance_id.p')

concat = train_cate.append(test_cate)


# ================================================
#      transpose item category and property
# ================================================

concat.replace('missing', np.nan, inplace=True)

# category
concat_category = concat[['instance_id'] + ['item_category_{}'.format(i) for i in range(3)]].copy()
concat_category.to_pickle('../processed/concat_item_category.p')

# property
concat_property = concat[['instance_id'] + ['item_property_{}'.format(i) for i in range(100)]].copy()

concat_property_T = concat_property.set_index('instance_id').stack().reset_index()
concat_property_T = concat_property_T.rename(columns={0: 'item_property'}).drop(['level_1'], axis=1)

concat_property_T.to_pickle('../processed/concat_item_property.p')

del concat_category, concat_property, concat_property_T
gc.collect()


# ================================================
#  transpose item predicted category and property
# ================================================

concat_predict = concat[['instance_id'] + [col for col in list(concat) if col.startswith('predict_')]].copy()

concat_predict_T = concat_predict.set_index('instance_id').stack().reset_index()
concat_predict_T = concat_predict_T.rename(columns={0: 'item_category_property'}).drop(['level_1'], axis=1)

split_df = concat_predict_T['item_category_property'].str.split(':', expand=True)
split_df.columns = ['predict_item_category', 'predict_item_property']

concat_predict_T = pd.concat([concat_predict_T.drop('item_category_property', axis=1), split_df], axis=1)
concat_predict_T = pd.concat([concat_predict_T.drop(['predict_item_property'], axis=1),
                              concat_predict_T.predict_item_property.str.split(',', expand=True)], axis=1)

concat_predict_T = concat_predict_T.set_index(['instance_id', 'predict_item_category'])
concat_predict_T = concat_predict_T.stack().reset_index()

concat_predict_T = concat_predict_T.drop(['level_2'], axis=1).rename(columns={0: 'predict_item_property'})
concat_predict_T.to_pickle('../processed/concat_predict_item_category_property.p')
