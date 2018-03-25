# -*- coding: utf-8 -*-

"""
@author: Dylan Chen

"""

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import svds

train = pd.read_pickle('../processed/train_test/train_id_processed.p')
test = pd.read_pickle('../processed/train_test/test_id_processed.p')

concat = train[['user_id', 'item_id']].append(test[['user_id', 'item_id']])

user_cnt = concat['user_id'].max() + 1
item_cnt = concat['item_id'].max() + 1

data = np.ones(len(concat))
user_id = concat['user_id'].values
item_id = concat['item_id'].values


rating = sparse.coo_matrix((data, (user_id, item_id)))
rating = (rating > 0) * 1.0

n_component = 20

[u, s, vt] = svds(rating, k=n_component)
print(s[::-1])
s_item = np.diag(s[::-1])

