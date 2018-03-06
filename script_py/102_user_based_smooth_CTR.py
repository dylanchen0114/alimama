# -*- coding: utf-8 -*-

"""
@author: Dylan Chen

"""

import random

import numpy as np
import pandas as pd
import scipy.special as special


class BayesianSmoothing(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def sample(self, alpha, beta, num, imp_upperbound):
        sample = np.random.beta(alpha, beta, num)
        I = []
        C = []
        for clk_rt in sample:
            imp = random.random() * imp_upperbound
            imp = imp_upperbound
            clk = imp * clk_rt
            I.append(imp)
            C.append(clk)
        return I, C

    def update(self, imps, clicks, iter_num, epsilon):
        for i in range(iter_num):
            new_alpha, new_beta = self._fixed_point_iter(imps, clicks, self.alpha, self.beta)
            if abs(new_alpha - self.alpha) < epsilon and abs(new_beta - self.beta) < epsilon:
                break
            self.alpha = new_alpha
            self.beta = new_beta

    def _fixed_point_iter(self, imps, clicks, alpha, beta):
        numerator_alpha = 0.0
        numerator_beta = 0.0
        denominator = 0.0

        for i in range(len(imps)):
            numerator_alpha += (special.digamma(clicks[i] + alpha) - special.digamma(alpha))
            numerator_beta += (special.digamma(imps[i] - clicks[i] + beta) - special.digamma(beta))
            denominator += (special.digamma(imps[i] + alpha + beta) - special.digamma(alpha + beta))

        return alpha * (numerator_alpha / denominator), beta * (numerator_beta / denominator)


columns = ['user_id', 'user_gender_id', 'user_age_level', 'user_occupation_id',
           'user_star_level', 'context_date_day']
train = pd.read_pickle('../processed/train_test/train_id_processed.p')[columns + ['is_trade']]
test = pd.read_pickle('../processed/train_test/test_id_processed.p')[columns]

for feat_1 in ['user_id', 'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']:

    res = pd.DataFrame()
    temp = train[[feat_1, 'context_date_day', 'is_trade']]

    for day in range(18, 25):
        count = temp.groupby([feat_1]).apply(lambda x: x['is_trade'][(x['context_date_day'] < day).values].count()).\
            reset_index(name=feat_1 + '_all')
        count1 = temp.groupby([feat_1]).apply(lambda x: x['is_trade'][(x['context_date_day'] < day).values].sum()).\
            reset_index(name=feat_1 + '_1')
        count[feat_1 + '_1'] = count1[feat_1 + '_1']
        # TODO: should handle first day conversion count and sum ?
        count.fillna(value=0, inplace=True)
        count['context_date_day'] = day
        res = res.append(count, ignore_index=True)

    # only smooth user_id here, cause user_id has a high cardinality
    if feat_1 == 'user_id':
        print('smoothing user_id')
        bs = BayesianSmoothing(1, 1)
        bs.update(res[feat_1 + '_all'].values, res[feat_1 + '_1'].values, 1000, 0.001)
        res[feat_1 + '_smooth'] = (temp[feat_1 + '_1'] + bs.alpha) / (temp[feat_1 + '_all'] + bs.alpha + bs.beta)

    # all features conversion rate
    res[feat_1 + '_rate'] = res[feat_1 + '_1'] / res[feat_1 + '_all']

    train = train.merge(res, how='left', on=[feat_1, 'context_date_day'])
    test = test.merge(res, how='left', on=[feat_1, 'context_date_day'])

    if feat_1 == 'user_id':
        train['user_id_smooth'] = train['user_id_smooth'].fillna(value=bs.alpha / (bs.alpha + bs.beta))
        test['user_id_smooth'] = test['user_id_smooth'].fillna(value=bs.alpha / (bs.alpha + bs.beta))

    train[feat_1 + '_rate'] = train[feat_1 + '_rate'].fillna(value=0)
    test[feat_1 + '_rate'] = test[feat_1 + '_rate'].fillna(value=0)


# ================================================
#                saving
# ================================================

feature_columns = [col for col in list(train) if
                   col.endswith(('_1', '_all', '_smooth', '_rate'))]

train[['instance_id'] + feature_columns].to_pickle('../features/train_test/train_feature_102.p')
test[['instance_id'] + feature_columns].to_pickle('../features/train_test/test_feature_102.p')
