# -*- coding: utf-8 -*-

"""
@author: Dylan Chen

"""

import datetime
import gc
from glob import glob

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from tqdm import tqdm


def concat_features(path):

    concat = pd.concat([pd.read_pickle(file) for file in sorted(glob(path))], axis=1).reset_index()

    return concat


train_concat = concat_features('../features/train*').drop(['gender', 'age'], axis=1)
test_concat = concat_features('../features/test*').drop(['gender', 'age'], axis=1)

label = 'age'

tr_label = pd.read_pickle('../proc/train_label.pkl')
te_label = pd.read_pickle('../proc/test_label.pkl')

meta_feats = pd.read_csv(f'../output/agg_meta_{label}_score.csv')
meta_feats.columns = ['user_id'] + [f'{label}_meta_{col}' for col in list(meta_feats)[1:]]

tr_label = tr_label[['user_id', label]].merge(meta_feats, how='left').merge(train_concat, how='left')
te_label = te_label[['user_id', label]].merge(meta_feats, how='left').merge(test_concat, how='left')

train_y = tr_label[label].values - 1
train_user_id = tr_label[['user_id']].copy()
# train_x = tr_label.drop([label] + ['user_id'], axis=1)

test_user_id = te_label[['user_id']].copy()
# test_x = te_label[list(train_x)]

train_click = pd.read_csv('../input/train_preliminary/click_log.csv')
test_click = pd.read_csv('../input/test/click_log.csv')

tr_ad_prop = pd.read_csv('../input/train_preliminary/ad.csv')
tr_ad_prop = tr_ad_prop.replace("\\N", np.nan)

te_ad_prop = pd.read_csv('../input/test/ad.csv')
te_ad_prop = te_ad_prop.replace("\\N", np.nan)

ad_prop = tr_ad_prop.append(te_ad_prop)
ad_prop = ad_prop.drop_duplicates('creative_id').reset_index(drop=True)

print('Merging creative_id property ...')
train_click = train_click.merge(ad_prop, how='left', on=['creative_id'])
test_click = test_click.merge(ad_prop, how='left', on=['creative_id'])

train_click = train_click.merge(tr_label[['user_id', label]], how='left', on=['user_id'])

del tr_ad_prop, te_ad_prop, ad_prop, train_concat, test_concat
gc.collect()

# print('converting to categorical features ... ')
# embedding_columns = ['creative_id', 'ad_id', 'product_id', 'product_category', 'advertiser_id', 'industry']
# for col in embedding_columns:
#     train_x[col] = train_x[col].astype('category')
#     test_x[col] = test_x[col].astype('category')

# print(test_x.head(5))

num_folds = 5
k_fold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=82)
kf = k_fold.split(tr_label, train_y)
num_boost_round = 10000

validate_fold_score = []
train_fold_score = []
evals_result = {}

train_predict = np.zeros(len(tr_label))
test_predict = np.zeros([len(te_label), num_folds])


def mean_encode(train, val, features_to_encode, target, drop=False):
    train_encode = train.copy(deep=True)
    val_encode = val.copy(deep=True)
    for feature in features_to_encode:

        # out-fold train click log
        tmp_train_click = train_click[train_click.user_id.isin(train.user_id.values)].copy()

        # global target mean of out-fold train user
        train_global_mean = tmp_train_click.drop_duplicates(['user_id'])[target].mean()

        train_encode_map = pd.DataFrame(index=tmp_train_click[feature].unique())

        for fun in ['mean', 'max', 'min', 'std']:
            train_encode[f'user_{feature}_{target}_mean' + f'_{fun}'] = np.nan

        kf_tmp = KFold(n_splits=5, shuffle=False, random_state=82)
        for rest, this in tqdm(kf_tmp.split(train)):
            # in-fold rest click log
            tmp_train_click_rest = tmp_train_click[tmp_train_click.user_id.isin(train.iloc[rest]['user_id'].values)].copy()

            # global target mean of in-fold rest user
            train_rest_global_mean = tmp_train_click_rest.drop_duplicates(['user_id'])[target].mean()

            # item_mean of in-fold rest click log
            rest_encode_map = tmp_train_click_rest.groupby(feature)[target].mean().to_frame(f'{feature}_{target}_mean')

            # encoded_feature = train.iloc[this][feature].map(encode_map).values

            # in-fold this click log
            tmp_train_click_this = tmp_train_click[tmp_train_click.user_id.isin(train.iloc[this]['user_id'].values)].copy()
            encode_map = tmp_train_click_this[['user_id', feature]].merge(rest_encode_map.reset_index(), how='left', on=feature)
            encode_map[f'{feature}_{target}_mean'].fillna(train_rest_global_mean, inplace=True)

            for fun in ['mean', 'max', 'min', 'std']:
                encode_map_new = encode_map.groupby(['user_id'])[f'{feature}_{target}_mean'].agg(fun)
                train_encode[f'user_{feature}_{target}_mean' + f'_{fun}'].iloc[this] =\
                    train['user_id'].iloc[this].map(encode_map_new).values

            train_encode_map = pd.concat((train_encode_map, rest_encode_map), axis=1, sort=False)
            train_encode_map.fillna(train_rest_global_mean, inplace=True)

        train_encode_map['avg'] = train_encode_map.mean(axis=1)
        train_encode_map = train_encode_map[['avg']].reset_index()
        train_encode_map.columns = [feature, 'avg']

        total_click = train_click[['user_id', feature]].append(test_click[['user_id', feature]])

        tmp_val_click = total_click[total_click.user_id.isin(val.user_id.values)].copy()
        encode_map = tmp_val_click[['user_id', feature]].merge(train_encode_map, how='left', on=feature)
        encode_map['avg'].fillna(train_global_mean, inplace=True)
        encode_map.loc[pd.isna(encode_map[feature]), 'avg'] = np.nan

        for fun in ['mean', 'max', 'min', 'std']:
            encode_map_new = encode_map.groupby(['user_id'])['avg'].agg(fun)
            val_encode[f'user_{feature}_{target}_mean' + f'_{fun}'] = val['user_id'].map(encode_map_new)

    if drop:  # drop unencoded features
        train_encode.drop(features_to_encode, axis=1, inplace=True)
        val_encode.drop(features_to_encode, axis=1, inplace=True)
    return train_encode, val_encode


params = {
    'objective': 'multiclass',
    'metric': ['multi_error'],
    'boosting_type': 'gbdt',

    'num_classes': 10,

    'learning_rate': 0.05,
    'max_depth': 12,
    'num_leaves': 2 ** 6 - 1,
    # 'num_leaves': 2**4 -1,

    'min_child_weight': 10,
    'min_data_in_leaf': 40,
    # 'reg_lambda': 150,  # L2
    # 'reg_alpha': 120,  # L1

    'feature_fraction': 0.75,
    'subsample': 0.75,
    'seed': 114,

    'nthread': -1,
    'bagging_freq': 1,
    'verbose': -1,
}

excluded_feats = ['user_id', label]

for i, (train_fold, validate) in enumerate(kf):
    # X_train, X_validate, label_train, label_validate = \
    #     tr_label.loc[train_fold], tr_label.loc[validate], train_y[train_fold], train_y[validate]

    train_user, val_user = tr_label.loc[train_fold].copy(), tr_label.loc[validate].copy()

    val_test_user = pd.concat([val_user, te_label], axis=0, sort=False)
    val_size = val_user.shape[0]
    test_size = test_user_id.shape[0]

    print('Mean Encoding ...')
    trn, val_test = mean_encode(train_user, val_test_user,
                                ['creative_id', 'product_id', 'product_category', 'advertiser_id', 'industry'],
                                'age', drop=False)
    features = [f_ for f_ in trn.columns if f_ not in excluded_feats]
    print(features)

    val_x = val_test.iloc[0:val_size, :].copy(deep=True)
    test_x = val_test.iloc[-test_size:, :].copy(deep=True)

    X_train, label_train = trn[features], trn[label].values - 1
    X_validate, label_validate = val_x[features], val_x[label].values - 1

    dtrain = lgb.Dataset(X_train, label_train)
    dvalid = lgb.Dataset(X_validate, label_validate, reference=dtrain)

    gbm = lgb.train(params, dtrain, num_boost_round,
                    valid_sets=[dtrain, dvalid], valid_names=['train', 'valid'],
                    evals_result=evals_result, verbose_eval=50, early_stopping_rounds=50)

    bst_round = np.argmin(evals_result['valid']['multi_error'])
    trn_score = evals_result['train']['multi_error'][bst_round]
    val_score = evals_result['valid']['multi_error'][bst_round]

    print('predicting test ...')
    if i == 0:
        test_predict = gbm.predict(test_x[features])
        print(test_predict[:5])
    else:
        test_predict += gbm.predict(test_x[features]) / num_folds
        print(test_predict[:5])

#     print('predicting OOF ...')
#     train_predict[validate] = gbm.predict(X_validate)

    train_fold_score.append(trn_score)
    validate_fold_score.append(val_score)

    feature_importance = pd.DataFrame({'name': gbm.feature_name(), 'importance': gbm.feature_importance('gain')}).sort_values(
        by='importance', ascending=False)
    feature_importance.to_csv(f'../doc/cv_main_{label}_feat_importance_{i}.csv', index=False, encoding='gb18030')

train_score, valid_score = np.mean(train_fold_score), np.mean(validate_fold_score)

print('Training Score : %.5f, Validation Score : %.5f' % (train_score, valid_score))

res = '%s,%s,%d,%s,%.4f,%d,%d,%d,%.4f,%.4f,%d,%.4e,%s,%.5f,%.5f\n' % \
          (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
           'TE',
           len(list(X_train)), params['boosting_type'],
           params['learning_rate'], params['num_leaves'], params['max_depth'],
           params['min_data_in_leaf'], params['feature_fraction'], params['subsample'],
           params['bagging_freq'], 0.0, bst_round+1, 1 - train_score, 1 - valid_score)

with open(f'../{label}_lgb_record.csv', 'a') as f:
    f.write(res)

test_user_id[label] = np.argmax(test_predict, axis=1)
test_user_id[label] = test_user_id[label] + 1

print(test_user_id[label].value_counts(normalize=True))

train_user_df = pd.read_csv('../input/train_preliminary/user.csv')
print(train_user_df[label].value_counts(normalize=True))

test_user_id.to_csv(f'../submit/TE_{label}.csv', index=None)

# train_user_id[label] = train_predict
# train_user_id['target_truth'] = train_y


# def thr_to_accuracy(thr, y_test, predictions):
#     return -accuracy_score(y_test, np.array(predictions > thr, dtype=np.int))


# best_thr = fmin(thr_to_accuracy, args=(train_user_id.target_truth.values,
#                                        train_user_id.gender.values), x0=0.30)

# print(f"best_thr: {best_thr}")
# test_user_id['gender'] = (test_user_id['gender'] > best_thr[0]) * 1

