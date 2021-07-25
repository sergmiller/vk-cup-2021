import argparse
import os
import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsRegressor

from collections import defaultdict

import catboost

from fast_pagerank import pagerank_power
from scipy.sparse import csr_matrix


GROUP_ALS_DIM = 32
FRIEND_ALS_DIM = 16
ALS_TRAIN_GROUPS_TH = 1  # 5 is worse that 1
GROUP_OVER_FRIEND_WEIGHT = 0.75
ALS_OVER_NAIVE_WEIGHT = 0.5
# NAIVE_OVER_CB_WEIGHT = 0
# FRIEND_ALS_OVER_CB_WEIGHT = 0
FULL_ALS_OVER_CB_WEIGHT = 0.5
FULL_ALS_OVER_BASE_WEIGHT = 1

CB_V1_FEATURE_SLICE = list(np.arange(8 + 8 + 2 + 1 + 16))
CB_V2_FEATURE_SLICE = list(np.arange(8 + 8 + 2 + 1)) + list(np.arange(FRIEND_ALS_DIM) + 8 + 8 + 2 + 1 + 16)
CB_V3_FEATURE_SLICE = CB_V2_FEATURE_SLICE + list(np.arange(GROUP_ALS_DIM) + 8 + 8 + 2 + 1 + 16  + FRIEND_ALS_DIM)
  # + FRIEND_ALS_DIM # second friend_als_dim for firends mean user embedding


### SET VARIABLES

TEST = None
EDUCATION = None
GROUPS = None
FRIENDS = "friends.csv"

def set_global(is_train: bool):
    global TEST, EDUCATION, GROUPS
    env = "train" if is_train else "test"
    TEST = "{}.csv".format(env)
    EDUCATION = "{}EducationFeatures.csv".format(env)
    GROUPS = "{}Groups.csv".format(env)

### END SET VARIABLES


def read_csv(input: str) -> pd.DataFrame:
    return pd.read_csv(input)


### UPLOAD TRAIN DATA

train_uids = read_csv('data/train.csv')
train_als_user_embeddings = read_csv('data/train_als_user_embeddings.csv').values
train_als_group_embeddings = read_csv('data/train_als_group_embeddings.csv').values
train_groups = read_csv('data/train_groups.csv').values.reshape(-1)
train_groups_set = set(list(train_groups))
assert len(train_groups) == len(train_als_group_embeddings)
train_group_2_emb = {g: e for g,e in zip(train_groups, train_als_group_embeddings)}
group_knn = KNeighborsRegressor(weights='distance', n_neighbors=25)
group_knn.fit(train_als_user_embeddings, train_uids['age'].values)

friend_als_user_embeddings = read_csv('data/friends_als_user_embeddings.csv').values
train_friend_als_user_embeddings = read_csv('data/train_friends_als_user_embeddings.csv').values
train_friend_als_friend_embeddings = read_csv('data/train_friends_als_friends_embeddings.csv').values
train_friends = read_csv('data/train_friends.csv').values.reshape(-1)
train_friends_set = set(list(train_friends))
assert len(train_uids['uid']) == len(train_friend_als_user_embeddings)
train_friends_2_user_emb = {g: e for g,e in zip(train_uids['uid'], train_friend_als_user_embeddings)}
assert len(train_friends) == len(train_friend_als_friend_embeddings)
train_friends_2_emb = {g: e for g,e in zip(train_friends, train_friend_als_friend_embeddings)}
assert len(train_friends) == len(friend_als_user_embeddings)
friends_2_user_emb = {g: e for g,e in zip(train_friends, friend_als_user_embeddings)}
friend_knn = KNeighborsRegressor(weights='distance', n_neighbors=25)
friend_knn.fit(train_friend_als_user_embeddings, train_uids['age'].values)

user_g_nodes = read_csv('data/graph_nodes_user_ids.csv').values
gembeddings = read_csv('data/graph_embeddings.csv').values
uid2gembeddding = {int(uid): gembeddings[i] for i, uid in enumerate(user_g_nodes)}

edu_cb_model_v1 = catboost.CatBoost().load_model('data/edu_v1_g.cbm')
edu_cb_model_v2 = catboost.CatBoost().load_model('data/edu_v2_2_10K.cbm')
edu_cb_model_v3 = catboost.CatBoost().load_model('data/edu_v3_2_10K.cbm')

user_embs_for_friend_knn = None
user_embs_for_group_knn = None
ids_order = None

### END UPLOAD TRAIN DATA

def write_csv(df: pd.DataFrame, output: str):
    df.to_csv(output, index=None, index_label=None)


def filter_train_seq(groups: list, filter_set: set) -> list:
    filtred = []
    for g in groups:
        if g in filter_set:
            filtred.append(g)
    return filtred

def calc_user_embed_by_train_seq(groups: list, als_dim: int, entity2emb: dict) -> np.array:
    e = np.zeros(als_dim)
    if len(groups) == 0:
        return e
    for g in groups:
        e += entity2emb[g]
    return e / len(groups)


def calc_friends_pagerank(friends: pd.DataFrame) -> np.array:
    friends2way = pd.DataFrame()
    friends2way['uid'] = list(friends['uid']) + list(friends['fuid'])
    friends2way['fuid'] = list(friends['fuid']) + list(friends['uid'])
    friend_weights = friends2way.groupby('uid').fuid.nunique()
    friend2weight = {u:w for u,w in zip(friend_weights.index.values, friend_weights.values)}
    rowf = friends2way['fuid'].values
    colf = friends2way['uid'].values
    dataf = [friend2weight[u] for u in colf]
    train_f2u = csr_matrix((dataf, (rowf, colf)), shape=(max(rowf) + 1, max(colf) + 1))
    friend_pr=pagerank_power(train_f2u, p=0.85, tol=1e-3)
    return friend_pr

def decision(
    uid: int,
    school: float,
    g5: float,
    register: float,
    groups: list,
    friends: list,
    group_als_prediction: float,
    friend_als_prediction: float,
    cb_v1_prediction: float,
    cb_v2_prediction: float,
    cb_v3_prediction: float
) -> float:
    # naive = decision_naive_impl(school, register, g5)
    # base = naive * NAIVE_OVER_CB_WEIGHT + (1 - NAIVE_OVER_CB_WEIGHT) * cb_v1_prediction
    base = cb_v1_prediction

    groups = filter_train_seq(groups, train_groups_set)
    # friends = filter_train_seq(friends, train_friends_set)

    # friend_als_prediction = friend_als_prediction * FRIEND_ALS_OVER_CB_WEIGHT + (1 - FRIEND_ALS_OVER_CB_WEIGHT) * cb_v2_prediction

    if len(groups) < ALS_TRAIN_GROUPS_TH and len(friends) < ALS_TRAIN_GROUPS_TH:
        return base
    if len(groups) < ALS_TRAIN_GROUPS_TH:
        return cb_v1_prediction * ALS_OVER_NAIVE_WEIGHT + (1 - ALS_OVER_NAIVE_WEIGHT) * base
    full_als = group_als_prediction * GROUP_OVER_FRIEND_WEIGHT + (1 - GROUP_OVER_FRIEND_WEIGHT) * cb_v1_prediction
    full_als = full_als * FULL_ALS_OVER_CB_WEIGHT + (1 -  FULL_ALS_OVER_CB_WEIGHT) * cb_v3_prediction
    full_als = full_als * FULL_ALS_OVER_BASE_WEIGHT + (1 - FULL_ALS_OVER_BASE_WEIGHT) * base
    return full_als

def get_als_friends_embed(uid: int) -> np.array:
    return user_embs_for_friend_knn[ids_order[uid]]

def get_als_group_embed(uid: int) -> np.array:
    return user_embs_for_group_knn[ids_order[uid]]

f_cache = {}
def get_friends_mean_embed(uid: int, friends_list: list)-> np.array:
    if uid in f_cache:
        return f_cache[uid]
    e = np.zeros(16)
    count = 0
    for f in friends_list[uid]:
        m = friends_2_user_emb[f]
        e += m
        count += 1
    if count > 0:
        e /= count
    f_cache[uid] = e
    return e

def make_common_features_v3(uids: pd.DataFrame, edu: pd.DataFrame, friends_df: pd.DataFrame, groups: defaultdict, friends: defaultdict) -> tuple:
    friends_pr = calc_friends_pagerank(friends_df)
    edu_features = []
    edu_ids = edu['uid']
    uid2register = {u:r for u,r in zip(uids['uid'].values, uids['registered_year'].values)}
    for x in edu.iterrows():
        x = x[1]
        uid = int(x['uid'])
        get_2000 = lambda name: x[name] - 2000 if not np.isnan(x[name]) else 0
        make_ind = lambda name: float(np.isnan(x[name]))
        features_ind = [make_ind('school_education')]
        for i in range(1, 8):
            features_ind.append(make_ind('graduation_{}'.format(i)))
        features = [get_2000('school_education')]
        for i in range(1, 8):
            features.append(get_2000('graduation_{}'.format(i)))
        register_year = uid2register[uid]
        features.append(len(friends[uid]))
        features.append(len(groups[uid]))
        features.append(friends_pr[uid])
        # features.append(register_year)
        # features.append(decision_naive_impl(x['school_education'], register_year, x['graduation_5']))
        f = features_ind + features + list(uid2gembeddding.get(uid, np.zeros(16))) + list(get_als_friends_embed(uid)) + list(get_als_group_embed(uid))  # + list(get_friends_mean_embed(uid, friends))
        edu_features.append(f)
    return edu_ids, np.array(edu_features)

#
def decision_naive_impl(school: float, register: float, g5: float) -> float:
    if np.isnan(school):
        if np.isnan(g5):
            r = 697.208 - 0.32883 * register  # LM approx 35
        else:
            r = 1287.602 + 0.26818 * register - 0.89101 * g5  # LM approx register and graduation_5
    else:
        if np.isnan(g5):
            r = 1918.977 - 0.05583 * register - 0.88422 * school  # LM approx 2021 - school + 18 + Residual(register)
        else:
            r = 1821.079 + 0.06561 * register - 0.82946 * school - 0.12747 * g5  # LM approx all available params
    r = max(14, r)
    r = min(89, r)
    return r


def calc_group_als_vectorized(ids: np.array, groups: defaultdict) -> np.array:
    global user_embs_for_group_knn
    user_embs_for_group_knn = np.array([calc_user_embed_by_train_seq(
        filter_train_seq(groups[_id], train_groups_set), GROUP_ALS_DIM, train_group_2_emb) for _id in ids])
    return group_knn.predict(user_embs_for_group_knn).reshape(-1)

def calc_friend_als_vectorized(ids: np.array, friends: defaultdict) -> np.array:
    global user_embs_for_friend_knn
    user_embs_for_friend_knn = np.array([calc_user_embed_by_train_seq(
        filter_train_seq(friends[_id], train_friends_set), FRIEND_ALS_DIM, train_friends_2_emb) for _id in ids])
    return friend_knn.predict(user_embs_for_friend_knn).reshape(-1)

def apply_cb_model_v1(ids: np.array, common_uids: np.array, features: np.array) -> np.array:
    approxes = edu_cb_model_v1.predict(features[:, CB_V1_FEATURE_SLICE])
    uid2approx = {u:a for u, a in zip(common_uids, approxes)}
    return np.array([uid2approx[u] for u in ids])

def apply_cb_model_v2(ids: np.array, common_uids: np.array, features: np.array) -> np.array:
    approxes = edu_cb_model_v2.predict(features[:, CB_V2_FEATURE_SLICE])
    uid2approx = {u:a for u, a in zip(common_uids, approxes)}
    return np.array([uid2approx[u] for u in ids])

def apply_cb_model_v3(ids: np.array, common_uids: np.array, features: np.array) -> np.array:
    approxes = edu_cb_model_v3.predict(features[:, CB_V3_FEATURE_SLICE])
    uid2approx = {u:a for u, a in zip(common_uids, approxes)}
    return np.array([uid2approx[u] for u in ids])

def make_raw_predictions(ids: pd.DataFrame, education: pd.DataFrame, friends_df: pd.DataFrame, groups: defaultdict, friends: defaultdict) -> pd.DataFrame:
    result = pd.DataFrame()
    result['uid'] = ids['uid']
    result['group-als'] = calc_group_als_vectorized(ids['uid'].values, groups)
    result['friend-als'] = calc_friend_als_vectorized(ids['uid'].values, friends)
    common_uids, cb_features_v3 = make_common_features_v3(ids, education, friends_df, groups, friends)  # requires both als embeddings ^
    result['edu-cb-v1'] = apply_cb_model_v1(ids['uid'].values, common_uids, cb_features_v3)
    result['edu-cb-v2'] = apply_cb_model_v2(ids['uid'].values, common_uids, cb_features_v3)
    result['edu-cb-v3'] = apply_cb_model_v3(ids['uid'].values, common_uids, cb_features_v3)
    assert result.shape[0] == ids.shape[0] and result.shape[1] == 6
    assert ['uid', 'group-als', 'friend-als', 'edu-cb-v1', 'edu-cb-v2', 'edu-cb-v3'] == list(result.columns)
    return result


def make_predictions(ids: pd.DataFrame, education: pd.DataFrame, groups: pd.DataFrame, friends: pd.DataFrame) -> pd.DataFrame:
    groups_list = defaultdict(list)
    for uid, gid in zip(groups['uid'].values, groups['gid'].values):
        groups_list[uid].append(gid)

    friends_list = defaultdict(list)
    for uid, fuid in zip(friends['uid'].values, friends['fuid'].values):
        friends_list[uid].append(fuid)
        friends_list[fuid].append(uid)

    raw_predictions = make_raw_predictions(ids, education, friends, groups_list, friends_list)
    user_2_group_als_prediction = {uid: r for uid, r in zip(raw_predictions['uid'], raw_predictions['group-als'])}
    user_2_friend_als_prediction = {uid: r for uid, r in zip(raw_predictions['uid'], raw_predictions['friend-als'])}
    user_2_cb_v1_prediction = {uid: r for uid, r in zip(raw_predictions['uid'], raw_predictions['edu-cb-v1'])}
    user_2_cb_v2_prediction = {uid: r for uid, r in zip(raw_predictions['uid'], raw_predictions['edu-cb-v2'])}
    user_2_cb_v3_prediction = {uid: r for uid, r in zip(raw_predictions['uid'], raw_predictions['edu-cb-v3'])}

    result = pd.DataFrame()
    result['uid'] = ids['uid']

    school = {uid : year for uid, year in zip(education['uid'].values, education['school_education'].values)}
    g5= {uid : year for uid, year in zip(education['uid'].values, education['graduation_5'].values)}
    register = {uid : year for uid, year in zip(ids['uid'].values, ids['registered_year'].values)}

    result['age'] = [decision(
        uid,
        school[uid],
        g5[uid],
        register[uid],
        groups_list[uid],
        friends_list[uid],
        user_2_group_als_prediction[uid],
        user_2_friend_als_prediction[uid],
        user_2_cb_v1_prediction[uid],
        user_2_cb_v2_prediction[uid],
        user_2_cb_v3_prediction[uid]
    ) for uid in result['uid'].values]
    assert result.shape[0] == ids.shape[0] and result.shape[1] == 2
    assert ['uid', 'age'] == list(result.columns)
    return result


def main(model: str, input: str, output: str):
    global ids_order
    ids = read_csv(os.path.join(input, TEST))
    ids_order = {uid: i for i, uid in enumerate(ids['uid'].values)}
    education = read_csv(os.path.join(input, EDUCATION))
    groups = read_csv(os.path.join(input, GROUPS))
    friends = read_csv(os.path.join(input, FRIENDS))
    ids_with_predictions = make_predictions(ids, education, groups, friends)
    write_csv(ids_with_predictions, output)


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--is-train-set', default=False, action='store_true')

    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--input', type=str, default="/tmp/data")
    parser.add_argument('--output', type=str, default="/opt/results/results.tsv")

    args, unparsed = parser.parse_known_args()
    assert unparsed is None or len(unparsed) == 0

    set_global(args.is_train_set)

    main(args.model, args.input, args.output)
