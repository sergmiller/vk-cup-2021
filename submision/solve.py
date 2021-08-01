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
GEMBEDDINGS_DIM = 64

# DEFAULT_GROUP_ALS_ITEM_EMB = np.array([0.00244036, 0.00203669, 0.00181023, 0.00288724, 0.00168105,
#        0.00253628, 0.0019611 , 0.00245336, 0.00279846, 0.00175673,
#        0.00231933, 0.00180845, 0.00269746, 0.00226039, 0.00224252,
#        0.00202341, 0.00218132, 0.00209919, 0.00211487, 0.00196292,
#        0.00131669, 0.00241805, 0.00255356, 0.0026769 , 0.00232002,
#        0.00262157, 0.00175571, 0.00160803, 0.00271791, 0.0023224 ,
#        0.00258682, 0.0021744 ])


COMMON_SLICE_LEN = 8 + 8 + 2 + 1 + 7
CB_V1_FEATURE_SLICE = list(np.arange(COMMON_SLICE_LEN + GEMBEDDINGS_DIM + FRIEND_ALS_DIM))# + GROUP_ALS_DIM))


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

friend_als_user_embeddings = read_csv('data/friends_als_user_embeddings.csv').values
train_friend_als_user_embeddings = read_csv('data/train_friends_als_user_embeddings.csv').values
train_friend_als_friend_embeddings = read_csv('data/train_friends_als_friends_embeddings.csv').values
train_friends = read_csv('data/train_friends.csv').values.reshape(-1)
train_friends_set = set(list(train_friends))

friend_als_all_user_embeddings = read_csv("data/friends_als_all_user_embeddings.csv").values
group_als_all_user_embeddings = read_csv("data/group_als_all_user_embeddings.csv").values

assert len(train_uids['uid']) == len(train_friend_als_user_embeddings)
# print(train_friend_als_user_embeddings.shape, train_friends.shape)
train_friends_2_user_emb = {g: e for g,e in zip(train_uids['uid'], train_friend_als_user_embeddings)}
assert len(train_friends) == len(train_friend_als_friend_embeddings)
train_friends_2_emb = {g: e for g,e in zip(train_friends, train_friend_als_friend_embeddings)}
assert len(train_friends) == len(friend_als_user_embeddings)
friends_2_user_emb = {g: e for g,e in zip(train_friends, friend_als_user_embeddings)}

def load_mean(a: np.array) -> dict:
    return {x:y for x,y in a}

f_min = load_mean(read_csv('data/f_min_p35.csv').values)
f_max = load_mean(read_csv('data/f_max_p35.csv').values)
f_med = load_mean(read_csv('data/f_med_p35.csv').values)
g_mean = load_mean(read_csv('data/g_mean_p35_a10.csv').values)
g_reg_mean = load_mean(read_csv('data/g_reg_mean_p2014_a10.csv').values)
f_mean = load_mean(read_csv('data/f_mean_p35_a10.csv').values)
f_reg_mean = load_mean(read_csv('data/f_reg_mean_p2014_a10.csv').values)

user_g_nodes = read_csv('data/graph_nodes_user_ids.csv').values
gembeddings = read_csv('data/graph_embeddings.csv').values
train_f2u_w1_svd_16 = read_csv("data/train_f2u_w1_svd_16.csv").values

uid2gembeddding = {int(uid): gembeddings[i] for i, uid in enumerate(user_g_nodes)}

class MultiheadCatboostModel:
    def __init__(self, paths: list):
        self.paths = paths
        self.models = []
        for p in self.paths:
            m = catboost.CatBoost().load_model(p)
            assert len(CB_V1_FEATURE_SLICE) == len(m.feature_names_)
            self.models.append(m)
    def predict(self, X: np.array):
        y = np.empty((X.shape[0], len(self.models)))
        for i, m in enumerate(self.models):
            y[:, i] = m.predict(X)
        return np.mean(y, axis=1)

MULTIMODEL_SIZE = 50
edu_cb_model_v1 = MultiheadCatboostModel(
    ["data/edu_v1_g_y_mmg_mmf_mmgreg_mmfreg_p35_a10_part_bins32_v2_{}_of_{}.cbm".format(i+1, MULTIMODEL_SIZE) for i in range(MULTIMODEL_SIZE)]
)

# edu_cb_model_v1 = catboost.CatBoost().load_model("data/edu_v1_g_y_mmg_mmf_mmgreg_mmfreg_p35_a10_bins32_v2.cbm")
# assert len(CB_V1_FEATURE_SLICE) == len(edu_cb_model_v1.feature_names_)
#
# user_embs_for_friend_knn = None
user_embs_for_group_knn = None
ids_order = None
# uid2mmg = None



### END UPLOAD TRAIN DATA

def calc_mean_of_mean_of_groups(uids: pd.DataFrame, groups: pd.DataFrame, g_mean: dict) -> dict:
    u2f = dict()
    g_list = defaultdict(list)
    for uid, guid in zip(groups['uid'].values, groups['gid'].values):
        g_list[uid].append(guid)
    for u in g_list.keys():
        verified_groups = [g for g in g_list[u] if g in g_mean]
        if len(verified_groups) == 0:
            continue
        u2f[u] = np.mean([g_mean[v] for v in verified_groups])
    return u2f

def calc_mean_of_mean_of_friends(uids: pd.DataFrame, friends: pd.DataFrame, f_mean: dict) -> dict:
    u2f = dict()
    f_list = defaultdict(list)
    for uid, fid in zip(friends['uid'].values, friends['fuid'].values):
        f_list[uid].append(fid)
        f_list[fid].append(uid)
    for u in f_list.keys():
        verified_friends = [f for f in f_list[u] if f in f_mean]
        if len(verified_friends) == 0:
            continue
        u2f[u] = np.mean([f_mean[v] for v in verified_friends])
    return u2f

def write_csv(df: pd.DataFrame, output: str):
    df.to_csv(output, index=None, index_label=None)


def filter_train_seq(groups: list, filter_set: set) -> list:
    filtred = []
    for g in groups:
        if g in filter_set:
            filtred.append(g)
    return filtred

def calc_user_embed_by_train_seq(groups: list, als_dim: int, entity2emb: dict, default_emb: np.array) -> np.array:
    e = np.zeros(als_dim)
    if len(groups) == 0:
        return default_emb
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
    cb_v1_prediction: float
) -> float:
    return cb_v1_prediction


def get_als_friends_embed(uid: int) -> np.array:
    return friend_als_all_user_embeddings[uid]

def get_als_group_embed(uid: int) -> np.array:
    # if uid < len(group_als_all_user_embeddings):
    #     return group_als_all_user_embeddings[uid]
    # return np.zeros(GROUP_ALS_DIM)
    return user_embs_for_group_knn[ids_order[uid]]

f_cache = {}
def get_friends_mean_embed(uid: int, friends_list: list)-> np.array:
    if uid in f_cache:
        return f_cache[uid]
    e = np.zeros(FRIEND_ALS_DIM)
    count = 0
    for f in friends_list[uid]:
        m = friends_2_user_emb[f]
        e += m
        count += 1
    if count > 0:
        e /= count
    f_cache[uid] = e
    return e

def make_common_features_v3(
        uids: pd.DataFrame,
        edu: pd.DataFrame,
        friends_df: pd.DataFrame,
        groups_df: pd.DataFrame,
        groups: defaultdict,
        friends: defaultdict
) -> tuple:
    global uid2mmg
    friends_pr = calc_friends_pagerank(friends_df)
    edu_features = []
    edu_ids = edu['uid']
    uid2register = {u:r for u,r in zip(uids['uid'].values, uids['registered_year'].values)}

    uid2mmg = calc_mean_of_mean_of_groups(uids, groups_df, g_mean)
    uid2mmf = calc_mean_of_mean_of_friends(uids, friends_df, f_mean)
    uid2mmg_reg = calc_mean_of_mean_of_groups(uids, groups_df, g_reg_mean)
    uid2mmf_reg = calc_mean_of_mean_of_friends(uids, friends_df, f_reg_mean)
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

        features.append(register_year)
        features.append(f_min.get(uid, 35))
        features.append(f_max.get(uid, 35))
        # features.append(f_med.get(uid, 35))
        # features.append(f_reg_mean.get(uid, 2014))
        features.append(uid2mmg.get(uid, 35))
        features.append(uid2mmf.get(uid, 35))
        features.append(uid2mmg_reg.get(uid, 2014))
        features.append(uid2mmf_reg.get(uid, 2014))
        # features.append(uid2meta_pseudo.get(uid, 0))
        # features.append(decision_naive_impl(x['school_education'], register_year, x['graduation_5']))
        f = features_ind + features + list(uid2gembeddding.get(uid, np.zeros(GEMBEDDINGS_DIM))) + list(get_als_friends_embed(uid))# + list(train_f2u_w1_svd_16[uid])
        # + list(get_als_group_embed(uid))  # + list(get_friends_mean_embed(uid, friends))
        edu_features.append(f)
    return edu_ids, np.array(edu_features)

# def calc_group_als_vectorized(ids: np.array, groups: defaultdict) -> np.array:
#     global user_embs_for_group_knn
#     user_embs_for_group_knn = np.array([calc_user_embed_by_train_seq(
#         filter_train_seq(groups[_id], train_groups_set),
#         GROUP_ALS_DIM,
#         train_group_2_emb,
#         DEFAULT_GROUP_ALS_ITEM_EMB) for _id in ids])
    # print(ids.shape,user_embs_for_group_knn.shape, len(user_embs_for_group_knn), len(ids_order))
    # return np.zeros(user_embs_for_group_knn.shape[0])
    # return group_knn.predict(user_embs_for_group_knn).reshape(-1)
#
# def calc_friend_als_vectorized(ids: np.array, friends: defaultdict) -> np.array:
#     global user_embs_for_friend_knn
#     user_embs_for_friend_knn = np.array([calc_user_embed_by_train_seq(
#         filter_train_seq(friends[_id], train_friends_set), FRIEND_ALS_DIM, train_friends_2_emb) for _id in ids])
#     return np.zeros(user_embs_for_friend_knn.shape[0])
#     return friend_knn.predict(user_embs_for_friend_knn).reshape(-1)

def apply_cb_model_v1(ids: np.array, common_uids: np.array, features: np.array) -> np.array:
    approxes = edu_cb_model_v1.predict(features[:, CB_V1_FEATURE_SLICE])
    uid2approx = {u:a for u, a in zip(common_uids, approxes)}
    return np.array([uid2approx[u] for u in ids])

def make_raw_predictions(ids: pd.DataFrame, education: pd.DataFrame, friends_df: pd.DataFrame, groups_df: pd.DataFrame, groups: defaultdict, friends: defaultdict) -> pd.DataFrame:
    result = pd.DataFrame()
    result['uid'] = ids['uid']
    common_uids, cb_features_v3 = make_common_features_v3(ids, education, friends_df, groups_df, groups, friends)
    result['edu-cb-v1'] = apply_cb_model_v1(ids['uid'].values, common_uids, cb_features_v3)
    assert result.shape[0] == ids.shape[0] and result.shape[1] == 2
    assert ['uid', 'edu-cb-v1'] == list(result.columns)
    return result


def make_predictions(ids: pd.DataFrame, education: pd.DataFrame, groups: pd.DataFrame, friends: pd.DataFrame) -> pd.DataFrame:
    groups_list = defaultdict(list)
    for uid, gid in zip(groups['uid'].values, groups['gid'].values):
        groups_list[uid].append(gid)

    friends_list = defaultdict(list)
    for uid, fuid in zip(friends['uid'].values, friends['fuid'].values):
        friends_list[uid].append(fuid)
        friends_list[fuid].append(uid)

    raw_predictions = make_raw_predictions(ids, education, friends, groups, groups_list, friends_list)
    user_2_cb_v1_prediction = {uid: r for uid, r in zip(raw_predictions['uid'], raw_predictions['edu-cb-v1'])}

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
        user_2_cb_v1_prediction[uid]
    ) for uid in result['uid'].values]
    assert result.shape[0] == ids.shape[0] and result.shape[1] == 2
    assert ['uid', 'age'] == list(result.columns)
    return result


def main(input: str, output: str):
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

    parser.add_argument('--input', type=str, default="/tmp/data")
    parser.add_argument('--output', type=str, default="/opt/results/results.tsv")

    args, unparsed = parser.parse_known_args()
    assert unparsed is None or len(unparsed) == 0

    set_global(args.is_train_set)

    main(args.input, args.output)
