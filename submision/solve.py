import argparse
import os
import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsRegressor

from collections import defaultdict



USE_NAIVE = False
GROUP_ALS_DIM = 32
FRIEND_ALS_DIM = 16
ALS_TRAIN_GROUPS_TH = 1  # 5 is worse that 1
GROUP_OVER_FRIEND_WEIGHT = 0.75
ALS_OVER_NAIVE_WEIGHT = 0.75

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
train_group_2_emb = {g: e for g,e in zip(train_groups, train_als_group_embeddings)}
group_knn = KNeighborsRegressor(weights='distance', n_neighbors=25)
group_knn.fit(train_als_user_embeddings, train_uids['age'].values)

train_friend_als_user_embeddings = read_csv('data/train_friends_als_user_embeddings.csv').values
train_friend_als_friend_embeddings = read_csv('data/train_friends_als_friends_embeddings.csv').values
train_friends = read_csv('data/train_friends.csv').values.reshape(-1)
train_friends_set = set(list(train_friends))
train_friends_2_emb = {g: e for g,e in zip(train_friends, train_friend_als_friend_embeddings)}
friend_knn = KNeighborsRegressor(weights='distance', n_neighbors=25)
friend_knn.fit(train_friend_als_user_embeddings, train_uids['age'].values)

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


def decision(
    uid: int,
    school: float,
    g5: float,
    register: float,
    groups: list,
    friends: list,
    group_als_prediction: float,
    friend_als_prediction: float
) -> float:
    naive = decision_naive_impl(school, register, g5)
    if USE_NAIVE:
        return naive

    groups = filter_train_seq(groups, train_groups_set)
    friends = filter_train_seq(friends, train_friends_set)
    if len(groups) < ALS_TRAIN_GROUPS_TH and len(friends) < ALS_TRAIN_GROUPS_TH:
        return naive
    if len(groups) < ALS_TRAIN_GROUPS_TH:
        return friend_als_prediction * ALS_OVER_NAIVE_WEIGHT + (1 - ALS_OVER_NAIVE_WEIGHT) * naive
    if len(friends) < ALS_TRAIN_GROUPS_TH:
        return group_als_prediction * ALS_OVER_NAIVE_WEIGHT + (1 - ALS_OVER_NAIVE_WEIGHT) * naive
    return group_als_prediction * GROUP_OVER_FRIEND_WEIGHT + (1 - GROUP_OVER_FRIEND_WEIGHT) * friend_als_prediction



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
    user_embs_for_group_knn = np.array([calc_user_embed_by_train_seq(
        filter_train_seq(groups[_id], train_groups_set), GROUP_ALS_DIM, train_group_2_emb) for _id in ids])
    return group_knn.predict(user_embs_for_group_knn).reshape(-1)

def calc_friend_als_vectorized(ids: np.array, friends: defaultdict) -> np.array:
    user_embs_for_friend_knn = np.array([calc_user_embed_by_train_seq(
        filter_train_seq(friends[_id], train_friends_set), FRIEND_ALS_DIM, train_friends_2_emb) for _id in ids])
    return friend_knn.predict(user_embs_for_friend_knn).reshape(-1)

def make_raw_predictions(ids: pd.DataFrame, education: pd.DataFrame, groups: defaultdict, friends: defaultdict) -> pd.DataFrame:
    result = pd.DataFrame()
    result['uid'] = ids['uid']
    result['group-als'] = calc_group_als_vectorized(ids['uid'].values, groups)
    result['friend-als'] = calc_friend_als_vectorized(ids['uid'].values, friends)
    assert result.shape[0] == ids.shape[0] and result.shape[1] == 3
    assert ['uid', 'group-als', 'friend-als'] == list(result.columns)
    return result


def make_predictions(ids: pd.DataFrame, education: pd.DataFrame, groups: pd.DataFrame, friends: pd.DataFrame) -> pd.DataFrame:
    groups_list = defaultdict(list)
    for uid, gid in zip(groups['uid'].values, groups['gid'].values):
        groups_list[uid].append(gid)

    friends_list = defaultdict(list)
    for uid, fuid in zip(friends['uid'].values, friends['fuid'].values):
        friends_list[uid].append(fuid)

    raw_predictions = make_raw_predictions(ids, education, groups_list, friends_list)
    user_2_group_als_prediction = {uid: r for uid, r in zip(raw_predictions['uid'], raw_predictions['group-als'])}
    user_2_friend_als_prediction = {uid: r for uid, r in zip(raw_predictions['uid'], raw_predictions['friend-als'])}

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
        user_2_friend_als_prediction[uid]
    ) for uid in result['uid'].values]
    assert result.shape[0] == ids.shape[0] and result.shape[1] == 2
    assert ['uid', 'age'] == list(result.columns)
    return result


def main(model: str, input: str, output: str):
    ids = read_csv(os.path.join(input, TEST))
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
