import argparse
import os
import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsRegressor

from collections import defaultdict



USE_NAIVE = False
ALS_DIM = 32

### SET VARIABLES

TEST = None
EDUCATION = None
GROUPS = None

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
knn = KNeighborsRegressor(weights='distance', n_neighbors=25)
knn.fit(train_als_user_embeddings, train_uids['age'].values)

### END UPLOAD TRAIN DATA

def write_csv(df: pd.DataFrame, output: str):
    df.to_csv(output, index=None, index_label=None)


def filter_train_groups(groups: list) -> list:
    filtred = []
    for g in groups:
        if g in train_groups_set:
            filtred.append(g)
    return filtred

def calc_user_embed_by_train_groups(groups: list) -> np.array:
    e = np.zeros(ALS_DIM)
    if len(groups) == 0:
        return e
    for g in groups:
        e += train_group_2_emb[g]
    return e / len(groups)


def decision_als(uid: int, groups: list, school: float, register: float) -> float:
    groups = filter_train_groups(groups)
    if len(groups) == 0:
        return decision_naive_impl(school, register)
    e = calc_user_embed_by_train_groups(groups)
    r = knn.predict([e])
    r = float(r)
    return r


def decision(uid: int, school: float, register: float, groups: list) -> float:
    if USE_NAIVE:
        return decision_naive_impl(school, register)
    return decision_als(uid, groups, school, register)

def decision_naive_impl(school: float, register: float) -> float:
    if np.isnan(school):
        r = 697.208 - 0.32883 * register  # LM approx 35
    else:
        r = 1918.977 - 0.05583 * register - 0.88422 * school  # LM approx 2021 - school + 18 + Residual(register)
    r = max(14, r)
    r = min(89, r)
    return r


def make_predictions(ids: pd.DataFrame, education: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame()
    result['uid'] = ids['uid']

    school = {uid : year for uid, year in zip(education['uid'].values, education['school_education'].values)}
    register = {uid : year for uid, year in zip(ids['uid'].values, ids['registered_year'].values)}

    groups_list = defaultdict(list)
    for uid, gid in zip(groups['uid'].values, groups['gid'].values):
        groups_list[uid].append(gid)

    result['age'] = [decision(uid, school[uid], register[uid], groups_list[uid]) for uid in result['uid'].values]
    assert result.shape[0] == ids.shape[0] and result.shape[1] == 2
    assert ['uid', 'age'] == list(result.columns)
    return result


def main(model: str, input: str, output: str):
    ids = read_csv(os.path.join(input, TEST))
    education = read_csv(os.path.join(input, EDUCATION))
    groups = read_csv(os.path.join(input, GROUPS))
    ids_with_predictions = make_predictions(ids, education, groups)
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
