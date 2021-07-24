import argparse
import os
import pandas as pd
import numpy as np


def read_csv(input: str) -> pd.DataFrame:
    return pd.read_csv(input)



def calc_score(approx: pd.DataFrame, golden: pd.DataFrame) -> dict:
    assert approx.shape[0] == golden.shape[0]
    assert np.all(approx['uid'].values == golden['uid'].values)
    approx_ages = approx['age'].values
    golden_ages = golden['age'].values
    rmse = np.mean((approx_ages - golden_ages) ** 2) ** 0.5
    return {"size": approx.shape[0], "rmse": rmse, "1/rmse": 1 / rmse}


def main(approx: str, golden: str):
    approx = read_csv(approx)
    golden = read_csv(golden)
    score = calc_score(approx, golden)
    print(score)


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--approx', type=str, default=None)
    parser.add_argument('--golden', type=str, default=None)

    args, unparsed = parser.parse_known_args()
    assert unparsed is None or len(unparsed) == 0

    main(args.approx, args.golden)
