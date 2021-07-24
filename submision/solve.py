import argparse
import os
import pandas as pd

TEST = "test.csv"

def read_csv(input: str) -> pd.DataFrame:
    return pd.read_csv(input)

def write_csv(df: pd.DataFrame, output: str):
    df.to_csv(output, index=None, index_label=None)

def make_predictions(ids: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame()
    result['uid'] = ids['uid']
    result['age'] = 35
    assert result.shape[0] == ids.shape[0] and result.shape[1] == 2
    assert ['uid', 'age'] == list(result.columns)
    return result

def main(model: str, input: str, output: str):
    input_path = os.path.join(input, TEST)
    input_ids = read_csv(input_path)
    ids_with_predictions = make_predictions(input_ids)
    write_csv(ids_with_predictions, output)


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--input', type=str, default="/tmp/data")
    parser.add_argument('--output', type=str, default="/opt/results/results.tsv")

    args, unparsed = parser.parse_known_args()
    assert unparsed is None or len(unparsed) == 0

    main(args.model, args.input, args.output)
