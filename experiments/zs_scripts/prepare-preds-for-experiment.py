import argparse

import pandas as pd


def load_preds(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def prepare_preds(preds: pd.DataFrame) -> pd.DataFrame:
    return preds[["ori_id", "pred"]]


def write_preds(preds: pd.DataFrame, outpath: str):
    preds.to_csv(outpath, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path to a predictions file from the zero-shot model")
    parser.add_argument("-o", "--outpath", default="preds.csv",
                        help="path to a predictions file from the zero-shot model")

    args = parser.parse_args()

    preds_path = args.path
    write_preds(prepare_preds(load_preds(args.path)), args.outpath)





