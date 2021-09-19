import argparse
import pandas as pd
from tqdm.auto import tqdm
from typing import Iterable

def up_to_root() -> str:
    file = os.path.abspath(__file__)
    current_dir = os.path.dirname(file)
    while os.path.split(current_dir)[-1] != "cmv-stance-classification":
        current_dir = os.path.dirname(current_dir)

    return current_dir


try:
    import conversant
except:
    import sys
    import os

    # ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ROOT_DIR = up_to_root()
    CONVERSANT_DIR = os.path.join(os.path.dirname(ROOT_DIR), "conversant")
    # EXPERIMENTS_DIR = os.path.join(ROOT_DIR, "experiments")
    # sys.path.append(ROOT_DIR)
    sys.path.append(CONVERSANT_DIR)

from conversant.conversation import Conversation
from conversant.conversation.conversation_utils import conversation_to_dataframe
from experiments.datahandlers.loaders import load_conversations


def conversations_to_dataframe(conversations: Iterable[Conversation]) -> pd.DataFrame:
    return pd.concat(map(conversation_to_dataframe, tqdm(conversations)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str,
                        help="name of the dataset to prepare")
    parser.add_argument("path", type=str,
                        help="Path to the IAC directory containing all dataset as downloaded and extracted")
    parser.add_argument("out", type=str,
                        help="Output path to store the dataset in the new format (similar to VAST)")

    args = parser.parse_args()

    convs = load_conversations(args.dataset, args.path)
    df = conversations_to_dataframe(convs)
    df.to_csv(args.out, index=False)
