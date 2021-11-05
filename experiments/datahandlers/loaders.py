from typing import List, Dict, Callable, Iterable

import pandas as pd
from tqdm.auto import tqdm

from conversant.conversation import Conversation
from conversation.parse import DataFrameConversationReader

from experiments.datahandlers.iac.fourforum_data import \
    load_post_records as load_4forums_post_records,\
    build_conversations as build_4forums_conversations


ConversationsLoader = Callable[[str], List[Conversation]]


def load_4forums_conversations(data_dir: str) -> List[Conversation]:
    records = tqdm(load_4forums_post_records(data_dir))
    return list(build_4forums_conversations(records))


loaders: Dict[str, ConversationsLoader] = {
    "4forums": load_4forums_conversations
}


def load_conversations(dataset_name: str, basedir: str) -> List[Conversation]:
    loader = loaders.get(dataset_name, None)
    if loader is None:
        raise ValueError(f"Unknown dataset name: {dataset_name}. must be one of f{list(loaders.keys())}")

    return load_4forums_conversations(basedir)




