from typing import List, Dict, Callable, Iterable

import pandas as pd
from tqdm.auto import tqdm

from conversant.conversation import Conversation
from conversation.parse import DataFrameConversationReader
from experiments.datahandlers.iac.createdebate_data import CreateDebateDataLoader

from experiments.datahandlers.iac.fourforum_data import \
    load_post_records as load_4forums_post_records,\
    build_conversations as build_4forums_conversations

from experiments.datahandlers.iac.createdebate_data import \
    build_conversations as build_cd_conversations


ConversationsLoader = Callable[[str], List[Conversation]]

LabelsLoader = Callable[[str]]


def load_4forums_conversations(data_dir: str) -> List[Conversation]:
    records = tqdm(load_4forums_post_records(data_dir))
    return list(build_4forums_conversations(records))


def load_cd_conversations(data_dir: str) -> List[Conversation]:
    loader = CreateDebateDataLoader(data_dir)
    records = tqdm(loader.load_post_records())
    return list(build_cd_conversations(records))


conversation_loaders: Dict[str, ConversationsLoader] = {
    "4forums": load_4forums_conversations,
    "createdebate": load_cd_conversations
}

labels_loaders: Dict[str, LabelsLoader]


def load_conversations(dataset_name: str, basedir: str) -> List[Conversation]:
    loader = conversation_loaders.get(dataset_name, None)
    if loader is None:
        raise ValueError(f"Unknown dataset name: {dataset_name}. must be one of f{list(conversation_loaders.keys())}")

    return loader(basedir)




