from typing import List, Dict, Callable, Iterable

import pandas as pd
from tqdm.auto import tqdm

from conversant.conversation import Conversation

from experiments.datahandlers.iac import FourForumsDataLoader, CreateDebateDataLoader, build_iac_conversations
from experiments.datahandlers.iac.convinceme_data import ConvinceMeDataLoader

ConversationsLoader = Callable[[str], List[Conversation]]


def load_4forums_conversations(data_dir: str) -> List[Conversation]:
    loader = FourForumsDataLoader(data_dir)
    records = tqdm(loader.load_post_records())
    return list(build_iac_conversations(records))


def load_cd_conversations(data_dir: str) -> List[Conversation]:
    loader = CreateDebateDataLoader(data_dir)
    records = tqdm(loader.load_post_records())
    return list(build_iac_conversations(records))


def load_cm_conversations(data_dir: str) -> List[Conversation]:
    loader = ConvinceMeDataLoader(data_dir)
    records = tqdm(loader.load_post_records())
    return list(build_iac_conversations(records))


conversation_loaders: Dict[str, ConversationsLoader] = {
    "4forums": load_4forums_conversations,
    "createdebate": load_cd_conversations,
    "convinceme": load_cm_conversations
}


def load_conversations(dataset_name: str, basedir: str) -> List[Conversation]:
    loader = conversation_loaders.get(dataset_name, None)
    if loader is None:
        raise ValueError(f"Unknown dataset name: {dataset_name}. must be one of f{list(conversation_loaders.keys())}")

    return loader(basedir)




