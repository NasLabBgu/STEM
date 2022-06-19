from typing import List, Dict, Callable

from tqdm.auto import tqdm

from conversant.conversation import Conversation

from experiments.datahandlers.iac import FourForumsDataLoader, build_iac_conversations, get_convinceme_records_loader, \
    get_createdebate_records_loader, IACRecordsLoader

ConversationsLoader = Callable[[str], IACRecordsLoader]


conversation_loaders: Dict[str, ConversationsLoader] = {
    "4forums": FourForumsDataLoader,
    "createdebate": get_createdebate_records_loader,
    "convinceme": get_convinceme_records_loader
}


def load_conversations(dataset_name: str, basedir: str) -> List[Conversation]:
    loader = conversation_loaders.get(dataset_name)
    if loader is None:
        raise ValueError(f"Unknown dataset name: {dataset_name}. must be one of f{list(conversation_loaders.keys())}")

    loader = loader(basedir)
    records = tqdm(loader.load_post_records())
    return list(build_iac_conversations(records))




