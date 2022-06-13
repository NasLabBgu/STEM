import abc
from collections import Counter
from itertools import groupby
from typing import Tuple, NamedTuple, Iterable, Any, List, Optional, Sequence, Dict, TypeVar

from conversant.conversation import NodeData
from conversant.conversation.parse import ConversationParser, NamedTupleConversationReader
from conversant.conversation.parse.conversation_parser import T
from conversant.conversation import Conversation
from experiments.datahandlers.iac.delegation_utils import DelegatingMeta
from experiments.datahandlers.iac.iac_data_records import DiscussionMetadata

NamedTuples = Iterable[NamedTuple]

PARSE_STRATEGY = {
    "node_id": "post_id",
    "author": "author_id",
    "timestamp": "creation_date",
    "parent_id": "parent",
}

NO_PARENT_VALUE = None

K = TypeVar('K')
N = TypeVar('N', bound=NamedTuple)


def get_most_common(it: Iterable[K]) -> K:
    return Counter(filter(lambda e: e is not None, it)).most_common(1)[0][0]


def copy_namedtuple(nt: N, **modifications: Any) -> N:
    attrs = nt._asdict()
    for k, v in modifications.items():
        if k in attrs:
            attrs[k] = v

    return type(nt)(**attrs)


class IACConversationParser(ConversationParser[Tuple[Any, NamedTuples], NamedTuple]):
    def __init__(self):
        self.namedtuple_parser = NamedTupleConversationReader(PARSE_STRATEGY, NO_PARENT_VALUE)

    def extract_conversation_id(self, raw_conversation: Tuple[Any, NamedTuples]) -> Any:
        return raw_conversation[0]

    def extract_node_data(self, raw_node: NamedTuple) -> NodeData:
        return self.namedtuple_parser.extract_node_data(raw_node)

    def iter_raw_nodes(self, raw_conversation: Tuple[Any, NamedTuples]) -> Iterable[T]:
        return self.namedtuple_parser.iter_raw_nodes(raw_conversation[1])


class IACPostRecord(NamedTuple):
    topic: int
    topic_name: str
    discussion_id: int
    post_id: int
    author_id: int
    creation_date: str
    parent: Optional[int]
    parent_missing: bool
    discussion_title: str
    text: str
    quote_source_ids: List[int] # post ids of the quotes contained in this ppost
    stance_id: int
    stance_name: str
    response_type: Optional[str]
    url: str
    author_stance_id: int
    author_stance_name: str


class IACRecordsLoader(abc.ABC, metaclass=abc.ABCMeta):

    NULL_VALUE = "\\N"
    ROOT_PARENT_ID = NULL_VALUE

    DISCUSSION_ID_INDEX = 0
    POST_ID_INDEX = 1
    AUTHOR_ID_INDEX = 2
    DATETIME_INDEX = 3
    PARENT_ID_INDEX = 4
    PARENT_MISSING_INDEX = 5
    TEXT_ID_INDEX = 6

    UNKNOWN_STANCE_VALUE = -1
    UNKNOWN_STANCE_NAME = "unknown"
    OTHER_STANCE_VALUE = -9
    OTHER_STANCE_NAME = "other"


    def load_post_records(self) -> Iterable[IACPostRecord]:
        records = self.iter_raw_records()
        parsed_records = map(self.parse_record, records)
        yield from parsed_records

    @abc.abstractmethod
    def iter_raw_records(self) -> Iterable[Sequence[str]]:
        raise NotImplementedError

    def parse_record(self, record: Sequence[str]) -> IACPostRecord:
        discussion_id = int(record[self.DISCUSSION_ID_INDEX])
        metadata = self.get_discussion_metadata(discussion_id)
        topic_id = metadata.topic_id
        topic_name = metadata.topic_str
        post_id = int(record[self.POST_ID_INDEX])
        author_id = int(record[self.AUTHOR_ID_INDEX])
        parent_id = self.get_parent_id(record[self.PARENT_ID_INDEX], discussion_id)
        parent_missing = self.get_is_parent_missing(record, parent_id)
        creation_date = str(record[self.AUTHOR_ID_INDEX])
        title = metadata.record.title
        text = self.get_post_content(int(record[self.TEXT_ID_INDEX]))
        quotes = self.get_quotes(discussion_id, post_id)
        stance_id = self.get_stance_id(author_id, discussion_id, post_id, record)
        stance_name = self.get_stance_name(discussion_id, stance_id) if stance_id >= 0 else "unknown"
        author_stance_id = self.get_author_stance_id(author_id, discussion_id, post_id, record)
        author_stance_name = self.get_stance_name(discussion_id, author_stance_id)
        url = metadata.record.url
        response_type = self.get_response_type(record)
        return IACPostRecord(topic_id, topic_name, discussion_id, post_id, author_id, creation_date, parent_id,
                             parent_missing, title, text, quotes, stance_id, stance_name, response_type, url,
                             author_stance_id, author_stance_name)

    @abc.abstractmethod
    def get_discussion_metadata(self, discussion_id: int):
        raise NotImplementedError

    @abc.abstractmethod
    def get_text_mapping(self) -> Dict[int, str]:
        raise NotImplementedError

    def get_post_content(self, text_id: int) -> str:
        return self.get_text_mapping().get(text_id, "[deleted]")

    @abc.abstractmethod
    def get_quotes(self, discussion_id: int, post_id: int) -> List[int]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_stance_id(self, author_id: int, discussion_id: int, post_id: int, record: Sequence[str]) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def get_stance_name(self, discussion_id: int, stance_id: int) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def get_response_type(self, record: Sequence[str]) -> str:
        raise NotImplementedError

    def get_parent_id(self, raw_parent_id: str, discussion_id: int) -> Optional[int]:
        if raw_parent_id == self.ROOT_PARENT_ID:
            return None

        return int(raw_parent_id)

    @abc.abstractmethod
    def get_author_stance_id(self, author_id: int, discussion_id: int, post_id: int, record: Sequence[str]) -> int:
        raise NotImplementedError

    @staticmethod
    def get_is_parent_missing(record: Sequence[str], parent_id: Optional[int]) -> bool:
        idx = IACRecordsLoader.PARENT_MISSING_INDEX
        return bool(int(record[idx])) or bool(parent_id)


class IACRecordsLoaderWithAuthorStanceInfer(IACRecordsLoader, metaclass=DelegatingMeta):

    def __init__(self, iac_records_loader: IACRecordsLoader):
        self._delegate = iac_records_loader

    def load_post_records(self) -> Iterable[IACPostRecord]:
        records = self._delegate.load_post_records()
        for discussion_id, discussion_records in groupby(records, key=lambda p: p.discussion_id):
            records_with_author_stance = self.__infer_authors_stances(discussion_records)
            yield from records_with_author_stance

    def __infer_authors_stances(self, discussion_records: Iterable[IACPostRecord]) -> Iterable[IACPostRecord]:
        records = list(discussion_records)
        discussion_id = records[self.DISCUSSION_ID_INDEX].discussion_id
        authors_stances: Dict[Any, List[int]] = {}
        stance_names = {}
        for record in records:
            author = record.author_id
            posts_stances = authors_stances.setdefault(author, [])
            post_stance = record.stance_id
            if post_stance >= 0:
                posts_stances.append(post_stance)
                if post_stance not in stance_names:
                    stance_names[post_stance] = record.stance_name

        # Aggregate stance per author
        authors_agg_stance = {author: get_most_common(stances) if len(stances) > 0 else self.UNKNOWN_STANCE_VALUE
                              for author, stances in authors_stances.items()}

        for record in records:
            author_stance_id = authors_agg_stance[record.author_id]
            author_stance_name = self.get_stance_name(discussion_id, author_stance_id) if author_stance_id >= 0 else "unknown"
            new_record = copy_namedtuple(record, author_stance_id=author_stance_id, author_stance_name=author_stance_name)
            yield new_record


class RootlessIACRecordsLoader(IACRecordsLoader, abc.ABC):

    def __init__(self):
        super().__init__()
        self.__artificial_roots: Dict[int, IACPostRecord] = {}

    def load_post_records(self) -> Iterable[IACPostRecord]:
        records = super().load_post_records()
        for discussion_id, discussion_records in groupby(records, key=lambda p: p.discussion_id):
            yield self.__get_artificial_root(discussion_id)
            yield from discussion_records

    def get_parent_id(self, raw_parent_id: str, discussion_id: int) -> int:
        parent_id = super().get_parent_id(raw_parent_id, discussion_id)
        if parent_id is None:
            parent_id = self.__get_artificial_root(discussion_id).post_id

        return parent_id

    def __get_artificial_root(self, discussion_id: int) -> IACPostRecord:
        artificial_root = self.__artificial_roots.get(discussion_id)
        if artificial_root is None:
            metadata = self.get_discussion_metadata(discussion_id)
            artificial_root = self.create_discussion_artificial_root(metadata)
            self.__artificial_roots[discussion_id] = artificial_root

        return artificial_root

    @staticmethod
    def create_discussion_artificial_root(discussion: DiscussionMetadata) -> IACPostRecord:
        title = discussion.record.title
        return IACPostRecord(
            discussion.topic_id, discussion.topic_str, discussion.discussion_id, -discussion.discussion_id,
            discussion.record.op, "", None, False, title, title, [], -1, "neutral", "root", discussion.record.url, -1, "unknown")


def build_iac_conversations(post_records: Iterable[IACPostRecord]) -> Iterable[Conversation]:
    parser = IACConversationParser()
    for discussion_id, posts in groupby(post_records, key=lambda r: r.discussion_id):
        posts = list(posts)
        conversation = parser.parse((discussion_id, posts))
        yield conversation

"""
    def iter_raw_records(self) -> Iterable[Sequence[str]]:
        return self._delegate.iter_raw_records()

    def get_discussion_metadata(self, discussion_id: int):
        return self._delegate.get_discussion_metadata(discussion_id)

    def get_text_mapping(self) -> Dict[int, str]:
        return self._delegate.get_text_mapping()

    def get_quotes(self, discussion_id: int, post_id: int) -> List[int]:
        return self._delegate.get_quotes(discussion_id, post_id)

    def get_stance_id(self, author_id: int, discussion_id: int, post_id: int, record: Sequence[str]) -> int:
        return self._delegate.get_stance_id(author_id, discussion_id, post_id, record)

    def get_stance_name(self, discussion_id: int, stance_id: int) -> str:
        return self._delegate.get_stance_name(discussion_id, stance_id)

    def get_response_type(self, record: Sequence[str]) -> str:
        return self._delegate.get_response_type(record)

    def get_author_stance_id(self, author_id: int, discussion_id: int, post_id: int, record: Sequence[str]) -> int:
        return self._delegate.get_author_stance_id(author_id, discussion_id, post_id, record)
        """