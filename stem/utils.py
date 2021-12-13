
import json
import re
from collections import deque
from itertools import islice
from typing import List, Tuple, Iterator, Iterable

USER_MENTION_PATTERN = re.compile(r"/u/[\w-]+", re.UNICODE)  # https://github.com/reddit-archive/reddit/blob/master/r2/r2/lib/validator/validator.py#L1570
QUOTE_PATTERN = re.compile(r"<quote>.*</quote>")

MENTION_PREFIX = "/u/"
QUOTE_START_SYMBOL = "<quote>"
QUOTE_END_SYMBOL = "</quote>"


def iter_trees_from_lines(data_path: str) -> Iterable[str]:
    """
    iterates a trees from a file where each line represent a whole conversation tree.
    :param data_path: a path to file with a tree per line.
    :return: An iterable of raw trees (i.e not parsed) as displayed in the file.
    """
    with open(data_path, 'r') as f:
        yield from f


def iter_trees_from_jsonl(data_path: str) -> Iterable[dict]:
    trees_as_json = iter_trees_from_lines(data_path)
    yield from map(json.loads, trees_as_json)


def find_user_mentions(text: str) -> List[Tuple[int, int]]:
    """
    find mentions of users in text from reddit
    :param text: text to search mentions.
    :return: list with pairs of indices (begin_index, end_index)
    """
    return [m.span() for m in USER_MENTION_PATTERN.finditer(text)]


def strip_mention_prefix(mention: str) -> str:
    """
    strips the prefix of a mention (i.e /u/)
    :param mention: mention to strip,
                  if 'mention' doesn't contain the prefix of a mention, the mention will be returned unchanged.
    :return: mention without its prefix.
    """
    start_indx = len(MENTION_PREFIX) if mention.startswith(MENTION_PREFIX) else 0
    return mention[start_indx:]


def find_quotes(text: str) -> List[Tuple[int, int]]:
    return [(m.pos, m.endpos) for m in QUOTE_PATTERN.finditer(text)]


def strip_quote_symbols(quote: str) -> str:
    """
    strips the prefix and suffix of a found quote (i.e <quote> and </quote> respectively)
    :param quote: quote to strip,
                  if the quote doesn't contain the prefix and suffix symbols of a quote, the quote won't be changed.
    :return: quote without its prefix and suffix symbols.
    """
    start_index = len(QUOTE_START_SYMBOL) if quote.startswith(QUOTE_START_SYMBOL) else 0
    end_offset = len(QUOTE_END_SYMBOL) if quote.startswith(QUOTE_END_SYMBOL) else 0
    end_index = len(quote) - end_offset
    return quote[start_index:end_index]


def is_source_of_quote(quote: str, text: str) -> bool:
    """
    check if 'text' is the actual source of 'quote' (i.e the original text from which 'quote' was taken)
    :param quote:
    :param text:
    :return: True if text is the original text of quote, False otherwise.
    """
    quote_pos = text.find(quote)
    if quote_pos > -1:

        # check if the found text is also a quote by searching <quote> symbol before the found text.
        quote_symbol_offset = quote_pos - len(QUOTE_START_SYMBOL)

        if quote_symbol_offset < 0:
            return True

        if text[quote_symbol_offset: quote_pos] != QUOTE_START_SYMBOL:
            return True

    return False


def skip_elements(it: Iterable, num_skip: int):
    """
    skip [num_skip] elements from iterator [it]
    :param it:
    :param num_skip:
    :return:
    """
    deque(islice(it, num_skip))
