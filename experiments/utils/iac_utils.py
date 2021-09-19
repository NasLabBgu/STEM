import os
from tqdm import tqdm

from typing import TextIO, Tuple, Dict


def parse_next_line(fd: TextIO) -> Tuple[int, str]:
    """
    parse next text record that can spread on multiple lines, into a pair of text_id and the text itself.
    """
    text_record = []
    for line in tqdm(fd):
        line = line.strip()
        if len(line) >= 1 and line[-1] == '\\':
            text_record.append(line[:-2])
            continue

        if len(text_record) == 0:
            text_record.append(line.strip())

        try:
            text_id, first_text_part = text_record[0].split(maxsplit=1)
        except ValueError:
            text_id = text_record[0]
            first_text_part = ""

        text_id = int(text_id.strip())
        text_record[0] = first_text_part
        text = "".join(text_record).strip()
        return text_id, text


def parse_cd_text_file(path: str) -> Dict[int, str]:
    text_by_id = {}
    with open(path, 'r') as text_f:
        while (next_record := parse_next_line(text_f)) is not None:
            text_id, text = next_record
            text_by_id[text_id] = text

        return text_by_id