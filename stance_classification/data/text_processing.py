import csv
from typing import List, Tuple, Union

from stance_classification.data.twokenize import tokenizeRawTweetText

ASCII_UPPER = range(65, 91)
ASCII_LOWER = range(97, 123)
ASCII_DIGITS = range(48, 58)
ASCII_UPPER_LOWER_DIFF = ASCII_LOWER[0] - ASCII_UPPER[0]


def is_delimiter(c: str) -> bool:
    """
    :param c: c must be a char (i.e a size 1 string)
    :return:
    """
    return c.isspace()


def is_punctuation(c: str) -> bool:
    return not (c.isspace() or c.isalnum())


def to_ascii_lower(c: str) -> str:
    """
    :param c: c must be a char (i.e a size 1 string)
    :return:
    """
    char_decimal = ord(c)
    if char_decimal in ASCII_UPPER:
        return chr(char_decimal + ASCII_UPPER_LOWER_DIFF)

    return c


def is_punctuation_only(token: str) -> bool:
    return not any(map(str.isalnum, token))


def fix_longation(token: str) -> Union[None, str]:
    found_longation = False
    new_token = []
    similar_prev = 0
    for i, c in enumerate(token):
        if (i > 0) and (c == token[i - 1]):
            similar_prev += 1
            if similar_prev == 2:
                found_longation = True
                new_token.pop()
            elif similar_prev < 2:
                new_token.append(c)

            continue

        similar_prev = 0
        new_token.append(c)

    if found_longation:
        return "".join(new_token)

    return None


def fix_end_longation(token: str) -> Union[None, str]:
    if len(token) < 3:
        return None

    if not ((token[-1] == token[-2]) and (token[-2] == token[-3])):
        return None

    i = len(token) - 4
    while (i >= 0) and (token[i] == token[i + 1]):
        i -= 1

    return token[:i + 1] + token[i + 1]


def fix_repeat(token: str, max_no_repeat: int = 1) -> Union[None, str]:
    """
    :param token: assumed to be a sequence of punctuation characters only!
    :param max_no_repeat:
    :return:
    """
    if len(token) <= max_no_repeat:
        return None

    return token[-1]

    # if "." in token and len(token) > max_no_repeat:
    #     return "."
    #
    # if "?" in token and len(token) >= 3:
    #     return "?"
    #
    # if len(set(token)) == 1:
    #     return token[0]

    return None


OTHER_SMILES = {"(:", ":')", "=)", ";-)"}
OTHER_EMOTICONS = {":s", "xxx", "^m", "xx", ":]", "xd", "^^", "o.o", "x"}
LOL = {":p", "=p"}
NEUTRAL = {"-_-"}
SAD = {":-("}


def detect_emoticon2(token: str) -> Union[None, str]:
    pass


def detect_emoticon(token: str) -> Union[None, str]:
    if token in OTHER_EMOTICONS:
        return token

    if len(token) <= 1:
        return None

    eyes = token[0]
    if eyes == ":" or eyes == ";":
        mouth = token[1]
        if mouth == ")" or mouth == "d":
            return "<SMILE>"
        if mouth == "(":
            return "<SADFACE>"
        if mouth == "p":
            return "<LOLFACE>"
        if mouth == "/":
            return "<NEUTRALFACE>"

    if token in OTHER_SMILES:
        return "<SMILE>"
    if token in SAD:
        return "<SADFACE>"
    if token in LOL:
        return "<LOLFACE>"
    if token in NEUTRAL:
        return "<NEUTRALFACE>"

    if token.startswith("<3"):
        return "<HEART>" + token[2:]

    if all(map(lambda c: c == "x", token)):
        return token

    return None


def is_mention(token: str) -> bool:
    return token.startswith("@") and len(token) > 1


def is_hashtag(token: str) -> bool:
    return token.startswith("#")


def is_url(token: str) -> bool:
    return token.startswith("http") or token.startswith("www")


def token_to_lower(token: str) -> str:
    return "".join(map(to_ascii_lower, token))


def fix_numbers(token: str) -> str:
    # check if starts with a number
    new_token = []
    digits_scope = False
    for i, c in enumerate(token):
        if c.isdigit():
            digits_scope = True
            continue

        if digits_scope:
            new_token.append("<NUMBER>")
            digits_scope = False

        new_token.append(c)

    if digits_scope:
        new_token.append("<NUMBER>")

    return "".join(new_token)


def split_token(token: str) -> List[str]:
    splitted = []
    start = 0
    punct_scope = False
    for i, c in enumerate(token):
        if c.isalpha():
            if punct_scope:
                splitted.append(token[start: i])
                start = i

            continue

        if (not punct_scope) and (i > 0):
            splitted.append(token[start: i])
            start = i
            punct_scope = True

    splitted.append(token[start:])
    return splitted


def parse_tokens(tokens: List[str]) -> List[str]:
    parsed = []
    for token in tokens:
        token = token_to_lower(token)

        if is_mention(token):
            parsed.append("<USER>")
            continue
        if is_hashtag(token):
            parsed.append("<HASHTAG>")
            continue
        if is_url(token):
            parsed.append("<URL>")
            continue

        emoticon = detect_emoticon(token)
        if emoticon is not None:
            parsed.append("<EMOTICON>")
            token = emoticon
            # if len(emoticon) > 0:
            #     parsed.append(emoticon)
            #     if len(residual) == 0:
            #         continue

            # token = residual

        token = fix_numbers(token)

        if is_punctuation_only(token):
            max_no_repeat = 1 if emoticon is None else 2
            new_token = fix_repeat(token, max_no_repeat)
            mark = "<REPEAT>"
        else:
            new_token = fix_end_longation(token)
            mark = "<ELONG>"

        if new_token is not None:
            parsed.append(new_token)
            parsed.append(mark)
            continue

        parsed.append(token)

    return parsed


def process_text(text: str) -> str:
    tokens = tokenizeRawTweetText(text)
    tokens = parse_tokens(tokens)
    return " ".join(tokens)


if __name__ == "__main__":

    text = "@MonikaRozynek http://twitpic.com/3jmwmd - AAWWWWWW:') i have a black lab :')"
    process_text(text)

    ignore = {136, 138, 140, 141, 152, 173, 181, 214, 223, 227, 235, 338, 344, 351}

    cleaning_compare_filepath = "twitter.data"
    with open(cleaning_compare_filepath, 'r', encoding='utf8') as f:
        reader = csv.reader(f, delimiter='\t')
        for i, record in enumerate(reader):
            if i in ignore:
                continue
            raw_text = record[3]
            expected = record[4]
            actual = process_text(raw_text)
            if expected != actual:
                print(f"row {i}")
                print(f"original: {raw_text}\n")
                print(f"expected: {expected}")
                print(f"actual:   {actual}")
                break
