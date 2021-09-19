from operator import itemgetter
from typing import Tuple, Dict, List, Union, Set
import spacy
from spacy.tokens import Doc, Token
from nltk.corpus import stopwords


from spacy.symbols import nsubj, nsubjpass, csubj, csubjpass, acl, \
    dobj, iobj, nmod, pobj, ccomp, acomp, prep, \
    xcomp, auxpass, aux,\
    NOUN, ADJ, VERB, PROPN, PUNCT

from experiments.utils.debug_utils import dprint, dprint2

RELEVANT_POS_TAGS = {NOUN, ADJ, VERB, PROPN}
SUBJ_RELATIONS = {nsubj, nsubjpass, csubj, csubjpass, acl}
OBJ_RELATIONS = {dobj, iobj, nmod, ccomp, acomp, pobj, prep}
RELEVANT_DEP_TAGS = SUBJ_RELATIONS | OBJ_RELATIONS


parser = spacy.load("en_core_web_sm")

STOPWORDS = stopwords.words()
# STOPWORDS = parser.Defaults.stop_words


# TOP1K_PATH = ""


# def load_top1k_en_words() -> Set[str]:
#     with open(TOP1K_PATH, 'r') as f:
#         return set(map(str.strip, f))


# TOP1k_EN_WORDS = load_top1k_en_words()


def get_relevant_tokens(root: Token) -> Dict[int, Token]:
    relevant_tokens: Dict[int, Token] = {}
    if root.pos in RELEVANT_POS_TAGS:
        relevant_tokens[root.i] = root

    for c in root.children:
        relevant_tokens.update(get_relevant_tokens(c))

    return relevant_tokens


# def is_valid_chunk(chunk: List[Token]) -> bool:
#     if len(chunk) == 0:
#         return False
#     if len(chunk) == 1:
#         token = chunk[0]
#         if token.pos == VERB:
#             return False
#         if token.lemma_ in TOP1k_EN_WORDS or token.lower_ in TOP1k_EN_WORDS:
#             return False
#         if token.pos == PROPN:
#             return False
#
#     return True


# def get_chunks(tokens_by_position: Dict[int, Token]) -> List[str]:
#     prev_i = -2
#     chunks = []
#     current_chunk = []
#     for i, token in sorted(tokens_by_position.items(), key=itemgetter(0)):
#         if prev_i == i - 1:
#             if token.pos != VERB:
#                 current_chunk.append(token)
#         else:
#             if is_valid_chunk(current_chunk):
#                 chunk_str = " ".join([t.lower_ for t in current_chunk])
#                 chunks.append(chunk_str)
#
#             current_chunk = [token]
#
#         prev_i = i
#
#     if is_valid_chunk(current_chunk):
#         chunk_str = " ".join([t.lower_ for t in current_chunk])
#         chunks.append(chunk_str)
#
#     return chunks


# def get_relevant_nps(root: Token) -> List[List[Token]]:
#     nps = []
#     xcomp_root: Union[Token, None] = None
#     aux_root: Union[Token, None] = None
#     auxpass_root: Union[Token, None] = None
#     found_subj = False
#     found_obj = False
#     for c in root.children:
#         # dprint(c.text)
#         if c.dep in SUBJ_RELATIONS and not found_subj:
#             relevant_tokens = get_relevant_tokens(c)
#             subj_chunks = get_chunks(relevant_tokens)
#             nps.append(subj_chunks)
#             found_subj = True
#         elif c.dep in OBJ_RELATIONS and not found_obj:
#             relevant_tokens = get_relevant_tokens(c)
#             obj_chunks = get_chunks(relevant_tokens)
#             nps.append(obj_chunks)
#             found_obj = True
#         elif c.dep == xcomp:
#             xcomp_root = c
#         elif c.dep == aux:
#             aux_root = c
#         elif c.dep == auxpass:
#             auxpass_root = c
#
#     if not (found_subj or found_obj):
#         if aux_root is not None:
#             dprint("Try in aux subtree")
#             nps.extend(get_relevant_nps(aux_root))
#         if auxpass_root is not None:
#             dprint("Try in auxpass subtree")
#             nps.extend(get_relevant_nps(auxpass_root))
#         if xcomp_root is not None:
#             dprint("Try in xcomp subtree")
#             nps.extend(get_relevant_nps(xcomp_root))
#
#     return [np for np in nps if len(np) > 0]


# def get_relevant_nps_from_text(doc: Doc) -> List[List[Token]]:
#     nps_per_sentence = []
#     for sent in doc.sents:
#         dprint(sent)
#         dprint2([(t.text, t.pos_, f"{t.head.text} -> {t.dep_}") for t in sent])
#         nps = get_relevant_nps(sent.root)
#         nps_per_sentence.extend(nps)
#
#     return nps_per_sentence


def is_valid_token(token: Token) -> bool:
    if token.lower_ in STOPWORDS:
        return False
    if token.pos == PUNCT:
        return False

    return True


def get_tokens_with_stopwords(doc: Doc) -> Tuple[List[List[str]], List[List[str]]]:
    all_tokens, all_pos = [], []
    for sent in doc.sents:
        tokens, pos = [], []
        for token in sent:
            if is_valid_token(token):
                tokens.append(token.lower_)
                pos.append(token.tag_)

        all_tokens.append(tokens)
        all_pos.append(pos)

    return all_tokens, all_pos


def parse_text(text: str) -> Doc:
    return parser(text)


if __name__ == "__main__":
    DEBUG_LEVEL = 1
    text = "Regulation of corporations has been subverted by corporations. States that incorporate corporations are " \
           "not equipped to regulate corporations that are rich enough to influence elections, are rich enough to " \
           "muster a legal team that can bankrupt the state. Money from corporations and their principals cannot be " \
           "permitted in the political process if democracy is to survive."

    text = "I totally agree with this premise. As a younger person I was against Nuclear power (I was in college " \
           "during 3 mile island) but now it seems that nuclear should be in the mix. Fission technology is better, " \
           "and will continue to get better if we actively promote its development. The prospect of fusion energy " \
           "also needs to be explored. If it's good enough for the sun and the stars, it's good enough for me."

    text = "Guns should be banned because they are not needed in any domestic issue. " \
           "The second ammendment was put in place because of fear that the british might invade america again or " \
           "take control of the government. " \
           "if this were the case the people would need weapons to defend themselves and regain america. " \
           "The british aren't going to invade so we don't need to protect our selves. even in the this day and age " \
           "america remains increadible safe compared to many other nations. we have no close enemies. " \
           "if a major army were to attack us a few men with pistols or shotguns wouldn't do much against a soldier " \
           "with an ak47 or tanks or bombersGuns in America just make it easier for crimes to be committed. " \
           "Some guns should never be considered allowed and this includes all semi automatic weapons as well as " \
           "shotgunsPoverty, drugs, and lack of education are the reasons people turn to guns to kill. " \
           "guns give you power to take life and should not be allowed to float around so that our students or " \
           "citizens can use them against one another"

    text = """Gun control is misguided.""" #When guns become illegal for law-abiding citizens, only criminals will have guns. If they're already criminals, they aren't following certain laws, and will almost certainly ignore more if they desire to do so. Restrictions and bans on the availability of guns to the normal populace only encourages crime, because it allows criminals to operate with more impunity, knowing that they cannot be harmed significantly by the victim of their crime. And no matter how good the police are, they cannot cover all potential crime scenes. Law-abiding citizens carrying guns can defend themselves and others in their immediate area more completely than even the best police force. Prohibiting guns from the good guys just means that the bad guys are safer"""

    doc = parse_text(text)
    tokens, pos = get_tokens_with_stopwords(doc)
    dprint(" ".join(" ".join(sent) + "." for sent in tokens))
    dprint(repr(tokens).replace("'", '"'))
    dprint(repr(pos).replace("'", '"'))
    # all_nps = get_relevant_nps_from_text(doc)
    # for sent_nps in all_nps:
    #     dprint(sent_nps)