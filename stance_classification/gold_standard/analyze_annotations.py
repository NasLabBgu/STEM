import re
from collections import Counter
from itertools import starmap
from operator import itemgetter

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

from matplotlib import pyplot as plt
import seaborn as sns

QUOTE_PATT = re.compile("<quote>.+</quote>", re.DOTALL)
QUOTE_TAGS_SIZE = len("<quote></quote>")


def load_annotation_results(path: str):
    return pd.read_csv(path)


def join_labels_to_annotations(annotations_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    df = annotations_df.merge(labels_df, left_on=["Input.tree_id", "Input.user_name"], right_on=["tree_id", "username"])
    return df


def quote_portion(text: str) -> float:
    num_quotes = 0
    quotes_len = 0
    for quote in QUOTE_PATT.finditer(text):
        num_quotes += 1
        quotes_len += (quote.end() - quote.start()) - QUOTE_TAGS_SIZE

    quote_ratio = quotes_len / (len(text) - (num_quotes * QUOTE_TAGS_SIZE))
    return quote_ratio


def calc_reply_weight(textlen, non_quote_ratio, depth, work_time):
    if depth > 3:
        return 0
    if textlen > 2500:
        return 0

    expected_time = np.sqrt(textlen)
    time_diff = abs(expected_time - work_time)
    if work_time < expected_time:
        time_factor = time_diff / (expected_time / 6)
    else:
        time_factor = time_diff / (expected_time / 2)

    # time_factor = max(1, time_factor)
    return 1 / time_factor


def is_relevant_answers(reply_answers):
    reply_answers = [a / abs(a) if ((not np.isnan(a)) and a != 0) else a for a in reply_answers]
    majority, max_count = Counter(reply_answers).most_common(1)[0]
    if max_count == 1:
        return False

    return (majority == 1) or (majority == -1) or (majority == 0)


def get_user_answer(claim, replies, text_lenghts, non_quote_ratios, depths, work_times, answers, gs):
    if len(claim[0]) > 120:
        calculated_answer = 0
    # elif len(replies) == 3:
    #     calculated_answer = 0
    else:
        weights = np.array(list(starmap(calc_reply_weight, zip(text_lenghts, non_quote_ratios, depths, work_times))))
        # relevant_replies = [b for i in range(0, len(answers), 3) for b in [is_relevant_answers(answers[i: i+3])]*3]
        # weights = np.multiply(relevant_replies, weights)
        calculated_answer = np.nansum(np.multiply(answers, weights))

    if calculated_answer != 0:
        calculated_answer -= 0.35
    if calculated_answer != 0:
        print(gs[0], calculated_answer)

    gs = int(gs[0])
    calculated_answer = 0 if calculated_answer == 0 else int(calculated_answer / abs(calculated_answer))
    return gs, calculated_answer


def analyze_results(df: pd.DataFrame):
    answers_map = {"Supports": 1, "Highly Supports": 2, "Neutral": 0, "Not clear / Off-topic": np.nan, "Rejects": -1,
                   "Strongly Rejects": -2}

    df["answer"] = df["Answer.stance.label"].map(answers_map)
    df["gs"] = df["label"].map(answers_map)
    df["textlen"] = df["Input.text"].map(len)

    text_lengths_hist = np.histogram(df["textlen"])
    # sns.scatterplot("textlen", "WorkTimeInSeconds", data=df)
    # plt.show()


    print(df[["Input.tree_id", "Input.user_name", "Input.claim", "Input.text", "textlen", "Input.reply_depth", "answer",
              "gs"]].head(20))

    # quotes filter
    quotes_mask = df["Input.text"].str.contains("</quote>")
    non_quote_ratio = df["Input.text"].map(lambda t: 1 - quote_portion(t))
    df["non_quote_rario"] = non_quote_ratio

    # check results by depth
    answers_match = np.multiply(df["answer"], df["gs"])
    # non_quote_answers_match = np.divide(answers_match, df["non_quote_rario"])
    non_quote_answers_match = np.multiply(answers_match, df["non_quote_rario"])
    for depth in range(1, 6):
        depth_mask = np.asarray(df["Input.reply_depth"] == depth)
        print(f"depth: {depth}\tnum replies: {np.sum(depth_mask)}\tmatch_score: {np.sum(answers_match[depth_mask])} " \
              + f"non-quote-score: {np.sum(non_quote_answers_match[depth_mask])}")

    users_data_df = df[["Input.tree_id", "Input.user_name", "Input.claim", "Input.text", "textlen", "non_quote_rario",
                        "Input.reply_depth", "WorkTimeInSeconds", "answer", "gs"]]
    tree_user_df = users_data_df.groupby(["Input.tree_id", "Input.user_name"]).agg(list)
    gs, final_answers = zip(*list(starmap(get_user_answer, map(itemgetter(1), tree_user_df.iterrows()))))
    final_answers = np.asarray(final_answers)
    answers_mask = final_answers != 0

    gs = np.asarray(gs)[answers_mask]
    final_answers = final_answers[answers_mask]

    print(classification_report(gs, final_answers))


if __name__ == "__main__":
    annotation_path = "/home/<user>/data/bgu/stance_annotation/Batch_3886298_batch_results.csv"
    gs_path = "/home/<user>/data/bgu/labeled/stance_gs_str.csv"

    annotations_df = pd.read_csv(annotation_path)
    labels_df = pd.read_csv(gs_path)

    annotations_df = join_labels_to_annotations(annotations_df, labels_df)
    pd.options.display.width = 0

    analyze_results(annotations_df)
