# MIT License
#
# Copyright (c) 2025 LoongMa
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import json
import re

import pandas as pd


def read_jsonl(file_path: str):
    """read jsonl file to list of dict"""

    data = []
    with open(file_path, encoding="utf-8") as fin:
        for line in fin:
            if line.strip():  # skip empty line
                data.append(json.loads(line))
    if len(data) > 1:
        if isinstance(data[0], dict):
            print("data[0].keys()", data[0].keys())
    return data


def read_jsonl_to_df(file_path: str):
    """read jsonl file to pandas DataFrame"""
    data = []
    with open(file_path, encoding="utf-8") as fin:
        for line in fin:
            if line.strip():  # skip empty line
                data.append(json.loads(line))
    if len(data) > 1:
        if isinstance(data[0], dict):
            print("data[0].keys()", data[0].keys())
    df = pd.DataFrame(data)
    print(df.shape, df.columns)
    return df


def detect_language(input_text: str):
    if isinstance(input_text, str):
        s = input_text.strip()
        if re.search(r"[üöäß]|[ÄÖÜ]", s) or re.search(
            r"\b(der|die|das|und|ein|nicht|ist|zu|in|den|mit|auf|für|sie|es|dem|von)\b", s, re.I
        ):
            return "ger"
        if re.search(r"[\u3040-\u309F\u30A0-\u30FF]", s):
            return "jap"
        if re.search(r"[a-zA-Z]", s):
            ascii_ratio = sum(1 for c in s if ord(c) < 128) / max(1, len(s))
            if ascii_ratio > 0.7:
                return "en"
    return "zh"


def merge_excel(path1: str, path2: str, on_col: list | str, how="left"):
    df1 = pd.read_excel(path1)
    print("df1.shape, df1.columns", df1.shape, df1.columns)
    df2 = pd.read_excel(path2)
    print("df2.shape, df2.columns", df2.shape, df2.columns)
    df = pd.merge(df1, df2, on=on_col, how=how)
    print("df.shape, df.columns", df.shape, df.columns)
    return df


def concat_excel_file(path1: str, path2: str):
    df1 = pd.read_excel(path1)
    print("df1.shape, df1.columns", df1.shape, df1.columns)
    df2 = pd.read_excel(path2)
    print("df2.shape, df2.columns", df2.shape, df2.columns)
    df = pd.concat([df1, df2], ignore_index=True)
    print("df.shape, df.columns", df.shape, df.columns)
    return df


def concat_df(df1: pd.DataFrame, df2: pd.DataFrame):
    print("df1.shape, df1.columns", df1.shape, df1.columns)
    print("df2.shape, df2.columns", df2.shape, df2.columns)
    concat_df = pd.concat([df1, df2], ignore_index=True)
    print("concat_df.shape, concat_df.columns", concat_df.shape, concat_df.columns)
    return concat_df


def read_excel(path):
    df = pd.read_excel(path)
    print("df.shape, df.columns", df.shape, df.columns)
    return df


def dedup(df: pd.DataFrame, subset_cols: list):
    # mark duplicate rows, not drop directly
    df["is_duplicate"] = df.duplicated(subset=subset_cols, keep=False)
    df_de = df.drop_duplicates(subset=subset_cols)
    return df_de, df


def count_score_and_proportion(df: pd.DataFrame, score_threshold: float = 8.0):
    # deascend sort by score, high score first
    score_counts = (
        df["score"]
        .value_counts(ascending=False)
        .to_frame()
        .reset_index()
        .rename(columns={"index": "score", "score": "count"})
    )
    print(score_counts)

    # statistic proportion of score > score_threshold
    above_some_value = df[df["score"] > score_threshold].shape[0]
    total = df.shape[0]
    proportion_above_score_threshold = above_some_value / total if total > 0 else 0
    print(f"the propo of score bigger than {score_threshold}: {proportion_above_score_threshold:.2%}")


def value_counts(df: pd.DataFrame, col_name: str):
    df_vc = (
        df[col_name]
        .value_counts(ascending=False)
        .to_frame()
        .reset_index()
        .rename(columns={"index": col_name, col_name: col_name})
    )
    tmp = list(set(df_vc[col_name].to_list()))
    print("---------------------------------------------")
    print(df_vc)
    print("------------value_counts unique values-------------------", col_name, len(tmp), tmp)
    print("------------value_counts shape, columns-------------------", df_vc.shape, df_vc.columns)
    return df_vc


def value_counts_by_language(df: pd.DataFrame, col_name: str):
    # groupby language and then statistic value_counts
    df_lan = df.groupby("language")[col_name].value_counts().to_frame("count")
    print(df_lan.to_string())

    return df_lan
