import numpy as np
import pandas as pd

from io_utils import (
    print_eda,
    read_csv_or_zip,
    read_parquet_file,
    read_tsv,
)


LIAR_COLUMNS = [
    "id",
    "label",
    "statement",
    "subjects",
    "speaker",
    "job_title",
    "state",
    "party_affiliation",
    "barely_true_count",
    "false_count",
    "half_true_count",
    "mostly_true_count",
    "pants_fire_count",
    "context",
]


def load_isot():
    """Load and prepare the ISOT fake-vs-real dataset exactly as in the original code."""
    fake_df = read_csv_or_zip([
        "Fake.csv",
        "fake.csv",
        "Fake.csv.zip",
        "fake.csv.zip",
    ])
    true_df = read_csv_or_zip([
        "True.csv",
        "true.csv",
        "True.csv.zip",
        "true.csv.zip",
        "True.csv(2).zip",
    ])

    fake_df["label"] = "fake"
    true_df["label"] = "real"

    isot_df = pd.concat(
        [fake_df, true_df],
        ignore_index=True,
    )
    isot_df["combined_text"] = (
        isot_df["title"].astype(str)
        + " "
        + isot_df["text"].astype(str)
    )

    print_eda(
        name="ISOT",
        df=isot_df,
        text_column="combined_text",
        label_column="label",
    )

    return isot_df


def load_liar():
    """Load the official LIAR train/validation/test TSV files."""
    liar_train = read_tsv(
        "train.tsv",
        names=LIAR_COLUMNS,
    )
    liar_valid = read_tsv(
        "valid.tsv",
        names=LIAR_COLUMNS,
    )
    liar_test = read_tsv(
        "test.tsv",
        names=LIAR_COLUMNS,
    )

    liar_full = pd.concat(
        [liar_train, liar_valid, liar_test],
        ignore_index=True,
    )

    print_eda(
        name="LIAR",
        df=liar_full,
        text_column="statement",
        label_column="label",
    )

    print("\nTop party affiliations:")
    print(
        liar_full["party_affiliation"]
        .value_counts()
        .head(20)
    )

    return liar_train, liar_valid, liar_test, liar_full


def clean_bias_labels(label_series):
    """Convert bias labels into left/center/right text labels if required."""
    mapping = {
        0: "left",
        1: "center",
        2: "right",
        "0": "left",
        "1": "center",
        "2": "right",
    }

    def clean_one(value):
        if pd.isna(value):
            return np.nan
        if value in mapping:
            return mapping[value]

        value_str = str(value).lower().strip()

        if value_str in mapping:
            return mapping[value_str]
        if "left" in value_str:
            return "left"
        if "center" in value_str or "centre" in value_str:
            return "center"
        if "right" in value_str:
            return "right"

        return value_str

    return label_series.apply(clean_one)


def prepare_article_bias_dataframe(df):
    """Prepare the Baly article political ideology dataframe for classification."""
    prepared = df.copy()

    if "bias_text" in prepared.columns:
        label_column = "bias_text"
    elif "bias" in prepared.columns:
        label_column = "bias"
    else:
        raise KeyError(
            "No bias label column found. Expected 'bias_text' or 'bias'."
        )

    prepared["bias_label"] = clean_bias_labels(
        prepared[label_column]
    )

    if "content" in prepared.columns:
        main_text = prepared["content"].astype(str)
    elif "content_original" in prepared.columns:
        main_text = prepared["content_original"].astype(str)
    else:
        raise KeyError(
            "No article text column found. "
            "Expected 'content' or 'content_original'."
        )

    if "title" in prepared.columns:
        prepared["combined_text"] = (
            prepared["title"].fillna("").astype(str)
            + " "
            + main_text
        )
    else:
        prepared["combined_text"] = main_text

    prepared["combined_text"] = (
        prepared["combined_text"]
        .astype(str)
        .str.strip()
    )
    prepared = prepared.dropna(
        subset=["bias_label", "combined_text"]
    )
    prepared = prepared[
        prepared["combined_text"].str.len() > 20
    ]

    return prepared


def load_article_bias():
    """Load and prepare the predefined Baly train/validation/test parquet files."""
    bias_train_raw = read_parquet_file([
        "train-00000-of-00001.parquet",
        "bias_train.parquet",
        "train.parquet",
    ])
    bias_valid_raw = read_parquet_file([
        "valid-00000-of-00001.parquet",
        "validation-00000-of-00001.parquet",
        "bias_valid.parquet",
        "bias_validation.parquet",
        "valid.parquet",
        "validation.parquet",
    ])
    bias_test_raw = read_parquet_file([
        "test-00000-of-00001.parquet",
        "bias_test.parquet",
        "test.parquet",
    ])

    bias_train = prepare_article_bias_dataframe(
        bias_train_raw
    )
    bias_valid = prepare_article_bias_dataframe(
        bias_valid_raw
    )
    bias_test = prepare_article_bias_dataframe(
        bias_test_raw
    )
    bias_full = pd.concat(
        [bias_train, bias_valid, bias_test],
        ignore_index=True,
    )

    print_eda(
        name="Article Political Bias",
        df=bias_full,
        text_column="combined_text",
        label_column="bias_label",
    )

    print("\nArticle bias dataset sources:")
    if "source" in bias_full.columns:
        print(
            bias_full["source"]
            .value_counts()
            .head(20)
        )

    return bias_train, bias_valid, bias_test, bias_full


def simplify_party(party):
    """Simplify LIAR party affiliation for the optional appendix experiment."""
    party = str(party).lower().strip()

    if "democrat" in party:
        return "democratic"
    if "republican" in party:
        return "republican"

    return "other_or_neutral"
