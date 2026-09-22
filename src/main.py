import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from config import (
    BIAS_BERT_TEST_SAMPLE,
    BIAS_BERT_TRAIN_SAMPLE,
    ISOT_BERT_TEST_SAMPLE,
    ISOT_BERT_TRAIN_SAMPLE,
    LIAR_BERT_TEST_SAMPLE,
    LIAR_BERT_TRAIN_SAMPLE,
    RANDOM_STATE,
    RUN_ARTICLE_BIAS_DATASET,
    RUN_LIAR_PARTY_PROXY,
    RUN_TRANSFORMER,
)
from datasets import (
    load_article_bias,
    load_isot,
    load_liar,
    simplify_party,
)
from evaluation import ensure_result_folders, save_final_outputs
from models import run_sklearn_models, run_transformer_model


def print_runtime_info():
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        print("No GPU detected. DistilBERT may run slowly on CPU.")


def run_isot(all_results):
    print("\n\n")
    print("#" * 80)
    print("PART 1: ISOT FAKE NEWS DATASET")
    print("#" * 80)

    isot_df = load_isot()

    X_isot = isot_df["combined_text"]
    y_isot = isot_df["label"]

    X_train_isot, X_test_isot, y_train_isot, y_test_isot = train_test_split(
        X_isot,
        y_isot,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_isot,
    )

    isot_results = run_sklearn_models(
        dataset_name="ISOT Fake vs Real",
        X_train=X_train_isot,
        X_test=X_test_isot,
        y_train=y_train_isot,
        y_test=y_test_isot,
    )
    all_results.extend(isot_results)

    if RUN_TRANSFORMER:
        isot_transformer_result = run_transformer_model(
            dataset_name="ISOT Fake vs Real",
            X_train=X_train_isot,
            X_test=X_test_isot,
            y_train=y_train_isot,
            y_test=y_test_isot,
            train_sample_size=ISOT_BERT_TRAIN_SAMPLE,
            test_sample_size=ISOT_BERT_TEST_SAMPLE,
        )
        all_results.append(isot_transformer_result)


def run_liar(all_results):
    print("\n\n")
    print("#" * 80)
    print("PART 2: LIAR DATASET")
    print("#" * 80)

    liar_train, liar_valid, liar_test, liar_full = load_liar()

    X_train_liar = liar_train["statement"].astype(str)
    y_train_liar = liar_train["label"].astype(str)

    X_test_liar = liar_test["statement"].astype(str)
    y_test_liar = liar_test["label"].astype(str)

    liar_results = run_sklearn_models(
        dataset_name="LIAR Truthfulness",
        X_train=X_train_liar,
        X_test=X_test_liar,
        y_train=y_train_liar,
        y_test=y_test_liar,
    )
    all_results.extend(liar_results)

    if RUN_TRANSFORMER:
        liar_transformer_result = run_transformer_model(
            dataset_name="LIAR Truthfulness",
            X_train=X_train_liar,
            X_test=X_test_liar,
            y_train=y_train_liar,
            y_test=y_test_liar,
            train_sample_size=LIAR_BERT_TRAIN_SAMPLE,
            test_sample_size=LIAR_BERT_TEST_SAMPLE,
        )
        all_results.append(liar_transformer_result)

    return liar_train, liar_test


def run_bias(all_results):
    if not RUN_ARTICLE_BIAS_DATASET:
        return

    print("\n\n")
    print("#" * 80)
    print("PART 3: ARTICLE-LEVEL POLITICAL BIAS DATASET")
    print("#" * 80)

    bias_train, bias_valid, bias_test, bias_full = load_article_bias()

    X_train_bias = bias_train["combined_text"].astype(str)
    y_train_bias = bias_train["bias_label"].astype(str)

    X_test_bias = bias_test["combined_text"].astype(str)
    y_test_bias = bias_test["bias_label"].astype(str)

    bias_results = run_sklearn_models(
        dataset_name="Article Political Bias Direction",
        X_train=X_train_bias,
        X_test=X_test_bias,
        y_train=y_train_bias,
        y_test=y_test_bias,
    )
    all_results.extend(bias_results)

    if RUN_TRANSFORMER:
        bias_transformer_result = run_transformer_model(
            dataset_name="Article Political Bias Direction",
            X_train=X_train_bias,
            X_test=X_test_bias,
            y_train=y_train_bias,
            y_test=y_test_bias,
            train_sample_size=BIAS_BERT_TRAIN_SAMPLE,
            test_sample_size=BIAS_BERT_TEST_SAMPLE,
        )
        all_results.append(bias_transformer_result)


def run_optional_party_proxy(all_results, liar_train, liar_test):
    if not RUN_LIAR_PARTY_PROXY:
        return

    print("\n\n")
    print("#" * 80)
    print("OPTIONAL PART 4: LIAR POLITICAL AFFILIATION PROXY")
    print("#" * 80)

    liar_train_party = liar_train.copy()
    liar_test_party = liar_test.copy()

    liar_train_party["party_simple"] = (
        liar_train_party["party_affiliation"]
        .apply(simplify_party)
    )
    liar_test_party["party_simple"] = (
        liar_test_party["party_affiliation"]
        .apply(simplify_party)
    )

    print("\nSimplified party labels in train:")
    print(
        liar_train_party["party_simple"]
        .value_counts()
    )

    print("\nSimplified party labels in test:")
    print(
        liar_test_party["party_simple"]
        .value_counts()
    )

    X_train_party = liar_train_party["statement"].astype(str)
    y_train_party = liar_train_party["party_simple"].astype(str)

    X_test_party = liar_test_party["statement"].astype(str)
    y_test_party = liar_test_party["party_simple"].astype(str)

    party_results = run_sklearn_models(
        dataset_name="LIAR Political Affiliation Proxy",
        X_train=X_train_party,
        X_test=X_test_party,
        y_train=y_train_party,
        y_test=y_test_party,
    )
    all_results.extend(party_results)


def main():
    ensure_result_folders()
    print_runtime_info()

    all_results = []

    run_isot(all_results)
    liar_train, liar_test = run_liar(all_results)
    run_bias(all_results)
    run_optional_party_proxy(
        all_results,
        liar_train,
        liar_test,
    )

    results_df = pd.DataFrame(all_results)

    print("\n\n")
    print("=" * 80)
    print("FINAL MODEL RESULTS SUMMARY")
    print("=" * 80)
    print(results_df)

    summary_path = save_final_outputs(results_df)

    print(f"\nSaved results summary to: {summary_path}")
    print("Saved charts, classification reports, and confusion matrices under results/.")


if __name__ == "__main__":
    main()
