import os
import re
import zipfile
import inspect
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

# Transformer imports
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

import torch
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

RUN_TRANSFORMER = True
RUN_ARTICLE_BIAS_DATASET = True
RUN_LIAR_PARTY_PROXY = False   # Keep False for main thesis. Use only as appendix/proxy if needed.

ISOT_BERT_TRAIN_SAMPLE = 20000
ISOT_BERT_TEST_SAMPLE = 5000

LIAR_BERT_TRAIN_SAMPLE = 10240
LIAR_BERT_TEST_SAMPLE = 1267

BIAS_BERT_TRAIN_SAMPLE = 10000
BIAS_BERT_TEST_SAMPLE = 2000

MAX_TRANSFORMER_LENGTH = 128
TRANSFORMER_MODEL = "distilbert-base-uncased"
RANDOM_STATE = 42

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    print("No GPU detected. DistilBERT may run slowly on CPU.")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_filename(text):
    """Creates a safe filename from a dataset/model name."""
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def find_file(possible_names):
    """Finds a file from a list of possible filenames."""
    for name in possible_names:
        if os.path.exists(name):
            return name
    raise FileNotFoundError(f"None of these files were found: {possible_names}")


def read_csv_or_zip(possible_names):
    """Reads a normal CSV file or the first CSV inside a ZIP file."""
    path = find_file(possible_names)

    if path.lower().endswith(".zip"):
        with zipfile.ZipFile(path, "r") as z:
            csv_files = [f for f in z.namelist() if f.lower().endswith(".csv")]
            if not csv_files:
                raise FileNotFoundError(f"No CSV file found inside {path}")
            with z.open(csv_files[0]) as f:
                return pd.read_csv(f)

    return pd.read_csv(path)


def read_parquet_file(possible_names):
    """Reads a parquet file and gives a clear message if pyarrow is missing."""
    path = find_file(possible_names)
    try:
        return pd.read_parquet(path)
    except ImportError as e:
        raise ImportError(
            "Parquet support is missing. Install pyarrow first using: pip install pyarrow"
        ) from e


def print_eda(name, df, text_column=None, label_column=None):
    """Prints basic dataset exploration for thesis reporting."""
    print("\n" + "=" * 70)
    print(f"EDA FOR {name}")
    print("=" * 70)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isnull().sum())

    if label_column and label_column in df.columns:
        print(f"\nClass distribution for {label_column}:")
        print(df[label_column].value_counts())

    if text_column and text_column in df.columns:
        text_lengths = df[text_column].astype(str).apply(len)
        print(f"\nText length statistics for {text_column}:")
        print(text_lengths.describe())


def save_classification_outputs(dataset_name, model_name, y_true, y_pred):
    """Saves classification report and confusion matrix for thesis evidence."""
    output_prefix = safe_filename(f"{dataset_name}_{model_name}")

    labels = sorted(pd.Series(y_true).astype(str).unique())

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f"{output_prefix}_classification_report.csv")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(f"{output_prefix}_confusion_matrix.csv")

    plt.figure(figsize=(8, 6))
    plt.imshow(cm)
    plt.title(f"Confusion Matrix: {dataset_name} - {model_name}")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(np.arange(len(labels)), labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_confusion_matrix.png", dpi=300)
    plt.close()

    return report_df


def get_metrics(y_true, y_pred):
    """Returns several metrics, including macro F1 for multi-class tasks."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Weighted_Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Weighted_Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "Weighted_F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "Macro_F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def run_sklearn_models(dataset_name, X_train, X_test, y_train, y_test):
    """Runs Logistic Regression, SVM, and MLP using TF-IDF features."""
    results = []

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "SVM": LinearSVC(random_state=RANDOM_STATE),
        "MLP Neural Network": MLPClassifier(
            hidden_layer_sizes=(128,),
            max_iter=50,
           
            n_iter_no_change=5,
            random_state=RANDOM_STATE,
        ),
    }

    for model_name, model in models.items():
        print("\n" + "-" * 70)
        print(f"{dataset_name} - {model_name}")
        print("-" * 70)

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=10000,
                stop_words="english",
                ngram_range=(1, 2),
                min_df=2,
            )),
            ("clf", model),
        ])

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        metrics = get_metrics(y_test, predictions)

        print("Accuracy:", metrics["Accuracy"])
        print("Weighted F1:", metrics["Weighted_F1"])
        print("Macro F1:", metrics["Macro_F1"])
        print("\nClassification Report:")
        print(classification_report(y_test, predictions, zero_division=0))

        save_classification_outputs(dataset_name, model_name, y_test, predictions)

        result_row = {
            "Dataset": dataset_name,
            "Model": model_name,
        }
        result_row.update(metrics)
        results.append(result_row)

    return results


def make_training_args(output_dir):
    """Handles different transformers versions."""
    signature = inspect.signature(TrainingArguments.__init__)
    params = signature.parameters

    kwargs = {
        "output_dir": output_dir,
        "save_strategy": "no",
        "num_train_epochs": 2,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "logging_steps": 50,
        "report_to": [],
    }

    if "eval_strategy" in params:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"

    if "fp16" in params:
        kwargs["fp16"] = torch.cuda.is_available()

    return TrainingArguments(**kwargs)


def run_transformer_model(dataset_name, X_train, X_test, y_train, y_test, train_sample_size, test_sample_size):
    """Runs DistilBERT transformer classification."""
    print("\n" + "=" * 70)
    print(f"{dataset_name} - TRANSFORMER MODEL: {TRANSFORMER_MODEL}")
    print("=" * 70)

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    train_df = pd.DataFrame({
        "text": X_train.astype(str).values,
        "label": y_train_encoded,
    })
    test_df = pd.DataFrame({
        "text": X_test.astype(str).values,
        "label": y_test_encoded,
    })

    if len(train_df) > train_sample_size:
        train_df = train_df.sample(train_sample_size, random_state=RANDOM_STATE)

    if len(test_df) > test_sample_size:
        test_df = test_df.sample(test_sample_size, random_state=RANDOM_STATE)

    train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
    test_dataset = Dataset.from_pandas(test_df, preserve_index=False)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_TRANSFORMER_LENGTH,
        )

    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    print("Loading transformer model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        TRANSFORMER_MODEL,
        num_labels=len(label_encoder.classes_),
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return get_metrics(labels, preds)

    output_dir = f"./{safe_filename(dataset_name)}_transformer_results"
    training_args = make_training_args(output_dir)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    print("Starting DistilBERT training...")
    trainer.train()

    print("Evaluating DistilBERT model...")
    evaluation = trainer.evaluate()
    print("\nTransformer evaluation results:")
    print(evaluation)

    prediction_output = trainer.predict(test_dataset)
    transformer_predictions = np.argmax(prediction_output.predictions, axis=-1)
    true_labels = prediction_output.label_ids

    true_label_names = label_encoder.inverse_transform(true_labels)
    predicted_label_names = label_encoder.inverse_transform(transformer_predictions)

    print("\nTransformer Classification Report:")
    print(classification_report(true_label_names, predicted_label_names, zero_division=0))

    report_df = save_classification_outputs(
        dataset_name,
        "DistilBERT Transformer",
        true_label_names,
        predicted_label_names,
    )

    metrics = get_metrics(true_label_names, predicted_label_names)
    result_row = {
        "Dataset": dataset_name,
        "Model": "DistilBERT Transformer",
    }
    result_row.update(metrics)
    return result_row


def clean_bias_labels(label_series):
    """Converts bias labels into left/center/right text labels if needed."""
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
    """Prepares the article political bias dataset for classification."""
    prepared = df.copy()

    if "bias_text" in prepared.columns:
        label_column = "bias_text"
    elif "bias" in prepared.columns:
        label_column = "bias"
    else:
        raise KeyError("No bias label column found. Expected 'bias_text' or 'bias'.")

    prepared["bias_label"] = clean_bias_labels(prepared[label_column])

    if "content" in prepared.columns:
        main_text = prepared["content"].astype(str)
    elif "content_original" in prepared.columns:
        main_text = prepared["content_original"].astype(str)
    else:
        raise KeyError("No article text column found. Expected 'content' or 'content_original'.")

    if "title" in prepared.columns:
        prepared["combined_text"] = prepared["title"].fillna("").astype(str) + " " + main_text
    else:
        prepared["combined_text"] = main_text

    prepared["combined_text"] = prepared["combined_text"].astype(str).str.strip()
    prepared = prepared.dropna(subset=["bias_label", "combined_text"])
    prepared = prepared[prepared["combined_text"].str.len() > 20]

    return prepared


def create_metric_chart(results_df, metric, title, filename):
    """Creates a readable horizontal bar chart with metric values."""
    plot_df = results_df.copy()
    plot_df["Label"] = plot_df["Dataset"] + " - " + plot_df["Model"]
    plot_df = plot_df.sort_values(["Dataset", metric], ascending=[True, True])

    y_positions = np.arange(len(plot_df))

    plt.figure(figsize=(12, max(6, len(plot_df) * 0.45)))
    plt.barh(y_positions, plot_df[metric])
    plt.yticks(y_positions, plot_df["Label"])
    plt.xlabel(metric.replace("_", " "))
    plt.title(title)
    plt.xlim(0, 1)

    for i, value in enumerate(plot_df[metric]):
        plt.text(min(value + 0.01, 0.98), i, f"{value:.3f}", va="center")

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()


def create_dataset_specific_charts(results_df):
    """Creates separate accuracy and weighted F1 charts for each dataset."""
    for dataset_name in results_df["Dataset"].unique():
        dataset_df = results_df[results_df["Dataset"] == dataset_name].copy()
        dataset_safe = safe_filename(dataset_name)

        create_metric_chart(
            dataset_df,
            "Accuracy",
            f"Accuracy Comparison - {dataset_name}",
            f"{dataset_safe}_accuracy_comparison.png",
        )

        create_metric_chart(
            dataset_df,
            "Weighted_F1",
            f"Weighted F1 Comparison - {dataset_name}",
            f"{dataset_safe}_weighted_f1_comparison.png",
        )


# ============================================================
# PART 1: ISOT FAKE NEWS DATASET
# ============================================================

print("\n\n")
print("#" * 80)
print("PART 1: ISOT FAKE NEWS DATASET")
print("#" * 80)

fake_df = read_csv_or_zip(["Fake.csv", "fake.csv", "Fake.csv.zip", "fake.csv.zip"])
true_df = read_csv_or_zip(["True.csv", "true.csv", "True.csv.zip", "true.csv.zip", "True.csv(2).zip"])

fake_df["label"] = "fake"
true_df["label"] = "real"

isot_df = pd.concat([fake_df, true_df], ignore_index=True)
isot_df["combined_text"] = isot_df["title"].astype(str) + " " + isot_df["text"].astype(str)

print_eda(
    name="ISOT",
    df=isot_df,
    text_column="combined_text",
    label_column="label",
)

X_isot = isot_df["combined_text"]
y_isot = isot_df["label"]

X_train_isot, X_test_isot, y_train_isot, y_test_isot = train_test_split(
    X_isot,
    y_isot,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y_isot,
)

all_results = []

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


# ============================================================
# PART 2: LIAR POLITICAL TRUTHFULNESS DATASET
# ============================================================

print("\n\n")
print("#" * 80)
print("PART 2: LIAR DATASET")
print("#" * 80)

liar_columns = [
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

liar_train = pd.read_csv("train.tsv", sep="\t", header=None, names=liar_columns)
liar_valid = pd.read_csv("valid.tsv", sep="\t", header=None, names=liar_columns)
liar_test = pd.read_csv("test.tsv", sep="\t", header=None, names=liar_columns)
liar_full = pd.concat([liar_train, liar_valid, liar_test], ignore_index=True)

print_eda(
    name="LIAR",
    df=liar_full,
    text_column="statement",
    label_column="label",
)

print("\nTop party affiliations:")
print(liar_full["party_affiliation"].value_counts().head(20))

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


# ============================================================
# PART 3: ARTICLE-LEVEL POLITICAL BIAS DATASET
# ============================================================

if RUN_ARTICLE_BIAS_DATASET:
    print("\n\n")
    print("#" * 80)
    print("PART 3: ARTICLE-LEVEL POLITICAL BIAS DATASET")
    print("#" * 80)

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

    bias_train = prepare_article_bias_dataframe(bias_train_raw)
    bias_valid = prepare_article_bias_dataframe(bias_valid_raw)
    bias_test = prepare_article_bias_dataframe(bias_test_raw)
    bias_full = pd.concat([bias_train, bias_valid, bias_test], ignore_index=True)

    print_eda(
        name="Article Political Bias",
        df=bias_full,
        text_column="combined_text",
        label_column="bias_label",
    )

    print("\nArticle bias dataset sources:")
    if "source" in bias_full.columns:
        print(bias_full["source"].value_counts().head(20))

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


# ============================================================
# OPTIONAL PART 4: LIAR POLITICAL AFFILIATION PROXY
# ============================================================

if RUN_LIAR_PARTY_PROXY:
    print("\n\n")
    print("#" * 80)
    print("OPTIONAL PART 4: LIAR POLITICAL AFFILIATION PROXY")
    print("#" * 80)

    def simplify_party(party):
        party = str(party).lower().strip()
        if "democrat" in party:
            return "democratic"
        elif "republican" in party:
            return "republican"
        else:
            return "other_or_neutral"

    liar_train_party = liar_train.copy()
    liar_test_party = liar_test.copy()

    liar_train_party["party_simple"] = liar_train_party["party_affiliation"].apply(simplify_party)
    liar_test_party["party_simple"] = liar_test_party["party_affiliation"].apply(simplify_party)

    print("\nSimplified party labels in train:")
    print(liar_train_party["party_simple"].value_counts())

    print("\nSimplified party labels in test:")
    print(liar_test_party["party_simple"].value_counts())

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


# ============================================================
# FINAL RESULTS TABLE AND CHARTS
# ============================================================

results_df = pd.DataFrame(all_results)

print("\n\n")
print("=" * 80)
print("FINAL MODEL RESULTS SUMMARY")
print("=" * 80)
print(results_df)

results_df.to_csv("model_results_summary.csv", index=False)
print("\nSaved results to: model_results_summary.csv")

create_metric_chart(
    results_df,
    "Accuracy",
    "Model Accuracy Comparison",
    "accuracy_comparison.png",
)

create_metric_chart(
    results_df,
    "Weighted_F1",
    "Model Weighted F1 Comparison",
    "f1_comparison.png",
)

create_metric_chart(
    results_df,
    "Macro_F1",
    "Model Macro F1 Comparison",
    "macro_f1_comparison.png",
)

create_dataset_specific_charts(results_df)

print("\nSaved charts and classification report files.")