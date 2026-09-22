import inspect

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from config import (
    LR_MAX_ITER,
    MAX_TRANSFORMER_LENGTH,
    MLP_HIDDEN_LAYER_SIZES,
    MLP_MAX_ITER,
    MLP_N_ITER_NO_CHANGE,
    RANDOM_STATE,
    RESULTS_DIR,
    TFIDF_MAX_FEATURES,
    TFIDF_MIN_DF,
    TFIDF_NGRAM_RANGE,
    TFIDF_STOP_WORDS,
    TRANSFORMER_EPOCHS,
    TRANSFORMER_EVAL_BATCH_SIZE,
    TRANSFORMER_MODEL,
    TRANSFORMER_TRAIN_BATCH_SIZE,
)
from evaluation import get_metrics, save_classification_outputs
from io_utils import safe_filename


def run_sklearn_models(dataset_name, X_train, X_test, y_train, y_test):
    """Run Logistic Regression, Linear SVM, and MLP using the original TF-IDF settings."""
    results = []

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=LR_MAX_ITER,
            random_state=RANDOM_STATE,
        ),
        "SVM": LinearSVC(random_state=RANDOM_STATE),
        "MLP Neural Network": MLPClassifier(
            hidden_layer_sizes=MLP_HIDDEN_LAYER_SIZES,
            max_iter=MLP_MAX_ITER,
            n_iter_no_change=MLP_N_ITER_NO_CHANGE,
            random_state=RANDOM_STATE,
        ),
    }

    for model_name, model in models.items():
        print("\n" + "-" * 70)
        print(f"{dataset_name} - {model_name}")
        print("-" * 70)

        pipeline = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=TFIDF_MAX_FEATURES,
                    stop_words=TFIDF_STOP_WORDS,
                    ngram_range=TFIDF_NGRAM_RANGE,
                    min_df=TFIDF_MIN_DF,
                ),
            ),
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

        save_classification_outputs(
            dataset_name,
            model_name,
            y_test,
            predictions,
        )

        result_row = {
            "Dataset": dataset_name,
            "Model": model_name,
        }
        result_row.update(metrics)
        results.append(result_row)

    return results


def make_training_args(output_dir):
    """Build TrainingArguments while remaining compatible with multiple transformers versions."""
    signature = inspect.signature(TrainingArguments.__init__)
    params = signature.parameters

    kwargs = {
        "output_dir": str(output_dir),
        "save_strategy": "no",
        "num_train_epochs": TRANSFORMER_EPOCHS,
        "per_device_train_batch_size": TRANSFORMER_TRAIN_BATCH_SIZE,
        "per_device_eval_batch_size": TRANSFORMER_EVAL_BATCH_SIZE,
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


def run_transformer_model(
    dataset_name,
    X_train,
    X_test,
    y_train,
    y_test,
    train_sample_size,
    test_sample_size,
):
    """Run DistilBERT using the same sampling and training logic as the original code."""
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
        train_df = train_df.sample(
            train_sample_size,
            random_state=RANDOM_STATE,
        )

    if len(test_df) > test_sample_size:
        test_df = test_df.sample(
            test_sample_size,
            random_state=RANDOM_STATE,
        )

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

    output_dir = (
        RESULTS_DIR
        / "huggingface_runs"
        / safe_filename(dataset_name)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
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
    transformer_predictions = np.argmax(
        prediction_output.predictions,
        axis=-1,
    )
    true_labels = prediction_output.label_ids

    true_label_names = label_encoder.inverse_transform(true_labels)
    predicted_label_names = label_encoder.inverse_transform(
        transformer_predictions
    )

    print("\nTransformer Classification Report:")
    print(
        classification_report(
            true_label_names,
            predicted_label_names,
            zero_division=0,
        )
    )

    save_classification_outputs(
        dataset_name,
        "DistilBERT Transformer",
        true_label_names,
        predicted_label_names,
    )

    metrics = get_metrics(
        true_label_names,
        predicted_label_names,
    )
    result_row = {
        "Dataset": dataset_name,
        "Model": "DistilBERT Transformer",
    }
    result_row.update(metrics)
    return result_row
