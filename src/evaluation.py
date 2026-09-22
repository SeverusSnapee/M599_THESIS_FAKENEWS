from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from config import RESULTS_DIR
from io_utils import safe_filename


def ensure_result_folders():
    """Create the common result folders used by the project."""
    (RESULTS_DIR / "summary").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "charts" / "overall").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "huggingface_runs").mkdir(parents=True, exist_ok=True)


def model_result_dir(dataset_name, model_name):
    """Return a clean folder for one dataset/model result set."""
    path = (
        RESULTS_DIR
        / safe_filename(dataset_name)
        / safe_filename(model_name)
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def dataset_chart_dir(dataset_name):
    """Return the chart folder for one dataset."""
    path = RESULTS_DIR / safe_filename(dataset_name) / "charts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_classification_outputs(dataset_name, model_name, y_true, y_pred):
    """Save report, confusion matrix CSV, and confusion matrix image in organized folders."""
    out_dir = model_result_dir(dataset_name, model_name)

    labels = sorted(pd.Series(y_true).astype(str).unique())

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(out_dir / "classification_report.csv")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(out_dir / "confusion_matrix.csv")

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
    plt.savefig(out_dir / "confusion_matrix.png", dpi=300)
    plt.close()

    return report_df


def get_metrics(y_true, y_pred):
    """Return the same evaluation metrics used in the original thesis code."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Weighted_Precision": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "Weighted_Recall": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "Weighted_F1": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "Macro_F1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
    }


def create_metric_chart(results_df, metric, title, filename, output_dir):
    """Create a readable horizontal bar chart with metric values."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    plt.savefig(output_dir / filename, dpi=300)
    plt.close()


def create_dataset_specific_charts(results_df):
    """Create separate accuracy and weighted F1 charts for each dataset."""
    for dataset_name in results_df["Dataset"].unique():
        dataset_df = results_df[results_df["Dataset"] == dataset_name].copy()
        out_dir = dataset_chart_dir(dataset_name)

        create_metric_chart(
            dataset_df,
            "Accuracy",
            f"Accuracy Comparison - {dataset_name}",
            "accuracy_comparison.png",
            out_dir,
        )

        create_metric_chart(
            dataset_df,
            "Weighted_F1",
            f"Weighted F1 Comparison - {dataset_name}",
            "weighted_f1_comparison.png",
            out_dir,
        )


def save_final_outputs(results_df):
    """Save the master summary CSV and all comparison charts."""
    ensure_result_folders()

    summary_path = RESULTS_DIR / "summary" / "model_results_summary.csv"
    results_df.to_csv(summary_path, index=False)

    overall_dir = RESULTS_DIR / "charts" / "overall"

    create_metric_chart(
        results_df,
        "Accuracy",
        "Model Accuracy Comparison",
        "accuracy_comparison.png",
        overall_dir,
    )
    create_metric_chart(
        results_df,
        "Weighted_F1",
        "Model Weighted F1 Comparison",
        "f1_comparison.png",
        overall_dir,
    )
    create_metric_chart(
        results_df,
        "Macro_F1",
        "Model Macro F1 Comparison",
        "macro_f1_comparison.png",
        overall_dir,
    )

    create_dataset_specific_charts(results_df)

    return summary_path
