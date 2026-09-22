import os
import re
import zipfile
from pathlib import Path

import pandas as pd

from config import DATA_DIR, PROJECT_ROOT


def safe_filename(text):
    """Create a filesystem-safe name from a dataset/model label."""
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _candidate_locations(name):
    """Search the structured data folder first, then project root and current directory."""
    return [
        DATA_DIR / name,
        PROJECT_ROOT / name,
        Path.cwd() / name,
    ]


def find_file(possible_names):
    """Find a file from a list of possible filenames."""
    for name in possible_names:
        for path in _candidate_locations(name):
            if path.exists():
                return path
    raise FileNotFoundError(
        "None of these files were found: "
        f"{possible_names}. Place dataset files in {DATA_DIR}"
    )


def read_csv_or_zip(possible_names):
    """Read a CSV file or the first CSV inside a ZIP file."""
    path = find_file(possible_names)

    if str(path).lower().endswith(".zip"):
        with zipfile.ZipFile(path, "r") as z:
            csv_files = [f for f in z.namelist() if f.lower().endswith(".csv")]
            if not csv_files:
                raise FileNotFoundError(f"No CSV file found inside {path}")
            with z.open(csv_files[0]) as f:
                return pd.read_csv(f)

    return pd.read_csv(path)


def read_parquet_file(possible_names):
    """Read a parquet file and provide a clear error if pyarrow is missing."""
    path = find_file(possible_names)
    try:
        return pd.read_parquet(path)
    except ImportError as exc:
        raise ImportError(
            "Parquet support is missing. Install pyarrow first using: pip install pyarrow"
        ) from exc


def read_tsv(filename, *, names):
    """Read one LIAR TSV file from the structured data folder or fallback locations."""
    path = find_file([filename])
    return pd.read_csv(path, sep="\t", header=None, names=names)


def print_eda(name, df, text_column=None, label_column=None):
    """Print basic dataset exploration for thesis reporting."""
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
