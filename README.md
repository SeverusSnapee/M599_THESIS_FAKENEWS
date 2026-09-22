# Fake News, Political Truthfulness, and Political Ideology Classification

This repository contains the implementation used for a comparative dissertation study of traditional machine learning, a neural network, and a transformer-based NLP model across three related text-classification tasks.

## Models

- Logistic Regression
- Linear Support Vector Machine
- Multi-Layer Perceptron
- DistilBERT (`distilbert-base-uncased`)

## Datasets / Tasks

- ISOT Fake News Dataset — fake vs. real news classification
- LIAR — six-class political truthfulness classification
- Baly et al. (2020) Article Political Ideology Dataset — left / center / right article ideology classification

## Project structure

```text
.
├── README.md
├── requirements.txt
├── data/
│   ├── README.md
│   └── raw/
├── legacy/
│   └── original_single_file.py
├── results/
│   └── README.md
└── src/
    ├── config.py
    ├── datasets.py
    ├── evaluation.py
    ├── io_utils.py
    ├── main.py
    └── models.py
```

## Important reproducibility note

The refactor intentionally preserves the experiment logic and model settings from the original single-file implementation. The main change is code organization and the location of output files.

For the Baly DistilBERT experiment, the configuration still allows a test sample size of 2,000. The predefined Baly test split used in the study contains 1,300 rows, so the program uses the full 1,300-row test set automatically.

## Installation

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

## Data

Place the dataset files in:

```text
data/raw/
```

See `data/README.md` for the expected filenames.

## Run

From the repository root:

```bash
python src/main.py
```

## Output organization

All experiment outputs are stored under `results/`.

Each dataset/model combination gets its own folder containing:

- `classification_report.csv`
- `confusion_matrix.csv`
- `confusion_matrix.png`

The overall summary is saved to:

```text
results/summary/model_results_summary.csv
```

Comparison charts are saved under:

```text
results/charts/overall/
```

Dataset-specific comparison charts are stored inside each dataset folder.

## Main experimental settings

- TF-IDF maximum features: 10,000
- TF-IDF n-grams: unigrams and bigrams
- TF-IDF minimum document frequency: 2
- Logistic Regression maximum iterations: 1,000
- MLP hidden layer: 128 units
- MLP maximum iterations: 50
- DistilBERT maximum sequence length: 128 tokens
- DistilBERT epochs: 2
- DistilBERT batch size: 8
- Random state: 42

