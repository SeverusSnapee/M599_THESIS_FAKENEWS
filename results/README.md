# Results folder

This folder is populated automatically when `src/main.py` is run.

The output is organized by dataset and model so the dissertation evidence is easy to inspect.

Example:

```text
results/
├── summary/
│   └── model_results_summary.csv
├── charts/
│   └── overall/
│       ├── accuracy_comparison.png
│       ├── f1_comparison.png
│       └── macro_f1_comparison.png
├── isot_fake_vs_real/
│   ├── logistic_regression/
│   │   ├── classification_report.csv
│   │   ├── confusion_matrix.csv
│   │   └── confusion_matrix.png
│   ├── svm/
│   ├── mlp_neural_network/
│   ├── distilbert_transformer/
│   └── charts/
├── liar_truthfulness/
│   └── ...
├── article_political_bias_direction/
│   └── ...
└── huggingface_runs/
```
