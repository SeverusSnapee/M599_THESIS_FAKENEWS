# Dataset files

Place the dataset files used by the dissertation in `data/raw/`.

Expected files:

## ISOT Fake News Dataset
- `Fake.csv`
- `True.csv`

ZIP versions are also supported by the code.

## LIAR Dataset
- `train.tsv`
- `valid.tsv`
- `test.tsv`

## Baly et al. (2020) Article Political Ideology Dataset
The loader supports these common names:

Training:
- `train-00000-of-00001.parquet`
- `bias_train.parquet`
- `train.parquet`

Validation:
- `valid-00000-of-00001.parquet`
- `validation-00000-of-00001.parquet`
- `bias_valid.parquet`
- `bias_validation.parquet`
- `valid.parquet`
- `validation.parquet`

Test:
- `test-00000-of-00001.parquet`
- `bias_test.parquet`
- `test.parquet`

The dataset files are intentionally not included in this repository. Obtain them from their original sources and follow their licensing terms.
