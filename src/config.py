from pathlib import Path

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"

# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

RUN_TRANSFORMER = True
RUN_ARTICLE_BIAS_DATASET = True
RUN_LIAR_PARTY_PROXY = False  # Keep False for the main thesis experiment.

ISOT_BERT_TRAIN_SAMPLE = 20000
ISOT_BERT_TEST_SAMPLE = 5000

LIAR_BERT_TRAIN_SAMPLE = 10240
LIAR_BERT_TEST_SAMPLE = 1267

BIAS_BERT_TRAIN_SAMPLE = 10000
BIAS_BERT_TEST_SAMPLE = 2000
# Note: the current Baly test split contains 1,300 rows, so the code
# automatically uses the full test set because 1,300 < 2,000.

MAX_TRANSFORMER_LENGTH = 128
TRANSFORMER_MODEL = "distilbert-base-uncased"
RANDOM_STATE = 42

TFIDF_MAX_FEATURES = 10000
TFIDF_STOP_WORDS = "english"
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2

LR_MAX_ITER = 1000
MLP_HIDDEN_LAYER_SIZES = (128,)
MLP_MAX_ITER = 50
MLP_N_ITER_NO_CHANGE = 5

TRANSFORMER_EPOCHS = 2
TRANSFORMER_TRAIN_BATCH_SIZE = 8
TRANSFORMER_EVAL_BATCH_SIZE = 8
