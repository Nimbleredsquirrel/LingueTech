import os

MODEL_NAME = "meta-llama/Llama-2-7b-hf"

# on Kaggle: set env vars via Kaggle Secrets or os.environ before importing config
# HF_TOKEN  -> your HuggingFace token (needed for Llama-2)
# HF_CACHE_DIR -> where to store the downloaded model weights
CACHE_DIR = os.getenv("HF_CACHE_DIR", "/kaggle/working/hf_cache")
HF_TOKEN = os.getenv("HF_TOKEN", "")

DATA_DIR = "data"
LAYERS_DIR = "layers"
PLOTS_DIR = "plots"

# PRM800K: git lfs install && git clone https://github.com/openai/prm800k.git
# then: cp prm800k/data/phase2/train.jsonl data/phase2_train.jsonl
PRM800K_PATH = os.path.join(DATA_DIR, "phase2_train.jsonl")
DATASET_PATH = os.path.join(DATA_DIR, "dataset.parquet")

NUM_LAYERS = 33   # embedding layer + 32 transformer layers in Llama-2-7b
HIDDEN_DIM = 4096
MAX_SAMPLES = 5000
BATCH_SIZE = 8

PROBE_N_EXPERIMENTS = 5
PROBE_TEST_SIZE = 0.2
PROBE_RANDOM_STATE = 42

# checkpoint for extract_hidden_states.py (save progress every N batches)
CHECKPOINT_EVERY = 50
HIDDEN_CHECKPOINT_PATH = os.path.join(DATA_DIR, "hidden_checkpoint.npy")
HIDDEN_PROGRESS_PATH = os.path.join(DATA_DIR, "hidden_progress.txt")

# INSIDE (eigenscore.py)
# generate N responses per sample, extract hidden states at INSIDE_LAYER_IDX
INSIDE_N_RESPONSES = 5
INSIDE_TEMPERATURE = 0.8
INSIDE_MAX_NEW_TOKENS = 150
INSIDE_LAYER_IDX = 16       # 0-indexed; layer 16 is mid-depth in Llama-2-7b
INSIDE_RESPONSES_PATH = os.path.join(DATA_DIR, "inside_responses.npy")
INSIDE_RESULTS_PATH = os.path.join(DATA_DIR, "eigenscore_results.csv")

# Concepts for mass-mean probing.
# Each key is a column in dataset.parquet; value is a human-readable label.
# All columns are created by prepare_dataset.py.
CONCEPTS = {
    "label": "Step correctness (PRM800K)",
    "has_equation": "Contains equation (= sign)",
    "is_long_step": "Step length above median",
    "has_conclusion_word": "Concluding language (therefore/thus/hence/so)",
    # epistemic/emotional markers — probes whether the model encodes these internally
    "has_certainty": "Certainty markers (clearly/obviously/must/exactly)",
    "has_hedging": "Uncertainty markers (maybe/might/approximately/roughly)",
    "has_negation": "Negation present (not/no/never/neither)",
    "has_error_word": "Error acknowledgment (mistake/wrong/incorrect/invalid)",
}
