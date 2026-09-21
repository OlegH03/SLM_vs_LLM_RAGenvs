# Paths
DATA_PATH = "data/en.json"
DATA_INT_PATH = "data/en_int.json"
DATA_FACT_PATH = "data/en_fact.json"
RESULTS_PATH = "results/"
MODEL_PATH = ""

# Hyperparameters
MAX_NEW_TOKENS = 300
TEMPERATURE = 0

# Models for Phase 1
MODELS = {
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "gpt-4o-mini": "openai/gpt-4o-mini",
}

# RGB Dataset
TOP_K = 5
NOISE_RATIO = 0.6
SEED = 2026  # arbitrary number, just important for mixing consistency
