"""Central config for mlx-cric."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Hugging Face tabular match-result datasets (same schema family)
HF_DATASETS = {
    "odi": {
        "repo_id": "bhuvaneshprasad/odi-cricket-dataset-1971-2014",
        "file": "odi_Matches_Data.csv",
    },
    "t20i": {
        "repo_id": "bhuvaneshprasad/t20i-cricket-dataset-2005-2014",
        "file": "t20i_Matches_Data.csv",
    },
    "ipl": {
        "repo_id": "bhuvaneshprasad/ipl-dataset",
        "file": "ipl_historical.csv",
    },
}

# Modern ball-by-ball (large, opt-in for v2)
HF_MODERN = {
    "cricsheet": "jeyasuryaur/cricket-data-by-cricsheet",  # up to Apr 2026, 6.9k dl
    "doosra": "Sarthak213/doosra-cricket",  # 22,983 matches up to 2026-09-17
}

CRICAPI_BASE = "https://api.cricapi.com/v1"
# NOTE: move this to env var CRICAPI_KEY in production.
DEFAULT_CRICSCORE_ENDPOINTS = ["cricScore", "currentMatches", "matches"]

RANDOM_SEED = 42

# Walk-forward splits (year-based, no future leakage)
TRAIN_END_YEAR = 2021
VAL_END_YEAR = 2023
# test = year > VAL_END_YEAR (2024-2026)

# Pre-2000 ODI was a different sport (60-over games, no fielding
# restrictions) — it hurts modern ODI prediction, so it's excluded
# from winner training.
ODI_CUTOFF_YEAR = 2000

# Form windows
FORM_WINDOW = 5
H2H_WINDOW = 5
MIN_GAMES_BACKOFF = 5
