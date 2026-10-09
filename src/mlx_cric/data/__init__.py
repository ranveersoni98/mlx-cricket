"""Match data: HF downloads, normalization, unification."""
from .doosra import FORMAT_MAP, download_matches, load_unified as load_doosra
from .hf import build_unified, dedup_unified, download_raw, load_or_build, save_processed
from .normalize import CITY_TO_COUNTRY, TEAM_ALIASES, infer_country, normalize_team
from .pool import add_pool_features, build_pool_cache, latest_pool
from .ratings import add_rating_features, build_ratings, download_innings, latest_ratings

__all__ = [
    "FORMAT_MAP", "download_matches", "load_doosra",
    "build_unified", "dedup_unified", "download_raw", "load_or_build", "save_processed",
    "CITY_TO_COUNTRY", "TEAM_ALIASES", "infer_country", "normalize_team",
    "add_rating_features", "build_ratings", "download_innings", "latest_ratings",
    "add_pool_features", "build_pool_cache", "latest_pool",
]
