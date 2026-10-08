"""Inference: CricAPI client + predictor."""
from .cricapi import (
    fetch_upcoming,
    get_countries,
    get_cric_score,
    get_current_matches,
    get_matches,
    get_players,
    get_series,
    normalize_fixture,
)
from .predict import Predictor, infer_team_type, split_venue

__all__ = [
    "fetch_upcoming",
    "get_countries",
    "get_cric_score",
    "get_current_matches",
    "get_matches",
    "get_players",
    "get_series",
    "normalize_fixture",
    "Predictor",
    "infer_team_type",
    "split_venue",
]
