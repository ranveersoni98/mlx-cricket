"""Thin client for CricAPI v1 (https://api.cricapi.com)."""
from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import requests
from dotenv import load_dotenv

from ..config import CRICAPI_BASE

load_dotenv()

API_KEY = os.environ.get("CRICAPI_KEY", "").strip()


def _require_key() -> str:
    if not API_KEY:
        raise RuntimeError(
            "CRICAPI_KEY is not set. Copy .env.template to .env and add your key "
            "(free at https://www.cricapi.com/)."
        )
    return API_KEY


def _get(endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = dict(params or {})
    params.setdefault("apikey", _require_key())
    r = requests.get(f"{CRICAPI_BASE}/{endpoint}", params=params, timeout=30)
    r.raise_for_status()
    return cast(dict[str, Any], r.json())


def get_countries(offset: int = 0) -> dict[str, Any]:
    return _get("countries", {"offset": offset})


def get_matches(offset: int = 0) -> dict[str, Any]:
    """All matches (past + scheduled). 25 per page."""
    return _get("matches", {"offset": offset})


def get_current_matches(offset: int = 0) -> dict[str, Any]:
    """Live / recent matches with score blocks."""
    return _get("currentMatches", {"offset": offset})


def get_cric_score(offset: int = 0) -> dict[str, Any]:
    """Compact live scores: t1, t2, t1s, t2s, ms in {fixture,live,result}."""
    return _get("cricScore", {"offset": offset})


def get_series(offset: int = 0) -> dict[str, Any]:
    return _get("series", {"offset": offset})


def get_players(offset: int = 0) -> dict[str, Any]:
    return _get("players", {"offset": offset})


def normalize_fixture(m: dict[str, Any]) -> dict[str, Any]:
    """Normalize cricScore/currentMatches/matches rows into one schema."""
    if "t1" in m:  # cricScore shape
        team1 = str(m.get("t1", "")).split("[")[0].strip()
        team2 = str(m.get("t2", "")).split("[")[0].strip()
        return {
            "id": m.get("id"),
            "team1": team1,
            "team2": team2,
            "match_type": m.get("matchType"),
            "status": m.get("status"),
            "state": m.get("ms"),  # fixture | live | result
            "date": m.get("dateTimeGMT"),
            "series": m.get("series"),
            "team1_score": m.get("t1s"),
            "team2_score": m.get("t2s"),
            "venue": "",
        }
    teams_raw = m.get("teams", ["", ""])
    teams: list[Any] = list(teams_raw) if isinstance(teams_raw, list) else ["", ""]
    return {
        "id": m.get("id"),
        "team1": teams[0] if len(teams) > 0 else "",
        "team2": teams[1] if len(teams) > 1 else "",
        "match_type": m.get("matchType", ""),
        "status": m.get("status", ""),
        "state": (
            "result"
            if m.get("matchEnded")
            else ("live" if m.get("matchStarted") else "fixture")
        ),
        "date": m.get("dateTimeGMT") or m.get("date"),
        "series": "",
        "team1_score": "",
        "team2_score": "",
        "venue": m.get("venue", ""),
        "score": m.get("score", []),
    }


def fetch_upcoming(limit: int = 50) -> list[dict[str, Any]]:
    """Upcoming fixtures from cricScore + currentMatches, deduped."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for fn in (get_cric_score, get_current_matches, get_matches):
        try:
            data_raw = fn(offset=0).get("data", [])
        except Exception:
            continue
        data: list[Any] = data_raw if isinstance(data_raw, list) else []
        for m in data:
            if not isinstance(m, dict):
                continue
            n = normalize_fixture(m)
            nid = str(n.get("id", ""))
            if nid in seen:
                continue
            seen.add(nid)
            out.append(n)
            if len(out) >= limit:
                return out
    return out


def cache_fetch(path: str | Path, fetch_fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    p = Path(path)
    if p.exists():
        return cast(dict[str, Any], json.loads(p.read_text()))
    data = fetch_fn()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2))
    return data
