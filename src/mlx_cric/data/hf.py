"""Download HF tabular datasets and unify into one match-results frame.

Sources (same schema family):
  - bhuvaneshprasad/odi-cricket-dataset-1971-2014  (odi_Matches_Data.csv, 4717 rows)
  - bhuvaneshprasad/t20i-cricket-dataset-2005-2014 (t20i_Matches_Data.csv, 2426 rows)
  - bhuvaneshprasad/ipl-dataset                    (ipl_historical.csv, 1026 rows)

Unified pre-match features only (no leakage):
  team1, team2, venue_stadium, venue_city, venue_country,
  toss_winner (-> team1/team2/none), toss_choice (bat/bowl/none),
  format (ODI/T20I/IPL-T20), year, label (0 = team1 won, 1 = team2 won)

Rows with draws / no-result / ties without a winner are dropped for v1
binary classification (Test draws handled by excluding empty winners).
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import hf_hub_download

from ..config import HF_DATASETS, PROCESSED_DIR, RAW_DIR


def download_raw(local_dir: Path | None = None) -> dict[str, Path]:
    local_dir = Path(local_dir or RAW_DIR)
    local_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for key, spec in HF_DATASETS.items():
        fp = hf_hub_download(
            repo_id=spec["repo_id"],
            filename=spec["file"],
            repo_type="dataset",
            local_dir=str(local_dir),
        )
        paths[key] = Path(fp)
    return paths


def _norm_team(s: str) -> str:
    from .normalize import normalize_team
    return normalize_team(s)


def _toss_side(row: Mapping[str, Any]) -> str:
    toss = _norm_team(row.get("toss_winner", ""))
    t1 = _norm_team(row.get("team1", ""))
    t2 = _norm_team(row.get("team2", ""))
    if toss == t1:
        return "team1"
    if toss == t2:
        return "team2"
    return "none"


def _load_one(path: Path, fmt: str, gender: str = "male", team_type: str = "international") -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    cols = {c.lower().strip(): c for c in df.columns}

    def pick(*names: str) -> str | None:
        for n in names:
            if n in cols:
                return cols[n]
        return None

    c_t1 = pick("team1 name", "team1_name")
    c_t2 = pick("team2 name", "team2_name")
    c_stad = pick("match venue (stadium)", "match_venue_stadium")
    c_city = pick("match venue (city)", "match_venue_city")
    c_country = pick("match venue (country)", "match_venue_country")
    c_toss_w = pick("toss winner", "toss_winner")
    c_toss_c = pick("toss winner choice", "toss_winner_choice")
    c_winner = pick("match winner", "match_winner")
    c_date = pick("match date", "match_date")

    out = pd.DataFrame(
        {
            "team1": df[c_t1].map(_norm_team) if c_t1 else "",
            "team2": df[c_t2].map(_norm_team) if c_t2 else "",
            "venue_stadium": df[c_stad].fillna("Unknown") if c_stad else "Unknown",
            "venue_city": df[c_city].fillna("Unknown") if c_city else "Unknown",
            "venue_country": df[c_country].fillna("Unknown") if c_country else "Unknown",
            "toss_winner_raw": df[c_toss_w].fillna("") if c_toss_w else "",
            "toss_choice": df[c_toss_c].fillna("none").str.lower() if c_toss_c else "none",
            "match_winner_raw": df[c_winner].fillna("") if c_winner else "",
            "date": df[c_date] if c_date else "",
            "format": fmt,
        }
    )
    out["toss_side"] = out.apply(
        lambda r: _toss_side(
            {"toss_winner": r["toss_winner_raw"], "team1": r["team1"], "team2": r["team2"]}
        ),
        axis=1,
    )
    out["toss_choice"] = out["toss_choice"].map(
        lambda x: "bat" if "bat" in str(x) else ("bowl" if "bowl" in str(x) or "field" in str(x) else "none")
    )
    # Label: 0 team1 won, 1 team2 won, else drop
    def label(r: pd.Series) -> int:
        w = _norm_team(r["match_winner_raw"])
        if w == r["team1"]:
            return 0
        if w == r["team2"]:
            return 1
        return -1

    out["label"] = out.apply(label, axis=1)
    dates = pd.to_datetime(out["date"], errors="coerce")
    out["date"] = dates.dt.strftime("%Y-%m-%d").fillna("")
    out["year"] = dates.dt.year.fillna(2000).astype(int)
    out["gender"] = gender
    out["team_type"] = team_type
    # venue country fallback via city when legacy data lacks it
    try:
        from .normalize import infer_country
        mask = out["venue_country"].astype(str).str.strip().isin(["", "Unknown", "nan", "None"])
        out.loc[mask, "venue_country"] = [
            infer_country(c, s) for c, s in zip(out.loc[mask, "venue_city"], out.loc[mask, "venue_stadium"])
        ]
    except Exception:
        pass
    out = out[(out["label"] >= 0) & (out["team1"] != "") & (out["team2"] != "")]
    out["method"] = ""
    out["result"] = ""
    out["tier"] = out["format"]
    return out[
        [
            "team1", "team2", "venue_stadium", "venue_city", "venue_country",
            "toss_side", "toss_choice", "format", "date", "year", "label",
            "gender", "team_type", "method", "result", "tier",
        ]
    ].reset_index(drop=True)


def dedup_unified(df: pd.DataFrame) -> pd.DataFrame:
    """Drop 2001-2014 overlap duplicates between legacy CSVs and doosra parquet.

    Key is team-order-insensitive: the same fixture may list teams swapped
    (which also flips the label), so sort the pair before keying.
    """
    df = df.copy()
    t1 = df["team1"].astype(str)
    t2 = df["team2"].astype(str)
    lo = pd.Series([min(a, b) for a, b in zip(t1, t2)], index=df.index)
    hi = pd.Series([max(a, b) for a, b in zip(t1, t2)], index=df.index)
    df["_dkey"] = (
        lo + "|" + hi + "|"
        + df["date"].astype(str) + "|" + df["venue_city"].astype(str)
        + "|" + df["format"].astype(str)
    )
    # keep doosra (has date+result detail) over legacy on collision: doosra rows
    # have non-empty match_id only if we kept it; fallback: keep last occurrence
    # after sorting legacy first. Legacy formats are ODI/T20; doosra same names,
    # so stable sort keeps earliest source first -> drop keep='last' prefers doosra.
    df["_src"] = (df.get("result", pd.Series([""] * len(df))).astype(str) != "").astype(int)
    df = df.sort_values(["_dkey", "_src"]).drop_duplicates("_dkey", keep="last")
    return df.drop(columns=["_dkey", "_src"]).reset_index(drop=True)


def build_unified(raw_paths: dict[str, Path] | None = None, include_modern: bool = True, dedup: bool = True, keep_draws: bool = False) -> pd.DataFrame:
    raw_paths = raw_paths or download_raw()
    frames = [
        _load_one(raw_paths["odi"], "ODI", "male", "international"),
        _load_one(raw_paths["t20i"], "T20", "male", "international"),
        _load_one(raw_paths["ipl"], "T20", "male", "club"),
    ]
    if include_modern:
        try:
            from .doosra import load_unified as load_doosra
            frames.append(load_doosra(keep_draws=keep_draws))
        except Exception as e:
            print(f"modern doosra skipped: {e}")
    df = pd.concat(frames, ignore_index=True)
    # canonical team casing (title) to merge "Mumbai Indians" variants minimally
    for c in ["team1", "team2"]:
        df[c] = df[c].str.strip()
    if dedup:
        # normalize key order-insensitively? No — team1/team2 order matters for label,
        # but same fixture may list teams swapped. Check both orders.
        before = len(df)
        df = dedup_unified(df)
        print(f"dedup: {before} -> {len(df)}")
    return df


def save_processed(df: pd.DataFrame, path: Path | None = None) -> Path:
    path = Path(path or (PROCESSED_DIR / "matches_unified.csv"))
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_or_build(force: bool = False, keep_draws: bool = False) -> pd.DataFrame:
    target = PROCESSED_DIR / ("matches_unified_draws.csv" if keep_draws else "matches_unified.csv")
    if target.exists() and not force:
        return pd.read_csv(target)
    df = build_unified(keep_draws=keep_draws)
    return df
