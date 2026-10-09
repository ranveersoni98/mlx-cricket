"""Modern data: Sarthak213/doosra-cricket matches.parquet (22,983 matches).

Schema: match_id, match_type (Test/ODI/T20/MDM/ODM/IT20), gender, team_type,
date, venue, city, team1, team2, toss_winner, toss_decision, winner, result.
Range: 2001-12-19 to 2026-09-17. Decisive (result IS NULL): ~21k rows.
Draws/no-result/tie dropped for binary v1/v2 (3-class roadmap).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = "Sarthak213/doosra-cricket"
FILENAME = "data/matches.parquet"

FORMAT_MAP = {
    "ODI": "ODI", "ODM": "ODI",
    "T20": "T20", "IT20": "T20",
    "Test": "TEST", "MDM": "TEST",
}


def download_matches(local_dir: Path | None = None) -> Path:
    from ..config import RAW_DIR
    local_dir = Path(local_dir or RAW_DIR)
    local_dir.mkdir(parents=True, exist_ok=True)
    fp = hf_hub_download(
        repo_id=REPO_ID, filename=FILENAME, repo_type="dataset",
        local_dir=str(local_dir),
    )
    return Path(fp)


def load_unified(path: Path | None = None, keep_draws: bool = False) -> pd.DataFrame:
    from .normalize import normalize_team, infer_country

    path = Path(path) if path else download_matches()
    df = pd.read_parquet(path)
    df["format"] = df["match_type"].map(FORMAT_MAP).fillna(df["match_type"])
    df["toss_choice"] = df["toss_decision"].fillna("none").str.lower().map(
        lambda x: "bat" if "bat" in str(x) else ("bowl" if ("bowl" in str(x) or "field" in str(x)) else "none")
    )
    df["toss_side"] = df.apply(
        lambda r: "team1" if r.get("toss_winner") == r.get("team1")
        else ("team2" if r.get("toss_winner") == r.get("team2") else "none"),
        axis=1,
    )

    def label(r: pd.Series) -> int:
        w = str(r.get("winner") or "").strip()
        t1, t2 = str(r.get("team1")).strip(), str(r.get("team2")).strip()
        if w and w == t1:
            return 0
        if w and w == t2:
            return 1
        res = str(r.get("result") or "").strip().lower()
        if keep_draws and res == "draw":
            return 2
        return -1

    df["team1"] = df["team1"].astype(str).map(normalize_team)
    df["team2"] = df["team2"].astype(str).map(normalize_team)
    df["winner"] = df["winner"].fillna("").astype(str).map(lambda w: normalize_team(str(w)) if w else "")
    df["label"] = df.apply(label, axis=1)
    dates = pd.to_datetime(df["date"], errors="coerce")
    df["date"] = dates.dt.strftime("%Y-%m-%d").fillna("")
    df["year"] = dates.dt.year.fillna(2010).astype(int)
    df["venue_stadium"] = df["venue"].fillna("Unknown")
    df["venue_city"] = df["city"].fillna("Unknown")
    df["venue_country"] = [
        infer_country(c, s) for c, s in zip(df["venue_city"], df["venue_stadium"])
    ]
    df["gender"] = df["gender"].fillna("male")
    df["team_type"] = df["team_type"].fillna("international")
    # raw tier kept: Test draws far less often than domestic MDM (19% vs 38%)
    df["tier"] = df["match_type"].fillna(df["format"])
    df["result"] = df["result"].fillna("") if "result" in df else ""
    out = df[(df["label"] >= 0)].copy()
    cols = ["team1", "team2", "venue_stadium", "venue_city", "venue_country",
         "toss_side", "toss_choice", "format", "date", "year", "label",
         "gender", "team_type", "tier"]
    if "result" in df:
        cols.append("result")
    for extra in ("win_by_runs", "win_by_wickets", "match_id", "player_of_match", "method"):
        if extra in df:
            cols.append(extra)
    out = out.copy()
    if "method" not in out.columns:
        out["method"] = ""
    else:
        out["method"] = out["method"].fillna("")
    return out[cols + (["method"] if "method" not in cols else [])].reset_index(drop=True)
