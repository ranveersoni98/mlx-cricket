"""Team strength ratings from doosra ball-by-ball innings data.

Per-player expected metrics (exp_runs/exp_outs/exp_wkts are context-adjusted)
aggregate to team level, then trail by date — strictly causal:

  bat_rating  = trailing mean of (runs - exp_runs) per match
  bowl_rating = trailing mean of (exp_runs - runs conceded + 5*(wkts - exp_wkts))

Positive = stronger than context expects. Diffed at match time.
Covers 2001+ (doosra range); older rows get 0 and rely on Elo.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

from ..config import RAW_DIR
from .normalize import normalize_team

REPO_ID = "Sarthak213/doosra-cricket"
BAT_FILE = "data/batting_innings.parquet"
BOWL_FILE = "data/bowling_innings.parquet"
WINDOW = 20
MIN_GAMES = 3


def download_innings(local_dir: Path | None = None) -> dict[str, Path]:
    local_dir = Path(local_dir or RAW_DIR)
    local_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for key, name in (("bat", BAT_FILE), ("bowl", BOWL_FILE)):
        out[key] = Path(hf_hub_download(
            repo_id=REPO_ID, filename=name, repo_type="dataset", local_dir=str(local_dir)))
    return out


def _team_match_table(df: pd.DataFrame, value: pd.Series, tag: str) -> pd.DataFrame:
    g = pd.DataFrame({"team": df["team"].astype(str).map(normalize_team),
                      "fgroup": df["fgroup"].fillna("").astype(str),
                      "date": pd.to_datetime(df["date"], errors="coerce"),
                      "v": value})
    g = g.dropna(subset=["date"]).sort_values("date")
    per_match = g.groupby(["team", "fgroup", "date"], as_index=False)["v"].sum()
    per_match = per_match.sort_values("date")
    per_match[f"trail_{tag}"] = (
        per_match.groupby(["team", "fgroup"])["v"]
        .transform(lambda s: s.shift(1).rolling(WINDOW, min_periods=MIN_GAMES).mean())
        .fillna(0.0)
    )
    return per_match[["team", "fgroup", "date", f"trail_{tag}"]]


def build_ratings(bat_path: Path | None = None, bowl_path: Path | None = None) -> pd.DataFrame:
    """Per-(team, format-group, date) trailing ratings: team, fgroup, date, trail_bat, trail_bowl."""
    paths = download_innings()
    bat = pd.read_parquet(bat_path or paths["bat"],
                          columns=["team", "fgroup", "date", "runs", "exp_runs"])
    bowl = pd.read_parquet(bowl_path or paths["bowl"],
                           columns=["team", "fgroup", "date", "runs", "wickets", "exp_runs", "exp_wkts"])
    bat_t = _team_match_table(bat, bat["runs"].fillna(0) - bat["exp_runs"].fillna(0), "bat")
    bowl_val = ((bowl["exp_runs"].fillna(0) - bowl["runs"].fillna(0))
                + 5.0 * (bowl["wickets"].fillna(0) - bowl["exp_wkts"].fillna(0)))
    bowl_t = _team_match_table(bowl, bowl_val, "bowl")
    merged = pd.merge(bat_t, bowl_t, on=["team", "fgroup", "date"], how="outer")
    merged[["trail_bat", "trail_bowl"]] = merged[["trail_bat", "trail_bowl"]].fillna(0.0)
    return merged.sort_values("date").reset_index(drop=True)


FMT_TO_FGROUP = {"ODI": "ODI", "T20": "T20", "TEST": "MULTI"}


def add_rating_features(df: pd.DataFrame, ratings: pd.DataFrame | None = None) -> pd.DataFrame:
    """Merge trailing bat/bowl ratings as of each match date (merge_asof per side)."""
    df = df.copy()
    if ratings is None:
        try:
            ratings = build_ratings()
        except Exception:
            df["bat_diff"] = 0.0
            df["bowl_diff"] = 0.0
            return df
    ratings = ratings.sort_values("date")
    out = df.copy()
    out["_dt"] = pd.to_datetime(out["date"], errors="coerce")
    out["_dt"] = out["_dt"].fillna(pd.to_datetime(out["year"].astype(str) + "-06-15", errors="coerce"))
    out["_fg"] = out["format"].map(FMT_TO_FGROUP).fillna(out["format"])
    for side in ("team1", "team2"):
        # merge_asof `by` keys must share names on both sides; align them first.
        r2 = ratings.rename(columns={"team": side, "fgroup": "_fg"})
        m = pd.merge_asof(out.sort_values("_dt"), r2.sort_values("date"),
                          left_on="_dt", right_on="date", by=[side, "_fg"], direction="backward")
        tag = "1" if side == "team1" else "2"
        out = out.assign(**{f"bat{tag}": m["trail_bat"].fillna(0.0).values,
                            f"bowl{tag}": m["trail_bowl"].fillna(0.0).values})
    out["bat_diff"] = out["bat1"] - out["bat2"]
    out["bowl_diff"] = out["bowl1"] - out["bowl2"]
    return out.drop(columns=["_dt", "_fg", "bat1", "bat2", "bowl1", "bowl2"])


def latest_ratings(ratings: pd.DataFrame) -> dict[tuple[str, str], tuple[float, float]]:
    """Final rating per (team, format-group) for live inference."""
    last = ratings.sort_values("date").groupby(["team", "fgroup"]).tail(1)
    return {(str(r["team"]), str(r["fgroup"])): (float(r["trail_bat"]), float(r["trail_bowl"]))
            for _, r in last.iterrows()}
