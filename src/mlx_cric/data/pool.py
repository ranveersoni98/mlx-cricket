"""Player-pool strength: who a team can field, and how good they are.

Per-player trailing ratings from ball-by-ball innings (expected-adjusted),
snapshotted monthly. Team strength at match time = mean of its top-11
*active* players (played within the last 365 days) from the *prior* month's
snapshot — no within-month leakage, no XI needed at predict time.

When CricAPI squads are available the same ratings can score an exact XI;
the pool mean is the no-squad fallback and works for every historical row.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from ..config import RAW_DIR
from .normalize import normalize_team
from .ratings import BAT_FILE, BOWL_FILE, REPO_ID

POOL_WINDOW = 15
POOL_MIN = 5
ACTIVE_DAYS = 365
POOL_TOP_N = 11

SNAP_FILE = "player_pool.parquet"


def _monthly_pool(
    df: pd.DataFrame, value: pd.Series, tag: str,
) -> pd.DataFrame:
    g = pd.DataFrame({
        "player": df["player"].astype(str),
        "team": df["team"].astype(str).map(normalize_team),
        "fgroup": df["fgroup"].fillna("").astype(str),
        "date": pd.to_datetime(df["date"], errors="coerce"),
        "v": value,
    }).dropna(subset=["date"]).sort_values("date")
    g["trail"] = (
        g.groupby(["player", "fgroup"])["v"]
        .transform(lambda s: s.shift(1).rolling(POOL_WINDOW, min_periods=POOL_MIN).mean())
    )
    months = pd.period_range(g["date"].min().to_period("M"), g["date"].max().to_period("M"), freq="M")
    out: list[pd.DataFrame] = []
    for m in months:
        end = m.end_time
        start = end - pd.Timedelta(days=ACTIVE_DAYS)
        w = g[(g["date"] > start) & (g["date"] <= end)]
        if not len(w):
            continue
        last = w.sort_values("date").groupby(["player", "team", "fgroup"]).tail(1)
        last = last.dropna(subset=["trail"])
        if not len(last):
            continue
        top = (last.sort_values("trail", ascending=False)
               .groupby(["team", "fgroup"]).head(POOL_TOP_N)
               .groupby(["team", "fgroup"], as_index=False)["trail"].mean())
        top["snap"] = end.strftime("%Y-%m")
        top = top.rename(columns={"trail": f"pool_{tag}"})
        out.append(top)
    if not out:
        return pd.DataFrame(columns=["team", "fgroup", "snap", f"pool_{tag}"])
    return pd.concat(out, ignore_index=True)


def build_pool_cache(local_dir: Path | None = None) -> Path:
    """Monthly (team, fgroup) pool ratings. Cached to data/raw/player_pool.parquet."""
    local_dir = Path(local_dir or RAW_DIR)
    target = local_dir / SNAP_FILE
    if target.exists():
        return target
    bat_p = hf_hub_download(repo_id=REPO_ID, filename=BAT_FILE, repo_type="dataset", local_dir=str(local_dir))
    bowl_p = hf_hub_download(repo_id=REPO_ID, filename=BOWL_FILE, repo_type="dataset", local_dir=str(local_dir))
    bat = pd.read_parquet(bat_p, columns=["player", "team", "fgroup", "date", "runs", "exp_runs"])
    bowl = pd.read_parquet(bowl_p, columns=["player", "team", "fgroup", "date", "runs", "wickets", "exp_runs", "exp_wkts"])
    bat_pool = _monthly_pool(bat, bat["runs"].fillna(0) - bat["exp_runs"].fillna(0), "bat")
    bowl_val = ((bowl["exp_runs"].fillna(0) - bowl["runs"].fillna(0))
                + 5.0 * (bowl["wickets"].fillna(0) - bowl["exp_wkts"].fillna(0)))
    bowl_pool = _monthly_pool(bowl, bowl_val, "bowl")
    merged = pd.merge(bat_pool, bowl_pool, on=["team", "fgroup", "snap"], how="outer")
    merged[["pool_bat", "pool_bowl"]] = merged[["pool_bat", "pool_bowl"]].fillna(0.0)
    merged.to_parquet(target, index=False)
    return target


def add_pool_features(df: pd.DataFrame, pool: pd.DataFrame | None = None) -> pd.DataFrame:
    """Merge prior-month pool ratings per side. Missing -> 0."""
    df = df.copy()
    if pool is None:
        try:
            pool = pd.read_parquet(build_pool_cache())
        except Exception:
            df["pool_bat_diff"] = 0.0
            df["pool_bowl_diff"] = 0.0
            return df
    out = df.copy()
    out["_snap"] = (pd.to_datetime(out["date"], errors="coerce") - pd.offsets.MonthBegin(1)).dt.strftime("%Y-%m")
    out["_fg"] = out["format"].map({"ODI": "ODI", "T20": "T20", "TEST": "MULTI"}).fillna(out["format"])
    for side, pfx in (("team1", "p1"), ("team2", "p2")):
        r = pool.rename(columns={"team": side})
        m = pd.merge(out[["_snap", "_fg", side]], r, left_on=["_snap", "_fg", side],
                     right_on=["snap", "fgroup", side], how="left")
        out[f"{pfx}_bat"] = m["pool_bat"].fillna(0.0).values
        out[f"{pfx}_bowl"] = m["pool_bowl"].fillna(0.0).values
    out["pool_bat_diff"] = out["p1_bat"] - out["p2_bat"]
    out["pool_bowl_diff"] = out["p1_bowl"] - out["p2_bowl"]
    return out.drop(columns=["_snap", "_fg", "p1_bat", "p1_bowl", "p2_bat", "p2_bowl"])


def latest_pool(pool: pd.DataFrame) -> dict[tuple[str, str], tuple[float, float]]:
    """Newest snapshot per (team, fgroup) for live inference."""
    last = pool.sort_values("snap").groupby(["team", "fgroup"]).tail(1)
    return {(str(r["team"]), str(r["fgroup"])): (float(r["pool_bat"]), float(r["pool_bowl"]))
            for _, r in last.iterrows()}
