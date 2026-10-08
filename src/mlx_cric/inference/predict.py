"""Inference: P(team1 wins) for any fixture + live CricAPI predictions."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import numpy.typing as npt
import pandas as pd

from ..config import ARTIFACTS_DIR
from .cricapi import fetch_upcoming
from ..features import FeatureStore, add_chrono_features
from ..data.hf import load_or_build
from ..models.mlp import CricketMLP
from ..data.normalize import normalize_team, infer_country


def _history_snapshot() -> pd.DataFrame:
    """Chronological features over full history; returns frame + final state maps."""
    df_hist = add_chrono_features(load_or_build(keep_draws=True))
    return df_hist


def _elo_lookup(df_hist: pd.DataFrame) -> dict[str, float]:
    """Latest Elo per team (unseen -> 1500)."""
    last: dict[str, float] = {}
    for _, r in df_hist.sort_values("year").iterrows():
        last[r["team1"]] = float(r.get("elo1", 1500.0))
        last[r["team2"]] = float(r.get("elo2", 1500.0))
    # elo1 col stores pre-match rating + home bump; strip bump approx by keeping raw map
    # from a clean replay for exactness:
    ratings: dict[str, float] = {}
    base = 1500.0
    for _, r in df_hist.sort_values(["year"]).iterrows():
        for t in (r["team1"], r["team2"]):
            ratings.setdefault(t, base)
        if r["label"] in (0, 1):
            s1 = 1.0 if r["label"] == 0 else 0.0
            r1, r2 = ratings[r["team1"]], ratings[r["team2"]]
            exp1 = 1.0 / (1.0 + 10 ** ((r2 - r1) / 400.0))
            ratings[r["team1"]] = r1 + 20.0 * (s1 - exp1)
            ratings[r["team2"]] = r2 + 20.0 * ((1 - s1) - (1 - exp1))
    return ratings


def _live_form(team: str, df_hist: pd.DataFrame, n: int = 5) -> float:
    d = df_hist[(df_hist["team1"] == team) | (df_hist["team2"] == team)].tail(n)
    if len(d) == 0:
        return 0.5
    wins = sum(1 for _, r in d.iterrows() if (r["team1"] == team and r["label"] == 0) or (r["team2"] == team and r["label"] == 1))
    return wins / len(d)


def _live_h2h(t1: str, t2: str, df_hist: pd.DataFrame, n: int = 5) -> float:
    d = df_hist[((df_hist["team1"] == t1) & (df_hist["team2"] == t2)) | ((df_hist["team1"] == t2) & (df_hist["team2"] == t1))].tail(n)
    if len(d) == 0:
        return 0.5
    wins = sum(1 for _, r in d.iterrows() if (r["team1"] == t1 and r["label"] == 0) or (r["team2"] == t1 and r["label"] == 1))
    return wins / len(d)


def _city_rate(team: str, city: str, df_hist: pd.DataFrame) -> float:
    d = df_hist[
        (df_hist["venue_city"] == city)
        & ((df_hist["team1"] == team) | (df_hist["team2"] == team))
    ]
    if len(d) < 3:
        return 0.5
    wins = sum(
        1
        for _, r in d.iterrows()
        if (r["team1"] == team and r["label"] == 0) or (r["team2"] == team and r["label"] == 1)
    )
    return (wins + 1.5) / (len(d) + 3.0)


def _city_toss_rate(city: str, df_hist: pd.DataFrame) -> float:
    d = df_hist[
        (df_hist["venue_city"] == city) & (df_hist["toss_side"].isin(["team1", "team2"]))
    ]
    if len(d) < 3:
        return 0.5
    wins = sum(
        1
        for _, r in d.iterrows()
        if (r["toss_side"] == "team1" and r["label"] == 0)
        or (r["toss_side"] == "team2" and r["label"] == 1)
    )
    return (wins + 1.0) / (len(d) + 2.0)


def _city_draw_rate(city: str, df_hist: pd.DataFrame) -> float:
    d = df_hist[(df_hist["venue_city"] == city) & (df_hist["format"] == "TEST")]
    if len(d) < 3 or "label" not in d:
        return 0.3
    w = int((d["label"] == 2).sum())
    return (w + 1.5) / (len(d) + 5.0)


def _days_rest(team: str, at: pd.Timestamp, df_hist: pd.DataFrame) -> float:
    d = df_hist[(df_hist["team1"] == team) | (df_hist["team2"] == team)]
    if not len(d):
        return 30.0
    last = pd.to_datetime(d["date"], errors="coerce").max()
    if pd.isna(last):
        return 30.0
    return max(0.0, min(60.0, (at - last).days))


def split_venue(venue: str) -> tuple[str, str]:
    """'Dubai International Cricket Stadium, Dubai' -> (stadium, city)."""
    parts = [p.strip() for p in str(venue or "").split(",") if p.strip()]
    if len(parts) >= 2:
        return ", ".join(parts[:-1]), parts[-1]
    if len(parts) == 1:
        return parts[0], parts[0]
    return "Unknown", "Unknown"


class Predictor:
    def __init__(self, artifacts: Path = ARTIFACTS_DIR):
        import json as _json
        import pickle as _pickle
        self.artifacts = Path(artifacts)
        self.fs = FeatureStore.load(self.artifacts / "features.json")
        try:
            _cfg = _json.loads((self.artifacts / "model_config.json").read_text())
            _team_dim, _ncls = int(_cfg.get("team_dim", 16)), int(_cfg.get("n_classes", 2))
            self.hgb_keys: list[str] = list(_cfg.get("hgb_keys", []))
            _n_seeds = int(_cfg.get("n_seeds", 1))
        except Exception:
            _team_dim, _ncls, _n_seeds = 16, 2, 1
            self.hgb_keys = []
        self.models: list[CricketMLP] = []
        for _s in range(_n_seeds):
            _m = CricketMLP(
                n_teams=len(self.fs.teams), n_country=len(self.fs.country),
                n_city=len(self.fs.city), n_fmt=len(self.fs.format),
                n_toss_side=max(4, len(self.fs.toss_side)),
                n_toss_choice=max(4, len(self.fs.toss_choice)),
                n_gender=max(3, len(getattr(self.fs, "gender", [0, 1]))),
                n_team_type=max(3, len(getattr(self.fs, "team_type", [0, 1]))),
                n_tier=max(8, len(getattr(self.fs, "tier", [0] * 8))),
                team_dim=_team_dim, n_classes=_ncls,
            )
            _wf = self.artifacts / f"model_{_s}.safetensors"
            if not _wf.exists():
                _wf = self.artifacts / "model.safetensors"
            _m.load_weights(str(_wf))
            _m.eval()
            self.models.append(_m)
        try:
            tp = self.artifacts / "temperature.json"
            self.temperature = float(_json.loads(tp.read_text())["temperature"]) if tp.exists() else 1.0
        except Exception:
            self.temperature = 1.0
        self.hgb: Any = None
        self.draw_lr: Any = None
        self.stacker: Any = None
        self.w_mlp: float = 1.0
        try:
            with open(self.artifacts / "hgb.pkl", "rb") as _f:
                self.hgb = _pickle.load(_f)
            _ens = _json.loads((self.artifacts / "ensemble.json").read_text())
            self.w_mlp = float(_ens.get("w_mlp", 0.5))
        except Exception:
            pass
        try:
            with open(self.artifacts / "draw_lr.pkl", "rb") as _f:
                self.draw_lr = _pickle.load(_f)
        except Exception:
            pass
        try:
            with open(self.artifacts / "stacker.pkl", "rb") as _f:
                self.stacker = _pickle.load(_f)
        except Exception:
            pass
        try:
            _rt = _json.loads((self.artifacts / "ratings.json").read_text())
            self.ratings_map: dict[tuple[str, str], tuple[float, float]] = {}
            for _k, _v in _rt.items():
                _t, _, _g = _k.partition("||")
                self.ratings_map[(str(_t), str(_g))] = (float(_v["bat"]), float(_v["bowl"]))
        except Exception:
            self.ratings_map = {}
        try:
            self.hist = _history_snapshot()
            self.elo_map = _elo_lookup(self.hist)
        except Exception:
            self.hist = pd.DataFrame()
            self.elo_map = {}

    def _frame(
        self,
        team1: str,
        team2: str,
        venue_country: str = "Unknown",
        venue_city: str = "Unknown",
        venue_stadium: str = "Unknown",
        toss_side: str = "none",
        toss_choice: str = "none",
        fmt: str = "T20",
        year: int | None = None,
        month: int | None = None,
        gender: str | None = None,
        team_type: str | None = None,
    ) -> pd.DataFrame:
        import math as _math
        from datetime import datetime, timezone as _tz
        now = datetime.now(_tz.utc).replace(tzinfo=None)
        team1, team2 = normalize_team(team1), normalize_team(team2)
        year = year or now.year
        month = month or now.month
        msin, mcos = _math.sin(2 * _math.pi * month / 12.0), _math.cos(2 * _math.pi * month / 12.0)
        at = pd.Timestamp(now)
        # auto-detect women's + domestic/club when not given
        if gender is None:
            gender = "female" if ("women" in team1.lower() or "women" in team2.lower()) else "male"
        if team_type is None:
            team_type = infer_team_type(team1, team2)
        if not venue_country or venue_country == "Unknown":
            venue_country = infer_country(venue_city, venue_stadium or venue_country)
        # normalize legacy format names to v2: ODI/T20/TEST
        fmt = {"T20I": "T20", "IPL-T20": "T20", "ODI": "ODI", "T20": "T20", "TEST": "TEST"}.get(fmt, fmt)
        # tier: domestic multi-day draws far more often than Tests (38% vs 19%)
        tier = "MDM" if (fmt == "TEST" and team_type == "club") else {"TEST": "Test", "ODI": "ODI", "T20": "T20"}.get(fmt, fmt)
        e1 = self.elo_map.get(team1, 1500.0)
        e2 = self.elo_map.get(team2, 1500.0)
        diff = (e1 - e2) / 400.0
        prob = 1.0 / (1.0 + 10 ** (-diff))
        # live rolling form from history (causal: history ends before today)
        if len(self.hist):
            f1, f2 = _live_form(team1, self.hist), _live_form(team2, self.hist)
            h = _live_h2h(team1, team2, self.hist)
            vedge = _city_rate(team1, venue_city, self.hist) - _city_rate(team2, venue_city, self.hist)
            tven = _city_toss_rate(venue_city, self.hist)
            drw_v = _city_draw_rate(venue_city, self.hist)
            rstd = max(-30.0, min(30.0, _days_rest(team1, at, self.hist) - _days_rest(team2, at, self.hist))) / 30.0
        else:
            f1 = f2 = 0.5
            h = 0.5
            vedge = 0.0
            tven = 0.5
            drw_v = 0.3
            rstd = 0.0
        fg = {"ODI": "ODI", "T20": "T20", "TEST": "MULTI"}.get(fmt, fmt)
        b1, w1 = self.ratings_map.get((team1, fg), (0.0, 0.0))
        b2, w2 = self.ratings_map.get((team2, fg), (0.0, 0.0))
        df = pd.DataFrame([{
            "team1": team1, "team2": team2,
            "venue_stadium": venue_stadium, "venue_city": venue_city,
            "venue_country": venue_country, "toss_side": toss_side,
            "toss_choice": toss_choice, "format": fmt, "year": year,
            "elo_diff": diff, "elo_win_prob": prob, "label": 0,
            "gender": gender, "team_type": team_type, "tier": tier,
            "form_diff": f1 - f2, "h2h": h, "venue_edge": vedge,
            "toss_venue": tven, "star_diff": 0.0, "exp_diff": 0.0, "draw_venue": drw_v,
            "rest_diff": rstd, "month_sin": msin, "month_cos": mcos,
        }])
        return df

    def _hgb_proba(self, arr: dict[str, npt.NDArray[Any]]) -> npt.NDArray[np.float64] | None:
        if self.hgb is None:
            return None
        try:
            import numpy as _np
            cols = [_np.asarray(arr[k]).reshape(len(arr["y"]), -1).astype(float) for k in self.hgb_keys]
            X = _np.concatenate(cols, axis=1)
            return _np.asarray(self.hgb.predict_proba(X), dtype=float)
        except Exception:
            return None

    def _draw_proba(self, row: pd.Series) -> float | None:
        if self.draw_lr is None:
            return None
        try:
            import numpy as _np
            X = _np.array([[
                1.0 if str(row.get("tier", "")) == "MDM" else 0.0,
                float(row.get("draw_venue", 0.3)),
                abs(float(row.get("elo_diff", 0.0))),
                float(row.get("year", 2025)),
            ]])
            sc, clf = self.draw_lr["scaler"], self.draw_lr["model"]
            return float(clf.predict_proba(sc.transform(X))[0][1])
        except Exception:
            return None

    def predict_proba(self, team1: str, team2: str, **kw: Any) -> dict[str, Any]:
        df = self._frame(team1, team2, **kw)
        arr = self.fs.transform(df)
        batch = {k: mx.array(v) for k, v in arr.items()}
        seed_logits: list[npt.NDArray[np.float64]] = []
        for _m in self.models:
            _lg = _m(batch)
            mx.eval(_lg)
            seed_logits.append(np.array(_lg, dtype=np.float64))
        logits = np.mean(seed_logits, axis=0)
        l = logits / max(0.05, self.temperature)
        e = np.exp(l - l.max(axis=1, keepdims=True))
        mlp_p = e / e.sum(axis=1, keepdims=True)
        hgb_p = self._hgb_proba(arr)
        probs = mlp_p
        if hgb_p is not None:
            if self.stacker is not None:
                import numpy as _np2
                elo1 = float(_np2.asarray(arr["elo_prob"])[0])
                X = _np2.array([[float(mlp_p[0][1]), float(hgb_p[0][1]), elo1]])
                probs = _np2.asarray(self.stacker["model"].predict_proba(X), dtype=float)
            else:
                probs = self.w_mlp * mlp_p + (1 - self.w_mlp) * hgb_p
        p = probs[0]
        row = df.iloc[0]
        out: dict[str, Any] = {
            "team1": team1, "team2": team2,
            "p_team1": float(p[0]), "p_team2": float(p[1]),
            "venue_country": str(row["venue_country"]),
            "venue_city": str(row["venue_city"]),
            "toss_side": str(row["toss_side"]),
            "toss_choice": str(row["toss_choice"]),
            "fmt": str(row["format"]),
            "gender": str(row["gender"]),
            "team_type": str(row["team_type"]),
        }
        # Tests can be drawn — split off draw probability first (two-stage).
        if str(row["format"]) == "TEST":
            p_draw = self._draw_proba(row)
            if p_draw is not None:
                p1, p2 = float(p[0]) * (1 - p_draw), float(p[1]) * (1 - p_draw)
                out.update({"p_team1": p1, "p_draw": p_draw, "p_team2": p2})
                best = max(("team1", p1), ("draw", p_draw), ("team2", p2), key=lambda t: t[1])
                out["winner"] = {"team1": team1, "draw": "Draw", "team2": team2}[best[0]]
                out["confidence"] = float(best[1])
                return out
        winner = team1 if p[0] >= p[1] else team2
        out["winner"] = winner
        out["confidence"] = float(max(p[0], p[1]))
        return out

    def predict_upcoming(self, limit: int = 20) -> list[dict[str, Any]]:
        fixtures = fetch_upcoming(limit=limit)
        out: list[dict[str, Any]] = []
        for f in fixtures:
            if f["state"] == "result":
                continue
            fmt = {"odi": "ODI", "t20": "T20", "test": "TEST"}.get(
                str(f.get("match_type", "")).lower(), "T20"
            )
            stadium, city = split_venue(str(f.get("venue", "") or ""))
            venue_country = infer_country(city, stadium)
            dt = pd.to_datetime(f.get("date"), errors="coerce")
            try:
                out.append(self.predict_proba(
                    str(f["team1"]), str(f["team2"]),
                    venue_country=venue_country, venue_city=city, venue_stadium=stadium,
                    fmt=fmt, year=datetime.now(timezone.utc).year,
                    month=int(dt.month) if not pd.isna(dt) else None,
                ) | {"status": f.get("status"), "match_type": f.get("match_type"),
                     "date": f.get("date"), "venue": f.get("venue")})
            except Exception as e:
                out.append({"team1": f["team1"], "team2": f["team2"], "error": str(e)})
        return sorted(out, key=lambda x: -float(x.get("confidence", 0)))


COUNTRY_TEAMS = {
    "india", "australia", "england", "pakistan", "south africa", "new zealand",
    "sri lanka", "west indies", "bangladesh", "zimbabwe", "afghanistan",
    "ireland", "netherlands", "scotland", "namibia", "nepal", "oman",
    "papua new guinea", "usa", "canada", "kenya", "hong kong", "singapore",
    "malaysia", "germany", "gibraltar", "israel", "isle of man",
}

def infer_team_type(t1: str, t2: str) -> str:
    """International if both look like countries (or women country sides), else club."""
    def is_country(t: str) -> bool:
        tl = t.lower().replace(" women", "").strip()
        return tl in COUNTRY_TEAMS or len(tl.split()) <= 2 and tl not in {
            "mumbai indians", "chennai super kings", "royal challengers bangalore",
            "kolkata knight riders", "delhi daredevils", "rajasthan royals",
            "sunrisers hyderabad", "kings xi punjab",
        } and " " not in tl or tl in COUNTRY_TEAMS
    # simple: known countries -> international, else club (franchise/domestic e.g. Gujarat vs Odisha)
    c1 = t1.lower().replace(" women", "").strip() in COUNTRY_TEAMS
    c2 = t2.lower().replace(" women", "").strip() in COUNTRY_TEAMS
    return "international" if (c1 and c2) else "club"


def _guess_format(match_type: str) -> str:
    m = str(match_type or "").lower()
    if "odi" in m or "50" in m or "one" in m:
        return "ODI"
    if "test" in m:
        return "TEST"
    return "T20"
