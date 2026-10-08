"""Vocabularies + fitted normalization stats. Saved to artifacts/."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

UNK = "__UNK__"

NUMERIC_COLS = ["elo_diff", "abs_elo_diff", "elo_win_prob", "year", "toss1",
                "form_diff", "h2h", "venue_edge", "toss_venue",
                "star_diff", "exp_diff", "draw_venue"]


class Vocab:
    def __init__(self, min_freq: int = 1, max_size: int | None = None) -> None:
        self.min_freq = min_freq
        self.max_size = max_size
        self.stoi: dict[str, int] = {UNK: 0}
        self.itos: list[str] = [UNK]

    def fit(self, series: pd.Series) -> Vocab:
        vc = series.fillna(UNK).astype(str).value_counts()
        items = [(k, v) for k, v in vc.items() if v >= self.min_freq and k != UNK]
        if self.max_size:
            items = items[: self.max_size - 1]
        for k, _ in items:
            self.stoi[str(k)] = len(self.itos)
            self.itos.append(str(k))
        return self

    def encode(self, v: object) -> int:
        if v is None or (isinstance(v, float) and v != v):
            return 0
        return self.stoi.get(str(v), 0)

    def __len__(self) -> int:
        return len(self.itos)


class FeatureStore:
    """Fitted vocabularies + normalization stats. Saved to artifacts/."""

    def __init__(self) -> None:
        # min_freq=3: single-game domestic sides share the UNK embedding
        # instead of memorizing noise (club slice was coin-flip before this).
        self.teams = Vocab(min_freq=3, max_size=600)
        self.country = Vocab(max_size=80)
        self.city = Vocab(min_freq=2, max_size=500)
        self.format = Vocab()
        self.toss_side = Vocab()
        self.toss_choice = Vocab()
        self.gender = Vocab()
        self.team_type = Vocab()
        self.tier = Vocab()
        self.num_mean: dict[str, float] = {}
        self.num_std: dict[str, float] = {}
        self.elo_mean = 0.0
        self.elo_std = 1.0
        self.year_min = 1970
        self.year_max = 2030

    def fit(self, df: pd.DataFrame) -> FeatureStore:
        teams_all = pd.concat([df["team1"], df["team2"]]).astype(str)
        self.teams.fit(teams_all)
        self.country.fit(df["venue_country"].astype(str))
        self.city.fit(df["venue_city"].astype(str))
        self.format.fit(df["format"].astype(str))
        self.toss_side.fit(df["toss_side"].astype(str))
        self.toss_choice.fit(df["toss_choice"].astype(str))
        self.gender.fit(df.get("gender", pd.Series(["male"] * len(df))).astype(str))
        self.team_type.fit(df.get("team_type", pd.Series(["international"] * len(df))).astype(str))
        self.tier.fit(df.get("tier", df["format"]).astype(str))
        for c in NUMERIC_COLS:
            if c == "year":
                continue
            v = df[c].astype(float) if c in df else pd.Series([0.5 if c in ("elo_win_prob", "h2h", "toss_venue") else 0.0] * len(df))
            self.num_mean[c] = float(v.mean())
            self.num_std[c] = float(v.std() or 1.0)
        self.elo_mean = self.num_mean.get("elo_diff", 0.0)
        self.elo_std = self.num_std.get("elo_diff", 1.0)
        self.year_min = int(df["year"].min())
        self.year_max = int(df["year"].max())
        return self

    def transform(self, df: pd.DataFrame) -> dict[str, npt.NDArray[Any]]:
        n = len(df)
        if "gender" not in df:
            df = df.copy()
            df["gender"] = "male"
        if "team_type" not in df:
            df = df.copy()
            df["team_type"] = "international"
        t1 = np.array([self.teams.encode(x) for x in df["team1"]], dtype=np.int32)
        t2 = np.array([self.teams.encode(x) for x in df["team2"]], dtype=np.int32)
        co = np.array([self.country.encode(x) for x in df["venue_country"]], dtype=np.int32)
        ci = np.array([self.city.encode(x) for x in df["venue_city"]], dtype=np.int32)
        fm = np.array([self.format.encode(x) for x in df["format"]], dtype=np.int32)
        ts = np.array([self.toss_side.encode(x) for x in df["toss_side"]], dtype=np.int32)
        tc = np.array([self.toss_choice.encode(x) for x in df["toss_choice"]], dtype=np.int32)
        gd = np.array([self.gender.encode(x) for x in df["gender"]], dtype=np.int32)
        tt = np.array([self.team_type.encode(x) for x in df["team_type"]], dtype=np.int32)
        tr = np.array([self.tier.encode(x) for x in (df["tier"] if "tier" in df else df["format"])], dtype=np.int32)
        def std_col(name: str, default: float) -> npt.NDArray[np.float32]:
            if name in df:
                v = df[name].values.astype(np.float32)
            else:
                v = np.full(n, default, np.float32)
            m = self.num_mean.get(name, 0.0 if default != 0.5 else 0.0)
            s = self.num_std.get(name, 1.0)
            if name in ("elo_win_prob", "h2h", "toss_venue"):
                return v.astype(np.float32)  # already [0,1], keep raw
            return ((v - m) / s).astype(np.float32)
        elo = std_col("elo_diff", 0.0)
        abs_elo = std_col("abs_elo_diff", 0.0)
        eprob = df["elo_win_prob"].values.astype(np.float32) if "elo_win_prob" in df else np.full(n, 0.5, np.float32)
        year = ((df["year"].values.astype(np.float32) - self.year_min) / max(1, self.year_max - self.year_min)).astype(np.float32)
        # toss from team1's view: +1 won it, -1 lost it, 0 unknown
        toss_side = df["toss_side"].values
        toss1 = np.where(toss_side == "team1", 1.0, np.where(toss_side == "team2", -1.0, 0.0)).astype(np.float32)
        form_diff = std_col("form_diff", 0.0)
        h2h_v = df["h2h"].values.astype(np.float32) if "h2h" in df else np.full(n, 0.5, np.float32)
        vedge = std_col("venue_edge", 0.0)
        tven = df["toss_venue"].values.astype(np.float32) if "toss_venue" in df else np.full(n, 0.5, np.float32)
        star = std_col("star_diff", 0.0)
        expd = std_col("exp_diff", 0.0)
        drw_v = df["draw_venue"].values.astype(np.float32) if "draw_venue" in df else np.full(n, 0.3, np.float32)
        y = df["label"].values.astype(np.int32) if "label" in df else np.zeros(n, np.int32)
        return {
            "t1": t1, "t2": t2, "country": co, "city": ci, "fmt": fm,
            "toss_side": ts, "toss_choice": tc, "gender": gd, "team_type": tt, "tier": tr,
            "elo": elo, "abs_elo": abs_elo, "elo_prob": eprob, "year": year, "toss1": toss1,
            "form_diff": form_diff, "h2h": h2h_v, "venue_edge": vedge,
            "toss_venue": tven, "star_diff": star, "exp_diff": expd, "draw_venue": drw_v, "y": y,
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "teams": self.teams.itos, "country": self.country.itos, "city": self.city.itos,
            "format": self.format.itos, "toss_side": self.toss_side.itos,
            "toss_choice": self.toss_choice.itos,
            "gender": self.gender.itos, "team_type": self.team_type.itos,
            "tier": self.tier.itos,
            "num_mean": self.num_mean, "num_std": self.num_std,
            "elo_mean": self.num_mean.get("elo_diff", 0.0), "elo_std": self.num_std.get("elo_diff", 1.0),
            "year_min": self.year_min, "year_max": self.year_max,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> FeatureStore:
        fs = cls()
        for attr, key in [("teams", "teams"), ("country", "country"), ("city", "city"),
                          ("format", "format"), ("toss_side", "toss_side"), ("toss_choice", "toss_choice"),
                          ("gender", "gender"), ("team_type", "team_type"), ("tier", "tier")]:
            v = Vocab()
            itos = d.get(key, [UNK])
            v.itos = list(itos)
            v.stoi = {t: i for i, t in enumerate(v.itos)}
            setattr(fs, attr, v)
        num_mean = d.get("num_mean", {"elo_diff": d.get("elo_mean", 0.0)})
        num_std = d.get("num_std", {"elo_diff": d.get("elo_std", 1.0)})
        fs.num_mean = {k: float(x) for k, x in num_mean.items()}
        fs.num_std = {k: float(x) for k, x in num_std.items()}
        fs.elo_mean = fs.num_mean.get("elo_diff", 0.0)
        fs.elo_std = fs.num_std.get("elo_diff", 1.0)
        fs.year_min = int(d["year_min"])
        fs.year_max = int(d["year_max"])
        return fs

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2))

    @classmethod
    def load(cls, path: Path) -> FeatureStore:
        return cls.from_json(json.loads(Path(path).read_text()))
