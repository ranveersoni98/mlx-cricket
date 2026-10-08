"""MLX winner-prediction model: team/categorical embeddings + MLP."""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class CricketMLP(nn.Module):
    def __init__(
        self,
        n_teams: int,
        n_country: int,
        n_city: int,
        n_fmt: int,
        n_toss_side: int = 4,
        n_toss_choice: int = 4,
        n_gender: int = 3,
        n_team_type: int = 3,
        n_tier: int = 8,
        team_dim: int = 32,
        cat_dim: int = 8,
        n_classes: int = 2,
        hidden: tuple[int, ...] = (128, 64, 32),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.team_emb = nn.Embedding(n_teams, team_dim)
        self.country_emb = nn.Embedding(n_country, cat_dim)
        self.city_emb = nn.Embedding(n_city, cat_dim)
        self.fmt_emb = nn.Embedding(n_fmt, cat_dim)
        self.ts_emb = nn.Embedding(n_toss_side, 4)
        self.tc_emb = nn.Embedding(n_toss_choice, 4)
        self.gender_emb = nn.Embedding(max(2, n_gender), 4)
        self.ttype_emb = nn.Embedding(max(2, n_team_type), 4)
        self.tier_emb = nn.Embedding(max(2, n_tier), 4)

        n_num = 12  # + abs_elo_diff (mismatch size), draw_venue (draw prior)
        in_dim = team_dim * 2 + cat_dim * 3 + 4 * 2 + 4 + 4 + 4 + n_num
        layers: list = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.mlp = nn.Sequential(*layers)
        self.n_classes = n_classes

    def __call__(self, batch: dict[str, mx.array]) -> mx.array:
        t1 = self.team_emb(batch["t1"])
        t2 = self.team_emb(batch["t2"])
        co = self.country_emb(batch["country"])
        ci = self.city_emb(batch["city"])
        fm = self.fmt_emb(batch["fmt"])
        ts = self.ts_emb(batch["toss_side"])
        tc = self.tc_emb(batch["toss_choice"])
        gd = self.gender_emb(batch.get("gender", mx.zeros_like(batch["t1"])))
        tt = self.ttype_emb(batch.get("team_type", mx.zeros_like(batch["t1"])))
        tr = self.tier_emb(batch.get("tier", mx.zeros_like(batch["t1"])))
        def col(name: str, default: float = 0.5) -> mx.array:
            if name in batch:
                v = batch[name]
            else:
                v = mx.full_like(batch["elo"], default)
            return v[..., None]
        num = mx.concatenate(
            [
                col("elo", 0.0), col("abs_elo", 0.0), col("elo_prob", 0.5), col("year", 0.5), col("toss1", 0.0),
                col("form_diff", 0.0), col("h2h", 0.5), col("venue_edge", 0.0),
                col("toss_venue", 0.5), col("star_diff", 0.0), col("exp_diff", 0.0),
                col("draw_venue", 0.3),
            ],
            axis=-1,
        )
        x = mx.concatenate([t1, t2, co, ci, fm, ts, tc, gd, tt, tr, num], axis=-1)
        return self.mlp(x)


def accuracy(logits: mx.array, y: mx.array) -> mx.array:
    return mx.mean(mx.argmax(logits, axis=-1) == y)
