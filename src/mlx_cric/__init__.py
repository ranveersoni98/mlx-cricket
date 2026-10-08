"""mlx-cric: MLX cricket result predictor (HF history + CricAPI live)."""
from __future__ import annotations

import argparse
import json


def main() -> None:
    p = argparse.ArgumentParser(prog="mlx-cric", description="Predict cricket results with MLX")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="Train MLX model on HF datasets")
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--batch-size", type=int, default=512)
    t.add_argument("--lr", type=float, default=3e-3)
    t.add_argument("--label-smoothing", type=float, default=0.05)
    t.add_argument("--patience", type=int, default=8)
    t.add_argument("--team-dim", type=int, default=32)
    t.add_argument("--seeds", type=int, default=5)
    t.add_argument("--no-baseline", action="store_true")
    t.add_argument("--rebuild", action="store_true")

    pr = sub.add_parser("predict", help="Predict one match")
    pr.add_argument("--team1", required=True)
    pr.add_argument("--team2", required=True)
    pr.add_argument("--venue-country", default="Unknown")
    pr.add_argument("--venue-city", default="Unknown")
    pr.add_argument("--toss-side", default="none", choices=["team1", "team2", "none"])
    pr.add_argument("--toss-choice", default="none", choices=["bat", "bowl", "none"])
    pr.add_argument("--format", dest="fmt", default="T20")
    pr.add_argument("--gender", default=None, choices=["male", "female"])
    pr.add_argument("--team-type", default=None, choices=["international", "club"])

    up = sub.add_parser("upcoming", help="Predict upcoming CricAPI fixtures")
    up.add_argument("--limit", type=int, default=15)

    ev = sub.add_parser("evaluate", help="Walk-forward test metrics + per-slice (no training)")
    ev.add_argument("--rebuild", action="store_true")

    sub.add_parser("check-data", help="Download HF + sample CricAPI status")

    a = p.parse_args()
    if a.cmd == "train":
        from .training import train
        hist = train(epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
                     label_smoothing=a.label_smoothing, patience=a.patience,
                     team_dim=a.team_dim, n_seeds=a.seeds, force_rebuild=a.rebuild,
                     run_baseline=not a.no_baseline)
        print(json.dumps(hist, indent=2))
    elif a.cmd == "predict":
        from .inference import Predictor
        out = Predictor().predict_proba(
            a.team1, a.team2, venue_country=a.venue_country,
            venue_city=a.venue_city, toss_side=a.toss_side,
            toss_choice=a.toss_choice, fmt=a.fmt,
            gender=a.gender, team_type=getattr(a, "team_type", None),
        )
        print(json.dumps(out, indent=2))
    elif a.cmd == "upcoming":
        from .inference import Predictor
        for o in Predictor().predict_upcoming(limit=a.limit):
            print(json.dumps(o, indent=2))
    elif a.cmd == "evaluate":
        from .training import evaluate_saved
        print(json.dumps(evaluate_saved(rebuild=a.rebuild), indent=2))
    elif a.cmd == "check-data":
        from .data import build_unified, download_raw
        from .inference import get_cric_score, get_current_matches
        paths = download_raw()
        print("HF files:", {k: str(v) for k, v in paths.items()})
        df = build_unified(paths)
        print(f"unified: {len(df)} rows, formats={df['format'].value_counts().to_dict()}")
        if "gender" in df:
            print(f"gender={df['gender'].value_counts().to_dict()} team_type={df['team_type'].value_counts().to_dict()}")
        print("cricScore sample:", str(get_cric_score())[:400])
        print("currentMatches sample:", str(get_current_matches())[:400])
