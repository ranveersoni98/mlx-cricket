"""Training loop: walk-forward split, label smoothing, early stopping,
temperature scaling, per-slice metrics, sklearn baseline + ensemble."""
from __future__ import annotations

import json
import pickle
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import numpy.typing as npt
import pandas as pd

from ..config import ARTIFACTS_DIR, PROCESSED_DIR, RANDOM_SEED, TRAIN_END_YEAR, VAL_END_YEAR, ODI_CUTOFF_YEAR
from ..data.hf import load_or_build, save_processed
from ..features import FeatureStore, add_chrono_features, fit_temperature
from ..models.mlp import CricketMLP, accuracy

Arrays = dict[str, npt.NDArray[Any]]

# Column order for the sklearn sidecar (must match Predictor).
HGB_KEYS: list[str] = [
    "elo", "abs_elo", "elo_prob", "year", "toss1", "form_diff", "h2h",
    "venue_edge", "toss_venue", "star_diff", "exp_diff", "draw_venue",
    "t1", "t2", "country", "city", "fmt", "toss_side",
    "toss_choice", "gender", "team_type", "tier",
]


def to_mx(batch: Arrays) -> dict[str, mx.array]:
    return {k: mx.array(v) for k, v in batch.items()}


def batches(arrays: Arrays, bs: int, shuffle: bool, seed: int = 0) -> Iterator[Arrays]:
    n = len(arrays["y"])
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    for s in range(0, n, bs):
        j = idx[s : s + bs]
        yield {k: v[j] for k, v in arrays.items()}


def to_hgb_matrix(A: Arrays) -> npt.NDArray[np.float64]:
    n = len(A["y"])
    cols = [np.asarray(A[k]).reshape(n, -1).astype(np.float64) for k in HGB_KEYS]
    return np.concatenate(cols, axis=1)


def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tr = df[df["year"] <= TRAIN_END_YEAR].reset_index(drop=True)
    va = df[(df["year"] > TRAIN_END_YEAR) & (df["year"] <= VAL_END_YEAR)].reset_index(drop=True)
    te = df[df["year"] > VAL_END_YEAR].reset_index(drop=True)
    # fallback if a bucket is empty (tiny datasets)
    if len(va) == 0 or len(te) == 0:
        from sklearn.model_selection import train_test_split
        tr2, rest = train_test_split(df, test_size=0.3, random_state=RANDOM_SEED, stratify=df["label"])
        va2, te2 = train_test_split(rest, test_size=0.5, random_state=RANDOM_SEED, stratify=cast(pd.DataFrame, rest)["label"])
        return (
            cast(pd.DataFrame, tr2).reset_index(drop=True),
            cast(pd.DataFrame, va2).reset_index(drop=True),
            cast(pd.DataFrame, te2).reset_index(drop=True),
        )
    return tr, va, te


def softmax(logits: npt.NDArray[np.float64], temp: float = 1.0) -> npt.NDArray[np.float64]:
    scaled = logits / max(0.05, temp)
    e = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def log_loss_and_brier(probs: npt.NDArray[np.float64], y: npt.NDArray[Any]) -> tuple[float, float]:
    eps = 1e-7
    p = np.clip(probs, eps, 1 - eps)
    yi = np.asarray(y).astype(int)
    ll = float(-np.log(p[np.arange(len(yi)), yi]).mean())
    onehot = np.zeros_like(p)
    onehot[np.arange(len(yi)), yi] = 1
    brier = float(np.mean((p - onehot) ** 2))
    return ll, brier


def slice_metrics(df: pd.DataFrame, probs: npt.NDArray[np.float64], pred: npt.NDArray[Any]) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    y = np.asarray(df["label"].values)
    for col in ["format", "gender", "team_type"]:
        if col not in df:
            continue
        groups = df.groupby(col).groups
        for val, idx in groups.items():
            ii = np.array(list(idx))
            acc = float((np.asarray(pred[ii]) == y[ii]).mean()) if len(ii) else 0.0
            out[f"{col}={val}"] = {"n": int(len(ii)), "acc": round(acc, 4)}
    return out


def train_hgb(Atr: Arrays, Ava: Arrays, Ate: Arrays) -> tuple[Any, float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Fit HistGradientBoosting; return (model, test_acc, val_proba, test_proba)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import accuracy_score

    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, random_state=RANDOM_SEED)
    clf.fit(to_hgb_matrix(Atr), np.asarray(Atr["y"]))
    val_proba: npt.NDArray[np.float64] = clf.predict_proba(to_hgb_matrix(Ava))
    test_proba: npt.NDArray[np.float64] = clf.predict_proba(to_hgb_matrix(Ate))
    acc = float(accuracy_score(np.asarray(Ate["y"]), test_proba.argmax(axis=1)))
    return clf, acc, val_proba, test_proba


DRAW_FEATS = ["tier_mdm", "draw_venue", "abs_elo", "year"]


def draw_features(d: pd.DataFrame) -> tuple[npt.NDArray[np.float64], npt.NDArray[Any]]:
    """Small, robust draw-signal columns (a 500-tree HGB overfit these; LogReg transfers)."""
    t = d[d["format"] == "TEST"].reset_index(drop=True)
    X = np.column_stack([
        (t["tier"].values == "MDM").astype(float),
        t["draw_venue"].values.astype(float) if "draw_venue" in t else np.full(len(t), 0.3),
        t["abs_elo_diff"].values.astype(float) if "abs_elo_diff" in t else np.zeros(len(t)),
        t["year"].values.astype(float),
    ])
    return X, (t["label"].values == 2).astype(int)


def train_draw_hgb(
    dtr: pd.DataFrame, dva: pd.DataFrame, dte: pd.DataFrame
) -> tuple[Any, dict[str, Any]]:
    """Draw-vs-decisive LogReg on 4 robust columns (val/test AUC ~0.62).

    Returns (bundle, info); bundle = {"scaler", "model"} pickled to draw_lr.pkl.
    (Kept the old name so call sites don't churn.)
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    Xtr, ytr = draw_features(dtr)
    Xva, yva = draw_features(dva)
    Xte, yte = draw_features(dte)
    info: dict[str, Any] = {
        "n_train": int(len(ytr)), "n_val": int(len(yva)), "n_test": int(len(yte)),
        "draw_rate_train": round(float(np.asarray(ytr).mean()), 4) if len(ytr) else 0.0,
    }
    if len(ytr) == 0 or len(yva) == 0 or len(yte) == 0:
        info["skipped"] = True
        return None, info
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.5, max_iter=1000).fit(sc.transform(Xtr), np.asarray(ytr))
    from sklearn.metrics import roc_auc_score
    va_p = clf.predict_proba(sc.transform(Xva))[:, 1]
    te_p = clf.predict_proba(sc.transform(Xte))[:, 1]
    yva_i = np.asarray(yva).astype(int)
    yte_i = np.asarray(yte).astype(int)
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.2, 0.8, 0.05):
        pred = (va_p >= t).astype(int)
        tp = int(((pred == 1) & (yva_i == 1)).sum())
        fp = int(((pred == 1) & (yva_i == 0)).sum())
        fn = int(((pred == 0) & (yva_i == 1)).sum())
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    te_pred = (te_p >= best_t).astype(int)
    info.update({
        "threshold": round(best_t, 2), "val_f1": round(float(best_f1), 4),
        "test_acc": round(float((te_pred == yte_i).mean()), 4),
        "test_draw_rate": round(float(yte_i.mean()), 4),
    })
    try:
        info["val_auc"] = round(float(roc_auc_score(yva_i, va_p)), 4)
        info["test_auc"] = round(float(roc_auc_score(yte_i, te_p)), 4)
    except Exception:
        pass
    return {"scaler": sc, "model": clf}, info


def draw_proba_bundle(bundle: Any, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.asarray(bundle["model"].predict_proba(bundle["scaler"].transform(X))[:, 1], dtype=float)


def tune_ensemble_weight(
    mlp_val: npt.NDArray[np.float64], hgb_val: npt.NDArray[np.float64], y: npt.NDArray[Any]
) -> float:
    yi = np.asarray(y).astype(int)
    best_w, best_ll = 0.5, float("inf")
    for w in np.arange(0.0, 1.01, 0.05):
        mix = w * mlp_val + (1 - w) * hgb_val
        ll, _ = log_loss_and_brier(mix, yi)
        if ll < best_ll:
            best_ll, best_w = ll, float(w)
    return best_w


def train(
    epochs: int = 30,
    batch_size: int = 512,
    lr: float = 3e-3,
    label_smoothing: float = 0.05,
    patience: int = 6,
    team_dim: int = 16,
    force_rebuild: bool = False,
    run_baseline: bool = True,
    artifacts: Path = ARTIFACTS_DIR,
) -> dict[str, Any]:
    artifacts = Path(artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)

    df = load_or_build(force=force_rebuild, keep_draws=True)
    if "method" not in df.columns or int((df["label"] == 2).sum()) == 0:
        print("stale processed file (no method/draws) — rebuilding")
        df = load_or_build(force=True, keep_draws=True)
    if df.empty:
        raise RuntimeError("Empty dataset — check HF downloads.")
    n_draws = int((df["label"] == 2).sum()) if "label" in df else 0
    print(f"draws in frame: {n_draws} (two-stage draw model, not dropped)")
    save_processed(df, PROCESSED_DIR / "matches_unified_draws.csv")
    df = add_chrono_features(df)

    # Winner model trains on decisive, fair-weather, modern-ODI games:
    # D/L (rain-rule) labels are weather lotteries, unknown pre-match;
    # pre-2000 ODI was a different sport (60 overs, no field restrictions).
    fair = df[(df["label"] < 2) & (~df["method"].isin(["D/L", "VJD"]))].reset_index(drop=True)
    fair = fair[~((fair["format"] == "ODI") & (fair["year"] < ODI_CUTOFF_YEAR))].reset_index(drop=True)
    n_dl = int(df["method"].isin(["D/L", "VJD"]).sum())
    print(f"D/L-affected rows excluded from winner training: {n_dl}")
    save_processed(fair, PROCESSED_DIR / "matches_unified.csv")

    tr, va, te = time_split(fair)
    # parallel splits of the full frame (with draws) for the TEST draw model
    dtr, dva, dte = time_split(df)
    print(f"walk-forward train<={TRAIN_END_YEAR}: {len(tr)}, val: {len(va)}, test>{VAL_END_YEAR}: {len(te)}")
    fs = FeatureStore().fit(tr)
    fs.save(artifacts / "features.json")
    Atr, Ava, Ate = (fs.transform(d) for d in (tr, va, te))
    print("train label balance:", tr["label"].value_counts(normalize=True).to_dict())
    print("formats:", tr["format"].value_counts().to_dict())

    n_classes = int(fair["label"].nunique())
    (artifacts / "model_config.json").write_text(json.dumps(
        {"team_dim": team_dim, "n_classes": n_classes, "hgb_keys": HGB_KEYS}, indent=2))
    model = CricketMLP(
        n_teams=len(fs.teams), n_country=len(fs.country),
        n_city=len(fs.city), n_fmt=len(fs.format),
        n_toss_side=max(4, len(fs.toss_side)), n_toss_choice=max(4, len(fs.toss_choice)),
        n_gender=max(3, len(fs.gender)), n_team_type=max(3, len(fs.team_type)),
        n_tier=max(8, len(fs.tier)),
        team_dim=team_dim, n_classes=n_classes,
    )
    mx.eval(model.parameters())
    opt = optim.Adam(learning_rate=lr)

    def loss_fn(
        m: CricketMLP, b: dict[str, mx.array]
    ) -> tuple[mx.array, mx.array]:
        logits = m(b)
        if label_smoothing and n_classes == 2:
            # manual smoothing: y_s = (1-eps)*onehot + eps/K
            logp = nn.log_softmax(logits, axis=-1)
            ce = nn.losses.cross_entropy(logits, b["y"], reduction="none")
            smooth_loss = -logp.mean(axis=-1)
            loss = ((1 - label_smoothing) * ce + label_smoothing * smooth_loss).mean()
            return loss, logits
        ce_mean = nn.losses.cross_entropy(logits, b["y"], reduction="mean")
        return ce_mean, logits

    def step(batch: dict[str, mx.array]) -> tuple[mx.array, mx.array]:
        (loss, logits), grads = nn.value_and_grad(model, loss_fn)(model, batch)
        opt.update(model, grads)
        return loss, logits

    def eval_split(A: Arrays) -> tuple[float, npt.NDArray[np.float64], npt.NDArray[Any]]:
        logit_list: list[npt.NDArray[np.float64]] = []
        y_list: list[npt.NDArray[Any]] = []
        vacc, vtot = 0.0, 0
        for b_np in batches(A, batch_size * 4, shuffle=False):
            b = to_mx(b_np)
            logits = model(b)
            mx.eval(logits)
            logit_list.append(np.array(logits, dtype=np.float64))
            y_list.append(np.asarray(b_np["y"]))
            acc_arr = accuracy(logits, b["y"])
            vacc += float(np.asarray(acc_arr).item()) * len(b_np["y"])
            vtot += len(b_np["y"])
        return vacc / max(1, vtot), np.concatenate(logit_list), np.concatenate(y_list)

    best_acc, best_epoch, bad = 0.0, -1, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tot, nb = 0.0, 0
        for b_np in batches(Atr, batch_size, shuffle=True, seed=ep):
            b = to_mx(b_np)
            loss, _logits = step(b)
            mx.eval(loss, model.parameters(), opt.state)
            tot += float(np.asarray(loss).item())
            nb += 1
        vacc, _, _ = eval_split(Ava)
        print(f"epoch {ep:02d} loss={tot/max(1,nb):.4f} val_acc={vacc:.4f} ({time.time()-t0:.1f}s)")
        if vacc > best_acc + 1e-4:
            best_acc, best_epoch, bad = vacc, ep, 0
            model.save_weights(str(artifacts / "model.safetensors"))
        else:
            bad += 1
            if bad >= patience:
                print(f"early stop at epoch {ep} (patience {patience})")
                break

    model.load_weights(str(artifacts / "model.safetensors"))
    # temperature scaling on val
    _, val_logits, val_y = eval_split(Ava)
    temp = fit_temperature(val_logits, val_y)
    (artifacts / "temperature.json").write_text(json.dumps({"temperature": temp}, indent=2))
    print(f"temperature={temp:.2f}")

    # final test eval with temperature
    test_acc, test_logits, test_y = eval_split(Ate)
    probs = softmax(test_logits, temp)
    pred = probs.argmax(axis=1)
    ll, brier = log_loss_and_brier(probs, test_y)
    per_slice = slice_metrics(te, probs, pred)
    print(f"TEST acc={test_acc:.4f} (T={temp:.2f} scaled acc={float((pred==np.asarray(test_y)).mean()):.4f}) ll={ll:.4f} brier={brier:.4f}")
    for k, v in sorted(per_slice.items()):
        print(f"  {k}: {v}")

    hist: dict[str, Any] = {"best_val_acc": best_acc, "best_epoch": best_epoch,
            "test_acc": float((pred == np.asarray(test_y)).mean()), "test_logloss": ll,
            "test_brier": brier, "temperature": temp,
            "n_train": len(tr), "n_val": len(va), "n_test": len(te),
            "per_slice": per_slice, "n_classes": n_classes,
            "label_smoothing": label_smoothing, "team_dim": team_dim}
    if run_baseline:
        try:
            clf, hgb_acc, hgb_val_proba, hgb_test_proba = train_hgb(Atr, Ava, Ate)
            mlp_val_proba = softmax(val_logits, temp)
            w_mlp = tune_ensemble_weight(mlp_val_proba, hgb_val_proba, val_y)
            ens_test = w_mlp * probs + (1 - w_mlp) * hgb_test_proba
            ens_pred = ens_test.argmax(axis=1)
            ens_acc = float((ens_pred == np.asarray(test_y)).mean())
            ens_ll, ens_brier = log_loss_and_brier(ens_test, test_y)
            with open(artifacts / "hgb.pkl", "wb") as f:
                pickle.dump(clf, f)
            (artifacts / "ensemble.json").write_text(json.dumps(
                {"w_mlp": w_mlp, "ensemble_acc": ens_acc,
                 "ensemble_logloss": ens_ll, "ensemble_brier": ens_brier}, indent=2))
            hist.update({"baseline_hgb_acc": hgb_acc, "ensemble_acc": ens_acc,
                         "ensemble_logloss": ens_ll, "ensemble_brier": ens_brier,
                         "w_mlp": w_mlp})
            print(f"baseline HGB={hgb_acc:.4f} ensemble(w_mlp={w_mlp:.2f})={ens_acc:.4f} ll={ens_ll:.4f}")
        except Exception as e:
            print("baseline skipped:", e)
    # --- Stage A: TEST draw model + 3-way accuracy (draws are 33% of Tests) ---
    try:
        draw_bundle, draw_info = train_draw_hgb(dtr, dva, dte)
        print("draw model:", draw_info)
        hist["draw_model"] = draw_info
        if draw_bundle is not None:
            with open(artifacts / "draw_lr.pkl", "wb") as f:
                pickle.dump(draw_bundle, f)
            te_test = dte[dte["format"] == "TEST"].reset_index(drop=True)
            va_test = dva[dva["format"] == "TEST"].reset_index(drop=True)

            def winner_probs_for(frame: pd.DataFrame) -> npt.NDArray[np.float64]:
                """Ensemble winner probs for decisive rows of frame (order-preserving)."""
                dec = frame[frame["label"] != 2].reset_index()
                out = np.zeros((len(frame), 2))
                if not len(dec):
                    return out
                Ad = fs.transform(dec)
                lg = model(to_mx(Ad))
                mx.eval(lg)
                mlp_p = softmax(np.array(lg, dtype=np.float64), temp)
                try:
                    with open(artifacts / "hgb.pkl", "rb") as fh:
                        side = pickle.load(fh)
                    hgb_p: npt.NDArray[np.float64] = side.predict_proba(to_hgb_matrix(Ad))
                    w = float(hist.get("w_mlp", 0.5))
                    mlp_p = w * mlp_p + (1 - w) * hgb_p
                except Exception:
                    pass
                for kk, orig in enumerate(dec["index"].values):
                    out[int(orig)] = mlp_p[kk]
                return out

            def three_way_acc(
                frame: pd.DataFrame, draw_p: npt.NDArray[np.float64],
                win_p: npt.NDArray[np.float64], thr: float,
            ) -> float:
                y = frame["label"].values.astype(int)
                pred = np.where(draw_p >= thr, 2, win_p.argmax(axis=1))
                return float((pred == y).mean())

            Xva_test, _ = draw_features(va_test)
            draw_p_val = draw_proba_bundle(draw_bundle, Xva_test)
            win_p_val = winner_probs_for(va_test)
            best_t, best_3way = 0.5, -1.0
            for t in np.arange(0.15, 0.75, 0.05):
                a = three_way_acc(va_test, draw_p_val, win_p_val, float(t))
                if a > best_3way:
                    best_3way, best_t = a, float(t)
            draw_info["threshold_3way"] = round(best_t, 2)
            draw_info["val_3way_acc"] = round(float(best_3way), 4)
            (artifacts / "draw_threshold.json").write_text(
                json.dumps({"threshold": best_t}, indent=2))
            print(f"draw threshold 3-way-tuned: {best_t:.2f} (val 3-way acc={best_3way:.4f})")

            Xte_test, _ = draw_features(te_test)
            draw_p = draw_proba_bundle(draw_bundle, Xte_test)
            win_p_test = winner_probs_for(te_test)
            three_acc = three_way_acc(te_test, draw_p, win_p_test, best_t)
            hist["test_3way_acc"] = round(three_acc, 4)
            hist["draw_model"] = draw_info
            print(f"TEST 3-way (win/loss/draw) test acc={three_acc:.4f} (n={len(te_test)})")
    except Exception as e:
        print("draw model skipped:", e)
    (artifacts / "metrics.json").write_text(json.dumps(hist, indent=2))
    return hist


def evaluate_saved(artifacts: Path = ARTIFACTS_DIR, rebuild: bool = False) -> dict[str, Any]:
    """Score saved weights on the walk-forward test split (no training)."""
    from ..data.hf import load_or_build as _lb
    from ..features import FeatureStore as _FS, add_chrono_features as _ac

    artifacts = Path(artifacts)
    fs = _FS.load(artifacts / "features.json")
    df = _ac(_lb(force=rebuild))
    _, _, te = time_split(df)
    A = fs.transform(te)
    try:
        cfg = json.loads((artifacts / "model_config.json").read_text())
        team_dim, n_classes = int(cfg.get("team_dim", 16)), int(cfg.get("n_classes", 2))
    except Exception:
        team_dim, n_classes = 16, 2
    model = CricketMLP(
        n_teams=len(fs.teams), n_country=len(fs.country), n_city=len(fs.city),
        n_fmt=len(fs.format), n_toss_side=max(4, len(fs.toss_side)),
        n_toss_choice=max(4, len(fs.toss_choice)),
        n_gender=max(3, len(fs.gender)), n_team_type=max(3, len(fs.team_type)),
        n_tier=max(8, len(fs.tier)),
        team_dim=team_dim, n_classes=n_classes)
    model.load_weights(str(artifacts / "model.safetensors"))
    model.eval()
    try:
        temp = float(json.loads((artifacts / "temperature.json").read_text())["temperature"])
    except Exception:
        temp = 1.0
    logit_list: list[npt.NDArray[np.float64]] = []
    for s in range(0, len(te), 1024):
        b = {kk: mx.array(v[s:s + 1024]) for kk, v in A.items()}
        mo = model(b)
        mx.eval(mo)
        logit_list.append(np.array(mo, dtype=np.float64))
    logits = np.concatenate(logit_list)
    probs = softmax(logits, temp)
    # ensemble with saved HGB if present
    try:
        with open(artifacts / "hgb.pkl", "rb") as f:
            clf = pickle.load(f)
        ens_cfg = json.loads((artifacts / "ensemble.json").read_text())
        w_mlp = float(ens_cfg.get("w_mlp", 0.5))
        hgb_proba: npt.NDArray[np.float64] = clf.predict_proba(to_hgb_matrix(A))
        probs = w_mlp * probs + (1 - w_mlp) * hgb_proba
    except Exception:
        pass
    pred = probs.argmax(axis=1)
    y = te["label"].values
    ll, brier = log_loss_and_brier(probs, y)
    out: dict[str, Any] = {"acc": float((pred == np.asarray(y)).mean()), "logloss": ll, "brier": brier,
            "n": len(te), "per_slice": slice_metrics(te, probs, pred)}
    # 3-way TEST (win/loss/draw) with saved draw model
    try:
        df3 = _ac(_lb(force=False, keep_draws=True))
        _, _, dte = time_split(df3)
        te_test = dte[dte["format"] == "TEST"].reset_index(drop=True)
        if len(te_test):
            with open(artifacts / "draw_lr.pkl", "rb") as f:
                draw_bundle = pickle.load(f)
            X3, _ = draw_features(te_test)
            draw_p = draw_proba_bundle(draw_bundle, X3)
            try:
                thr = float(json.loads((artifacts / "draw_threshold.json").read_text())["threshold"])
            except Exception:
                thr = 0.5
            # 3-way: draw where p>=thr else ensemble winner on decisive rows
            three = np.full(len(te_test), 2, dtype=int)
            dec3 = te_test[te_test["label"] != 2].reset_index()
            if len(dec3):
                Ad = fs.transform(dec3)
                lg_list: list[npt.NDArray[np.float64]] = []
                for s in range(0, len(dec3), 1024):
                    bb = {kk: mx.array(v[s:s + 1024]) for kk, v in Ad.items()}
                    oo = model(bb)
                    mx.eval(oo)
                    lg_list.append(np.array(oo, dtype=np.float64))
                win_p = softmax(np.concatenate(lg_list), temp)
                for kk, orig in enumerate(dec3["index"].values):
                    if draw_p[int(orig)] < thr:
                        three[int(orig)] = int(win_p[kk].argmax())
            y3 = te_test["label"].values.astype(int)
            out["test_3way_acc"] = round(float((three == y3).mean()), 4)
            out["test_3way_n"] = len(te_test)
    except Exception:
        pass
    return out


if __name__ == "__main__":
    train()
