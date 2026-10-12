"""Hub helpers: download/upload pretrained artifacts from Hugging Face.

Users should never need `mlx-cric train` — `Predictor` calls
`ensure_artifacts()` automatically, and `mlx-cric download-model`
does the same explicitly.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from .config import ARTIFACTS_DIR, HF_MODEL_REPO

# Everything Predictor reads. Excludes stale draw_hgb.pkl (unused leftover).
REQUIRED_FILES = [
    "features.json",
    "model_config.json",
    "temperature.json",
    "ensemble.json",
    "draw_threshold.json",
    "margin_stats.json",
    "metrics.json",
    "hgb.pkl",
    "stacker.pkl",
    "draw_lr.pkl",
    "pool.json",
    "ratings.json",
    "model.safetensors",
    "model_0.safetensors",
    "model_1.safetensors",
    "model_2.safetensors",
    "model_3.safetensors",
    "model_4.safetensors",
]


def artifacts_complete(dest: Path = ARTIFACTS_DIR) -> bool:
    dest = Path(dest)
    return all((dest / f).exists() for f in REQUIRED_FILES)


def download_artifacts(
    repo_id: str | None = None,
    revision: str = "main",
    dest: Path = ARTIFACTS_DIR,
) -> Path:
    """Snapshot-download weights from HF Hub into local artifacts dir."""
    from huggingface_hub import snapshot_download

    repo_id = repo_id or HF_MODEL_REPO
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    snap = Path(snapshot_download(repo_id=repo_id, revision=revision, repo_type="model"))
    copied = 0
    for f in REQUIRED_FILES:
        src = snap / f
        if src.exists():
            shutil.copy2(src, dest / f)
            copied += 1
    # also pick up any extra safetensors / json / pkl the repo may have
    for src in snap.glob("*"):
        if src.name in (".gitattributes"):
            continue
        tgt = dest / src.name
        if src.is_file() and not tgt.exists():
            shutil.copy2(src, tgt)
            copied += 1
    if not (dest / "features.json").exists():
        raise RuntimeError(
            f"Downloaded {repo_id} but features.json is missing "
            f"(copied {copied} files from {snap}). Is the repo ID right?"
        )
    return dest


def ensure_artifacts(
    dest: Path = ARTIFACTS_DIR,
    repo_id: str | None = None,
) -> Path:
    """Return dest, downloading from HF Hub first if weights are missing."""
    dest = Path(dest)
    if artifacts_complete(dest):
        return dest
    # allow explicit override, else the default hub repo
    return download_artifacts(repo_id=repo_id, dest=dest)


def upload_artifacts(
    repo_id: str | None = None,
    src: Path = ARTIFACTS_DIR,
    private: bool = False,
) -> str:
    """Push local artifacts/ to a HF model repo. Returns the repo URL."""
    from huggingface_hub import create_repo, upload_folder

    repo_id = repo_id or HF_MODEL_REPO
    src = Path(src)
    if not (src / "features.json").exists():
        raise RuntimeError(f"{src}/features.json missing — train first.")
    if not (src / "README.md").exists():
        raise RuntimeError(f"{src}/README.md missing.")
    create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(src),
        allow_patterns=["*.json", "*.pkl", "*.safetensors", "README.md"],
        delete_patterns=["draw_hgb.pkl"],
    )
    return f"https://huggingface.co/{repo_id}"
