#!/usr/bin/env python3
"""Isolated inference runner for the mnistboard verifier (gzip-of-solver).

TRUSTED code (lives in tests/, not in the submission). It runs in a separate,
unprivileged process so submission code only ever receives images -- never the
labels, the reward file, or the /tests /data directories.

The submission ships its OWN weight files inside its directory and loads them
itself; there is no manifest and nothing is unpacked for it. This runner just
imports ``predict.py``, calls ``load()`` once (if defined), then ``predict`` in
batches.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

BATCH = 2000


def load_solver(sub_dir: Path):
    entry = sub_dir / "predict.py"
    if not entry.is_file():
        raise FileNotFoundError("submission/predict.py not found")
    sys.path.insert(0, str(sub_dir))
    spec = importlib.util.spec_from_file_location("graded_solver", entry)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load predict.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "predict"):
        raise AttributeError("predict.py must define predict(model, images)")
    return module


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    images = np.load(args.images)["images"]
    if images.dtype != np.uint8 or images.ndim != 3 or images.shape[1:] != (28, 28):
        raise ValueError(f"unexpected images array: {images.shape} {images.dtype}")

    module = load_solver(args.submission)
    model = module.load() if hasattr(module, "load") else None

    preds = np.empty(len(images), dtype=np.int64)
    for start in range(0, len(images), BATCH):
        chunk = images[start : start + BATCH]
        out = np.asarray(module.predict(model, chunk)).reshape(-1)
        if len(out) != len(chunk):
            raise ValueError(
                f"predict() returned {len(out)} labels for {len(chunk)} images"
            )
        preds[start : start + len(chunk)] = out.astype(np.int64)

    np.save(args.out, preds)
    print(f"inference OK: {len(preds)} predictions", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
