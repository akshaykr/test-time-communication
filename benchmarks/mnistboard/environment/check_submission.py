#!/usr/bin/env python3
"""Score your current /app/submission locally. Run this as often as you like.

    python /app/check_submission.py                 # validation split
    python /app/check_submission.py --n 2000        # quicker, on a subset
    python /app/check_submission.py --json          # machine-readable
    python /app/check_submission.py --size-only     # gzip size only, no inference

It reports the EXACT model_bytes you are scored on -- the deterministic gzip -9
byte count of everything in /app/submission -- using the same code the grader
uses, and measures validation accuracy by importing your predict.py exactly the
way the grader will.

IMPORTANT ABOUT THE VALIDATION NUMBER: it is an estimate of the hidden test
score, and it gets optimistic the more you select against it. The default uses
all 5,000 validation images. ``--n`` uses a fresh random subset on each call, so
subset results vary. Leave real margin above 99.4%.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, "/app")
from model_spec import (  # noqa: E402
    SpecError,
    check_layout,
    format_report,
    gzip_size,
)

SUBMISSION_DIR = Path(
    os.environ.get("MNISTBOARD_SUBMISSION_DIR", "/app/submission")
)
VAL_NPZ = Path("/data/val.npz")
PROGRESS_LOG = Path(
    os.environ.get("MNISTBOARD_PROGRESS_LOG", "/logs/agent/mnistboard_progress.jsonl")
)

QUALIFY_ACCURACY = 99.4
INFER_TIMEOUT_SEC = 600

RUNNER = r"""
import importlib.util, sys
import numpy as np
sub, imgs, out = sys.argv[1:4]
sys.path.insert(0, sub)
spec = importlib.util.spec_from_file_location("solver_mod", sub + "/predict.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
images = np.load(imgs)["images"]
model = mod.load() if hasattr(mod, "load") else None
preds = np.empty(len(images), dtype=np.int64)
for i in range(0, len(images), 2000):
    c = images[i:i+2000]
    p = np.asarray(mod.predict(model, c)).reshape(-1)
    assert len(p) == len(c)
    preds[i:i+len(c)] = p.astype(np.int64)
np.save(out, preds)
"""


def run(sub: Path, images, tmp: Path):
    ip, op, rp = (tmp / name for name in ("i.npz", "p.npy", "r.py"))
    np.savez(ip, images=images)
    rp.write_text(RUNNER)
    proc = subprocess.run(
        [sys.executable, str(rp), str(sub), str(ip), str(op)],
        capture_output=True,
        text=True,
        timeout=INFER_TIMEOUT_SEC,
        cwd=str(tmp),
        env={**os.environ, "PYTHONSAFEPATH": "1"},
    )
    return proc, op


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="val images (0 = all)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--size-only", action="store_true")
    args = ap.parse_args()

    try:
        files = check_layout(SUBMISSION_DIR)
        model_bytes = gzip_size(SUBMISSION_DIR)
    except (OSError, SpecError) as exc:
        print(f"INVALID SUBMISSION: {exc}", file=sys.stderr)
        return 1

    if not args.json:
        print(format_report(SUBMISSION_DIR, model_bytes, files))
        print()
    if args.size_only:
        if args.json:
            print(json.dumps({"model_bytes": model_bytes}, indent=2))
        return 0

    d = np.load(VAL_NPZ)
    images, labels = d["images"], d["labels"]
    if args.n < 0 or args.n > len(labels):
        print(
            f"INVALID --n: expected 0..{len(labels)}, got {args.n}",
            file=sys.stderr,
        )
        return 1
    n = args.n or len(labels)
    # With --n, use a fresh random subset instead of encouraging selection
    # against one fixed subset. The default still evaluates all validation data.
    idx = np.random.default_rng(secrets.randbits(32)).permutation(len(labels))[:n]
    images, labels = images[idx], labels[idx]

    with tempfile.TemporaryDirectory() as td:
        try:
            proc, op = run(SUBMISSION_DIR, images, Path(td))
        except subprocess.TimeoutExpired:
            print(
                f"Inference timed out after {INFER_TIMEOUT_SEC}s",
                file=sys.stderr,
            )
            return 2
        except OSError as exc:
            print(f"Inference setup failed: {exc}", file=sys.stderr)
            return 2
        if proc.returncode != 0 or not op.exists():
            print(proc.stdout[-2000:])
            print(proc.stderr[-2000:], file=sys.stderr)
            print("Inference FAILED -- the grader would score this 0.", file=sys.stderr)
            return 2
        try:
            preds = np.asarray(np.load(op, allow_pickle=False)).reshape(-1)
        except (OSError, ValueError) as exc:
            print(f"Invalid predictions file: {exc}", file=sys.stderr)
            return 2

    if len(preds) != len(labels):
        print(
            f"Invalid predictions: len={len(preds)} != {len(labels)}",
            file=sys.stderr,
        )
        return 2
    if preds.min() < 0 or preds.max() > 9:
        print(
            f"Invalid predictions: range=[{preds.min()},{preds.max()}]",
            file=sys.stderr,
        )
        return 2
    preds = preds.astype(np.int64)
    accuracy = 100.0 * float((preds == labels).sum()) / len(labels)
    out = {
        "model_bytes": model_bytes,
        "num_files": len(files),
        "val_accuracy": round(accuracy, 4),
        "val_images": int(len(labels)),
        "qualified_on_val": accuracy >= QUALIFY_ACCURACY,
    }

    try:
        progress = {**out,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_id": os.environ.get("AGENT_ID", "unknown"),
        }
        PROGRESS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with PROGRESS_LOG.open("a") as fh:
            fh.write(json.dumps(progress, sort_keys=True) + "\n")
    except OSError as exc:
        print(f"PROGRESS LOG WRITE FAILED: {exc}", file=sys.stderr)

    if args.json:
        print(json.dumps(out, indent=2))
        return 0

    for k, v in out.items():
        print(f"{k:<22}: {v}")
    print()
    if accuracy < QUALIFY_ACCURACY:
        print(f"Below {QUALIFY_ACCURACY}% -- accuracy is the gate; fix that first. "
              "This validation number is also optimistic vs the hidden test.")
    else:
        print(f"Qualified at {accuracy:.2f}%. Keep shrinking from {model_bytes} B; "
              "among qualifying submissions, every byte counts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
