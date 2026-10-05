#!/usr/bin/env python3
"""Build /data/train.npz and /data/val.npz inside the Docker image.

Runs at IMAGE BUILD TIME inside the *builder* stage. Reads the locally-provided
MNIST IDX archives (COPYed into the build context) so the build needs no
network. Writes ONLY the train (55k) and val (5k) splits, both carved out of
the 60k official training set. The official 10k test split is never read here,
and this whole builder stage is discarded from the final image, so the test
data cannot land in an agent-visible layer.

Split seed matches prepare_data.py exactly.
"""

from __future__ import annotations

import gzip
import hashlib
import struct
import sys
from pathlib import Path

import numpy as np

RAW_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/build/mnist-raw")
OUT_DIR = Path("/data")
SPLIT_SEED = 1234
VAL_SIZE = 5000

IMAGES_GZ = "train-images-idx3-ubyte.gz"
LABELS_GZ = "train-labels-idx1-ubyte.gz"
IMAGES_MD5 = "f68b3c2dcbeaaa9fbdd348bbdeb94873"
LABELS_MD5 = "d53e105ee54ea40749a09fcbcd1e9432"


def check_md5(path: Path, want: str) -> None:
    got = hashlib.md5(path.read_bytes()).hexdigest()
    if got != want:
        raise SystemExit(f"ERROR: md5 mismatch for {path.name}: {got} != {want}")


def read_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as fh:
        magic, count, rows, cols = struct.unpack(">IIII", fh.read(16))
        if magic != 2051:
            raise SystemExit(f"bad image magic {magic}")
        buf = fh.read(count * rows * cols)
    return np.frombuffer(buf, dtype=np.uint8).reshape(count, rows, cols).copy()


def read_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as fh:
        magic, count = struct.unpack(">II", fh.read(8))
        if magic != 2049:
            raise SystemExit(f"bad label magic {magic}")
        buf = fh.read(count)
    return np.frombuffer(buf, dtype=np.uint8).astype(np.int64)


def main() -> int:
    images_path = RAW_DIR / IMAGES_GZ
    labels_path = RAW_DIR / LABELS_GZ
    check_md5(images_path, IMAGES_MD5)
    check_md5(labels_path, LABELS_MD5)

    x = read_images(images_path)
    y = read_labels(labels_path)
    if x.shape != (60000, 28, 28) or y.shape != (60000,):
        raise SystemExit(f"unexpected train shapes {x.shape} {y.shape}")

    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(len(x))
    val_idx, train_idx = perm[:VAL_SIZE], perm[VAL_SIZE:]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_DIR / "train.npz", images=x[train_idx], labels=y[train_idx])
    np.savez(OUT_DIR / "val.npz", images=x[val_idx], labels=y[val_idx])
    print(f"Wrote {OUT_DIR}/train.npz ({len(train_idx)}) and "
          f"{OUT_DIR}/val.npz ({len(val_idx)})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
