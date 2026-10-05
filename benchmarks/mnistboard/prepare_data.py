#!/usr/bin/env python3
"""Fetch MNIST and stage it for the mnistboard task (run this on the HOST).

Produces two gitignored artifacts:

  environment/mnist-raw/*.gz   the four canonical IDX archives. This is the
                               Docker build context input; the image's builder
                               stage turns the 60k train archive into
                               /data/{train,val}.npz and never writes the test
                               split.
  tests/mnist_test.npz         the held-out 10k test split used by the grader.
                               Harbor copies tests/ into the container only at
                               verify time, after the agent has terminated.

Usage:
    python3 prepare_data.py            # download (or reuse) + write both
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "environment" / "mnist-raw"
TEST_NPZ = HERE / "tests" / "mnist_test.npz"

MIRRORS = (
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
)

# Canonical md5s of the four IDX archives.
FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def download(name: str, dest: Path) -> None:
    last_error: Exception | None = None
    for base in MIRRORS:
        url = base + name
        try:
            print(f"  fetching {url}", flush=True)
            with urllib.request.urlopen(url, timeout=120) as resp:
                dest.write_bytes(resp.read())
            return
        except Exception as exc:  # noqa: BLE001 - try the next mirror
            last_error = exc
            print(f"  failed ({exc}); trying next mirror", flush=True)
    raise RuntimeError(f"could not download {name}: {last_error}")


def ensure_raw() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, want in FILES.items():
        dest = RAW_DIR / name
        if dest.is_file() and md5(dest) == want:
            print(f"  ok (cached) {name}")
            continue
        download(name, dest)
        got = md5(dest)
        if got != want:
            raise RuntimeError(f"md5 mismatch for {name}: {got} != {want}")
        print(f"  ok {name}")


def read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as fh:
        magic, count, rows, cols = struct.unpack(">IIII", fh.read(16))
        if magic != 2051:
            raise ValueError(f"{path.name}: bad image magic {magic}")
        buf = fh.read(count * rows * cols)
    return np.frombuffer(buf, dtype=np.uint8).reshape(count, rows, cols).copy()


def read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as fh:
        magic, count = struct.unpack(">II", fh.read(8))
        if magic != 2049:
            raise ValueError(f"{path.name}: bad label magic {magic}")
        buf = fh.read(count)
    return np.frombuffer(buf, dtype=np.uint8).astype(np.int64)


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()

    print("Staging MNIST archives ...")
    ensure_raw()

    test_x = read_idx_images(RAW_DIR / "t10k-images-idx3-ubyte.gz")
    test_y = read_idx_labels(RAW_DIR / "t10k-labels-idx1-ubyte.gz")
    if test_x.shape != (10000, 28, 28) or test_y.shape != (10000,):
        raise RuntimeError(f"unexpected test shapes {test_x.shape} {test_y.shape}")
    TEST_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(TEST_NPZ, images=test_x, labels=test_y)
    print(f"Wrote {TEST_NPZ} ({len(test_y)} images)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
