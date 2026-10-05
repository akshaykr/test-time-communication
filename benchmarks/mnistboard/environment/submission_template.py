"""Template solver for mnistboard (smallest gzipped solver at >= 99.4% on MNIST).

Copy this to /app/submission/predict.py and build your solver in
/app/submission/. EVERYTHING under /app/submission/ is your score:

    model_bytes = deterministic gzip -9 byte count of the whole directory

i.e. your inference code, weights, tables and generators are ALL counted. Move
complexity to whichever representation compresses smaller. Run
`bash /app/score.sh` for the exact number and
`python /app/check_submission.py` for size + validation accuracy.

You may split your solver across several files in /app/submission/ (a package,
helper modules, a weights file, a codebook, ...). The grader imports ONLY
predict.py; it must, directly or indirectly, load everything it needs from files
that live next to it in the submission directory.

Ship your weights as files in /app/submission/ and load them here. A compact,
gzip-friendly layout usually wins: raw bit-packed bytes, an entropy-coded blob
you decode in load(), a small .npy, etc. Locate them relative to this file:

    from pathlib import Path
    HERE = Path(__file__).resolve().parent

Inference has access to the shared A100 and has a 600 s budget over 10,000
images. CPU solvers remain valid. If predict() uses CUDA, it must stay under
the same 1,000 MiB aggregate VRAM limit as training.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load():
    # One-time setup, called once before predict(). Optional -- delete it and
    # the grader will pass model=None. Read your weight/table files from HERE.
    #   weights = np.load(HERE / "weights.npy")
    #   return {"weights": weights, ...}
    raise NotImplementedError("Load your weights/tables from files in this directory.")


def predict(model, images):
    # images: numpy uint8 (N, 28, 28), values 0..255. Handle any N >= 1.
    # Return int labels (N,) in [0, 9]. Must be a generic forward pass -- do not
    # hardcode answers or try to read the test data.
    raise NotImplementedError("Implement a generic forward pass here.")
