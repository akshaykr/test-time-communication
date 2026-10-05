# MNIST smallest solver (mnistboard)

## Goal

Build a self-contained MNIST classifier. It must reach at least **99.4%**
official MNIST test accuracy. Among all artifacts that clear the accuracy bar,
optimize **ONLY compressed size**. Accuracy above the target earns nothing.

Secondary stretch goal: get as close as possible to roughly 2x the compressed
information content of the MNIST train labels, while preserving legitimate test
accuracy.

## The artifact

Build your solver in **`/app/submission/`** — the directory that is snapshotted
and graded. It must contain everything needed to classify one 28x28 grayscale
MNIST image:

- inference code
- weights
- fixed tables, filters, constants, generated code, or deterministic generators

`/app/submission/` must contain a `predict.py` exposing:

```python
from pathlib import Path
HERE = Path(__file__).resolve().parent

def load():                      # optional; called once before predict()
    # Read your weights/tables from files next to this one, e.g.
    #   weights = np.load(HERE / "weights.npy")
    return model                 # any object your predict() understands

def predict(model, images):      # images: uint8 (N, 28, 28), 0..255
    # Return int labels (N,) in 0..9. Must accept any N >= 1. CPU or CUDA.
    return labels
```

The grader imports `predict.py`, calls `load()` once (if defined), then
`predict` in batches. Your code loads its **own** weight files from the
submission directory — there is no manifest and nothing is unpacked for you. You
may split the solver over several files (a package, a weights blob, a codebook);
the grader imports only `predict.py`, which must pull in everything else. A
template is at `/app/submission_template.py`.

**Save a working solver early and update it whenever you improve.** Whatever is
in `/app/submission/` when your run ends is what gets scored, including on a
timeout.

## Score = deterministic compressed byte count

```bash
bash /app/score.sh        # prints your model_bytes
```

`score.sh` builds a deterministic tar of `/app/submission/` and `gzip -9`s it;
`model_bytes` is that byte count.

- **Counted:** every file under `/app/submission/` — inference code, weights,
  tables, constants, and any compression/decompression logic you ship there.
  A byte is charged wherever it lives, so use whichever representation
  compresses smaller.
- **Not counted:** the Python / CUDA / numpy / torch runtime, any training code
  you keep OUTSIDE `/app/submission/`, and the task tools `/app/score.sh` and
  `/app/check_submission.py`.

Smaller always scores higher, with no floor — remove every byte that does not
buy needed accuracy.

## Check your score

```bash
python /app/check_submission.py            # size + validation accuracy
python /app/check_submission.py --size-only
python /app/check_submission.py --json
python /app/check_test.py                  # requires val >=99.4%; returns test accuracy
```

It uses the same size code the grader uses, so the `model_bytes` it prints is
your score, and it imports your `predict.py` exactly the way the grader will.

## Save every new qualifying frontier

You are responsible for saving frontiers; the checker does not do it
automatically. Whenever a default full check reports all three of:

- `val_images: 5000`
- `qualified_on_val: true`
- `model_bytes` strictly smaller than every frontier already saved

Checkpoint qualification and sealed-test eligibility both use the 99.4%
validation gate.

stop changing `/app/submission` and immediately save that candidate:

```bash
python /app/check_submission.py --json | tee /tmp/mnistboard-full-check.json
# Read model_bytes from the JSON, then:
stamp=$(date -u +%Y%m%dT%H%M%SZ)
dest=/logs/artifacts/mnistboard_frontier/<model_bytes>-$stamp-${AGENT_ID:-agent}
mkdir -p "$dest"
cp -a /app/submission "$dest/submission"
cp /tmp/mnistboard-full-check.json "$dest/result.json"
bash /app/score.sh "$dest/submission" | tee "$dest/model_bytes.txt"
saved_bytes=$(cat "$dest/model_bytes.txt")
result_bytes=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["model_bytes"])' "$dest/result.json")
test "$saved_bytes" = "$result_bytes"
test -f "$dest/submission/predict.py"
```

Replace `<model_bytes>` with the reported integer. The final commands verify
that the copied submission's size equals the `model_bytes` field in
`result.json` and that its entry point exists. Also write `provenance.txt`
naming the source checkpoint/work directory, parent candidate, and idea that
produced it. Subset and size-only checks never qualify. In team runs,
coordinate so only one agent checks and saves `/app/submission` at a time.
Never overwrite or delete a saved frontier.

## Hard rules

1. Train models on all 55,000 examples in `/data/train.npz`. Use
   `/data/val.npz` as the development set for evaluation, tuning, and model
   selection; do not include its examples in gradient updates or teacher
   training.
2. Do not inspect individual test images or test errors. Online inspection is cheating.
3. Do not use the network for downloading additional data or models.
4. No pretrained MNIST weights. If using a teacher, train it only from
   `/data/train.npz` and never ship it.
5. Test data is sealed. After full validation reaches 99.4%,
   `check_test.py` reports aggregate accuracy on all 10,000 test images. It
   never exposes individual images, predictions, labels, or errors.

## Data discipline

- `/data/train.npz` — 55,000 labeled training images; train on all of them
- `/data/val.npz` — 5,000 labeled development/validation images for evaluation,
  tuning, and model selection

Each has `images`: uint8 `(N, 28, 28)`, and `labels`: int64 `(N,)` in 0..9.

```python
import numpy as np
train = np.load("/data/train.npz")
val = np.load("/data/val.npz")
x_train, y_train = train["images"], train["labels"]
x_dev, y_dev = val["images"], val["labels"]
```

Both come from the official 60k MNIST training set. The official **10,000-image
test set is hidden** and used only by `check_test.py` and the final grader.

## Compute: shared GPU, 1,000 MiB limit per agent

- One shared NVIDIA A100 is available for training, train-derived evaluation,
  and final inference. CUDA-enabled PyTorch is installed; CPU solvers are also
  valid.
- **HARD LIMIT: each agent may use at most 1,000 MiB of aggregate GPU memory at
  any instant.** This is 1,000 x 1,024 x 1,024 bytes total across every CUDA
  process launched by that agent, including its submitted solver, not 1,000 MiB
  per process.
- **Using more than 1,000 MiB is CHEATING.**
- Use only one CUDA training process at a time and stream mini-batches rather
  than copying the full dataset to the GPU. CUDA context/library memory counts
  toward the 1,000 MiB.
- Leave allocator headroom. The environment exports
  `$MNISTBOARD_GPU_ALLOCATOR_MIB=400`; apply it once in each CUDA process:

  ```python
  import os
  total = torch.cuda.get_device_properties(0).total_memory
  allocator_mib = int(os.environ["MNISTBOARD_GPU_ALLOCATOR_MIB"])
  torch.cuda.set_per_process_memory_fraction(
      allocator_mib * 1024**2 / total, 0
  )
  ```

  The 400 MiB allocator cap leaves the rest of the 1,000 MiB budget for the
  CUDA context and libraries. Use `nvidia-smi` and reduce batch size if total
  process memory approaches the limit.
- Do not alter or unset `AGENT_ID` to disguise ownership, attribute work to
  another agent, or use multiple processes to obtain more than 1,000 MiB.
  Those are **CHEATING**.
- `predict()` may use CUDA or CPU. If it uses CUDA, return predictions to CPU
  (for example, `labels.cpu().numpy()`) and obey the same 1,000 MiB total limit.
- The container is hard-capped at `$MNISTBOARD_CPUS` cores; thread pools are
  already sized to your share (`$OMP_NUM_THREADS`). Raising them measurably
  makes things *slower*. If several agents share this container
  (`$MNISTBOARD_AGENTS` > 1) the pools are already divided — do not take more
  than your share.

## Environment

A pinned stack is preinstalled: Python 3.12.13, CUDA 12.4,
`torch==2.5.1+cu124`, `torchvision==0.20.1+cu124`, `numpy==2.5.1`,
`scipy==1.18.0`, `scikit-learn==1.9.0`, `pillow==11.3.0`, and
`tqdm==4.70.0`. Do not install packages.

**Train from scratch, here, on `/data`.** Every parameter you ship must come
from training you ran yourself in this container. Do **not** use any training
data other than `/data`.

## Logging

Maintain an append-only `IDEAS.md` (keep it OUTSIDE `/app/submission/` so it is
not counted).

An IDEA is one distinct conceptual intervention: architecture, compression
format, quantization scheme, generated feature family, training scheme,
decomposition, coding method, etc. Hyperparameter nudges are sub-moves, not new
ideas.

For every idea, append:

| idx | time stamp | one-line description | dev_acc before→after | gzip_bytes before→after | cum_wallclock_min | kept? | one-line why |

The log must be enough to plot:
- dev_acc vs idx
- dev_acc vs wallclock
- test_acc vs idx for sealed-test candidates

## Loop

1. First get dev accuracy comfortably above the target.
2. Shrink aggressively while staying above target on the development set.
3. Repeat: distill, prune, quantize, code-golf, share weights, generate tables,
   use low-rank or symbolic structure, exploit MNIST geometry, compress weights
   and labels, remove every byte that does not buy needed accuracy.
4. Only sealed-test candidates that are small and reach at least 99.4% on the
   full development set.

## Methods to try

Be creative. You are not limited to the methods below:
- tiny CNNs, MLPs, polynomial features, template matching, stroke geometry
- deskewing, centering, elastic/stroke-aware preprocessing if byte-cheap
- distillation from a large train-only teacher
- pruning, ternary/binary weights, int2/int1, shared scales
- low-rank factors, generated filters, DCT/Haar/random bases, looping weight sharing
- nearest prototypes, class centroids, learned masks
- arithmetic/range coding, bit packing, generated lookup tables
- cascades: tiny cheap classifier plus small fallback for ambiguous digits
You may search and adopt relevant advanced ML techniques or papers online if it would be helpful.

## What happens at grading time

1. `/app/submission/` is snapshotted read-only; `predict.py` must be present.
2. Your `model_bytes` is the deterministic gzip size of that snapshot.
3. `load()` is called once, then `predict` in batches over the 10,000 hidden
   test images in a privately shuffled order (fresh random seed each run).
4. Inference runs unprivileged, with no access to `/data` or `/tests`, but with
   the shared A100 available, in **600 seconds** total. CPU inference is also
   supported.

## How you are scored

- You **qualify** at `>= 99.4%` on the hidden test set.
- Among qualifying submissions, lower `model_bytes` ranks higher. Accuracy
  above the bar earns nothing, and there is no byte threshold.

## Deliverables

- `/app/submission/` — the final solver (`predict.py` + weights + tables)
- `IDEAS.md` (kept outside the submission)
- A concise explanation of the final model and compression method
- Final validation accuracy, gzipped bytes, and sealed-test accuracy

Do not stop until you either:
- reach the 99.4% test target with the smallest artifact you can make,
  preferably near the label-compression scale and ideally below 2 KB.