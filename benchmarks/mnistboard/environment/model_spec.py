"""Model size accounting for mnistboard -- deterministic gzip of the solver.

The scored artifact is the WHOLE submission directory (``/app/submission``): its
inference code, weights, tables, generators -- everything it ships. Size is the
deterministic gzip byte count of that directory::

    model_bytes = len( gzip -9 of a normalised tar of every regular file )

"Normalised" means the archive depends ONLY on each file's relative path and
its bytes: names are sorted and every other tar field (mtime, owner, mode,
order) is pinned. So the number is identical on the agent's live
``/app/submission`` and on the grader's read-only snapshot, and identical across
``score.sh``, ``check_submission.py`` and the grader -- there is exactly one
answer to "how big is this model".

Code and data are both counted: wherever a byte lives in the submission, it is
charged. The grader separately protects the sealed test through process
isolation and a read-only submission snapshot.
"""

from __future__ import annotations

import gzip
import io
import os
import stat
import tarfile
from pathlib import Path

#: The one file the grader imports for inference. Everything else in the
#: submission is free-form (weights, tables, helper modules, generators).
ENTRYPOINT = "predict.py"

#: Sanity limits (not a metric -- just to keep the scorer well-behaved).
MAX_FILES = 512
MAX_NAME_LEN = 100  # USTAR name field; keeps the archive trivially deterministic
MAX_UNCOMPRESSED_BYTES = 256 * 1024 * 1024


class SpecError(RuntimeError):
    """The submission is not a well-formed solver directory."""


def list_files(submission_dir: str | os.PathLike[str]) -> list[tuple[str, int]]:
    """Return sorted (relpath, size) for every regular file, rejecting junk.

    Symlinks, empty directories, ``__pycache__`` and ``.pyc`` are rejected.
    Empty directory names would otherwise be usable as uncounted data;
    bytecode is non-deterministic across interpreters; and symlinks could point
    outside the payload.
    """
    root = Path(submission_dir)
    files: list[tuple[str, int]] = []
    total = 0
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        if Path(dirpath) != root and not dirnames and not filenames:
            raise SpecError(f"empty directory is not allowed: {dirpath}")
        for d in list(dirnames):
            full = os.path.join(dirpath, d)
            if os.path.islink(full):
                raise SpecError(f"symlinked directory: {full}")
            if d == "__pycache__":
                raise SpecError("__pycache__ is not allowed in the submission")
        for name in sorted(filenames):
            src = Path(dirpath) / name
            st = os.lstat(src)
            if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
                raise SpecError(f"not a regular file: {src}")
            if src.suffix == ".pyc":
                raise SpecError(f"bytecode file not allowed: {src}")
            rel = str(src.relative_to(root))
            if len(rel.encode()) > MAX_NAME_LEN:
                raise SpecError(f"path too long (>{MAX_NAME_LEN} bytes): {rel}")
            files.append((rel, st.st_size))
            total += st.st_size
            if total > MAX_UNCOMPRESSED_BYTES:
                raise SpecError("submission exceeds the size sanity limit")
            if len(files) > MAX_FILES:
                raise SpecError(f"more than {MAX_FILES} files")
    files.sort()
    return files


def check_layout(submission_dir: str | os.PathLike[str]) -> list[tuple[str, int]]:
    """Validate the layout and that the inference entry point is present."""
    files = list_files(submission_dir)
    if not files:
        raise SpecError("submission directory is empty")
    names = {rel for rel, _ in files}
    if ENTRYPOINT not in names:
        raise SpecError(
            f"missing {ENTRYPOINT} -- it must define predict(model, images) and "
            "may define load()"
        )
    return files


def gzip_archive(submission_dir: str | os.PathLike[str]) -> bytes:
    """Return the canonical compressed archive of the submission directory.

    Built from a normalised in-memory tar (sorted names; mtime/owner/mode all
    pinned) so the result depends only on file paths and contents -- never on
    disk timestamps, ownership or permissions.
    """
    root = Path(submission_dir)
    files = list_files(submission_dir)
    buf = io.BytesIO()
    total = 0
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for rel, expected_size in files:
            with (root / rel).open("rb") as source:
                data = source.read(expected_size + 1)
            if len(data) != expected_size:
                raise SpecError(f"file changed while archiving: {rel}")
            total += len(data)
            if total > MAX_UNCOMPRESSED_BYTES:
                raise SpecError("submission exceeds the size sanity limit")
            info = tarfile.TarInfo(name=rel)
            info.size = len(data)
            info.mtime = 0
            info.mode = 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.type = tarfile.REGTYPE
            tar.addfile(info, io.BytesIO(data))
    return gzip.compress(buf.getvalue(), compresslevel=9, mtime=0)


def gzip_size(submission_dir: str | os.PathLike[str]) -> int:
    """Deterministic gzip byte count of the whole submission directory.

    This is the canonical scorer used by ``score.sh``, ``check_submission.py``
    and the trusted grader alike.
    """
    return len(gzip_archive(submission_dir))


def format_report(
    submission_dir: str | os.PathLike[str],
    model_bytes: int,
    files: list[tuple[str, int]] | None = None,
) -> str:
    if files is None:
        files = list_files(submission_dir)
    lines = [
        f"model_bytes : {model_bytes}  <-- THIS IS YOUR SCORE "
        "(gzip -9 of the whole submission)",
        f"files ({len(files)}, all counted):",
    ]
    for rel, size in files:
        lines.append(f"  {rel:<32} {size:>10} B")
    return "\n".join(lines)
