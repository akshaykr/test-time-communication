#!/usr/bin/env bash
# Print your mnistboard score: the deterministic gzip -9 byte count of the WHOLE
# submission directory (default /app/submission) -- inference code, weights,
# tables and all. This calls the same canonical scorer the grader and
# check_submission.py use, so the number it prints is exactly what you are
# scored on. score.sh itself lives OUTSIDE the submission and is never counted.
#
#   bash /app/score.sh                 # score /app/submission
#   bash /app/score.sh /path/to/dir    # score another directory
set -euo pipefail
SUBMISSION="${1:-/app/submission}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 - "$HERE" "$SUBMISSION" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from model_spec import gzip_size
print(gzip_size(sys.argv[2]))
PY
