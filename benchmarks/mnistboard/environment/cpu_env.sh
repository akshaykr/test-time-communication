#!/usr/bin/env bash
# Size the BLAS/OpenMP thread pools to the CPU quota this container was
# actually given, instead of hardcoding a number that silently drifts from
# task.toml (or from a --override-cpus launch).
#
# A mismatch is not harmless: 6 torch threads inside a 4-CPU cgroup measured
# ~20% SLOWER than a matched 4/4 setup, because the threads thrash against the
# quota. mnistboard is launched at several different core counts, so the only
# safe source of truth is the cgroup itself.
#
# When several agents share one container (MNISTBOARD_AGENTS > 1) each agent
# gets an equal share, so N concurrent trainings do not oversubscribe together.

_cpus_from_cgroup() {
  # cgroup v2
  if [ -r /sys/fs/cgroup/cpu.max ]; then
    read -r quota period < /sys/fs/cgroup/cpu.max
    if [ "$quota" != "max" ] && [ -n "$period" ] && [ "$period" -gt 0 ] 2>/dev/null; then
      echo $(( (quota + period - 1) / period ))
      return
    fi
  fi
  # Legacy cgroup hierarchy.
  if [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ] && [ -r /sys/fs/cgroup/cpu/cpu.cfs_period_us ]; then
    quota=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)
    period=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
    if [ "$quota" -gt 0 ] 2>/dev/null && [ "$period" -gt 0 ] 2>/dev/null; then
      echo $(( (quota + period - 1) / period ))
      return
    fi
  fi
  nproc 2>/dev/null || echo 1
}

MNISTBOARD_CPUS="$(_cpus_from_cgroup)"
[ "$MNISTBOARD_CPUS" -ge 1 ] 2>/dev/null || MNISTBOARD_CPUS=1

AGENTS="${MNISTBOARD_AGENTS:-1}"
[ "$AGENTS" -ge 1 ] 2>/dev/null || AGENTS=1

THREADS=$(( MNISTBOARD_CPUS / AGENTS ))
[ "$THREADS" -ge 1 ] || THREADS=1

export MNISTBOARD_CPUS MNISTBOARD_AGENTS="$AGENTS" MNISTBOARD_THREADS="$THREADS"
export OMP_NUM_THREADS="$THREADS" \
       MKL_NUM_THREADS="$THREADS" \
       OPENBLAS_NUM_THREADS="$THREADS" \
       NUMEXPR_NUM_THREADS="$THREADS" \
       TORCH_NUM_THREADS="$THREADS"

# Only act as an exec-wrapper when RUN as a script; when sourced (BASH_ENV,
# profile.d) "$@" belongs to the caller and must not be exec'd.
(return 0 2>/dev/null) && _mnistboard_sourced=1 || _mnistboard_sourced=0
if [ "$_mnistboard_sourced" -eq 0 ] && [ "$#" -gt 0 ]; then
  exec "$@"
fi
unset _mnistboard_sourced
