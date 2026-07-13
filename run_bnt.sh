#!/usr/bin/env bash
set -euo pipefail

mode="${1:-}"
workers="${BNT_MPI_WORKERS:-8}"

if [[ ! "$mode" =~ ^[0-9]{6}$ ]]; then
    echo "Usage: $0 MODE" >&2
    echo "MODE must have six digits, for example: 000000" >&2
    exit 2
fi

exec mpiexec -n "$workers" python -m mpi4py.futures -m bnt_nautilus.sampler "$mode"
