#!/bin/sh
set -eu
FDBEM_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ -z "${FDBEM_PYTHON:-}" ]; then
    if [ -x "$FDBEM_ROOT/.venv/bin/python" ]; then
        FDBEM_PYTHON="$FDBEM_ROOT/.venv/bin/python"
    elif [ -x /opt/homebrew/Caskroom/miniconda/base/envs/tatva/bin/python ]; then
        FDBEM_PYTHON=/opt/homebrew/Caskroom/miniconda/base/envs/tatva/bin/python
    else
        FDBEM_PYTHON=python3
    fi
fi
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$FDBEM_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export FDBEM_CORES=${FDBEM_CORES:-1}
export OPENBLAS_NUM_THREADS=$FDBEM_CORES
export OMP_NUM_THREADS=$FDBEM_CORES
export MKL_NUM_THREADS=$FDBEM_CORES
export VECLIB_MAXIMUM_THREADS=$FDBEM_CORES
export TF_NUM_INTRAOP_THREADS=$FDBEM_CORES
export TF_NUM_INTEROP_THREADS=1
export JAX_ENABLE_X64=true
export XLA_FLAGS="${XLA_FLAGS:---xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=$FDBEM_CORES}"
export MPLBACKEND=Agg
if [ "${1:-}" = test ]; then
    shift
    exec "$FDBEM_PYTHON" -B -m pytest "$FDBEM_ROOT/tests" "$@"
elif [ "${1:-}" = plot ]; then
    shift
    exec "$FDBEM_PYTHON" -B "$FDBEM_ROOT/tools/plot_results.py" "$@"
elif [ "${1:-}" = ad-benchmark ]; then
    shift
    exec "$FDBEM_PYTHON" -B "$FDBEM_ROOT/tools/benchmark_ad.py" "$@"
elif [ "${1:-}" = static-refinement ]; then
    shift
    exec "$FDBEM_PYTHON" -B "$FDBEM_ROOT/tools/static_refinement.py" "$@"
elif [ "${1:-}" = compare ]; then
    shift
    exec "$FDBEM_PYTHON" -B "$FDBEM_ROOT/tools/compare_event16.py" "$@"
fi
exec "$FDBEM_PYTHON" -B -m finiteblock_bem.cli "$@"
