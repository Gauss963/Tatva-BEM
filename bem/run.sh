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
export OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg
if [ "${1:-}" = test ]; then
    shift
    exec "$FDBEM_PYTHON" -B -m pytest "$FDBEM_ROOT/tests" "$@"
elif [ "${1:-}" = plot ]; then
    shift
    exec "$FDBEM_PYTHON" -B "$FDBEM_ROOT/tools/plot_results.py" "$@"
fi
exec "$FDBEM_PYTHON" -B -m finiteblock_bem.cli "$@"
