#!/bin/sh
set -eu
BEM_HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export MPLBACKEND=Agg
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=4
export JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH:-"$BEM_HERE/.julia-depot"}
BEM_JULIA=${BEM_JULIA:-$(command -v julia || true)}
BEM_PYTHON=${BEM_PYTHON:-python3}
BEM_ACTION=${1:-replay}
if [ "$#" -gt 0 ]; then shift; fi
case "$BEM_ACTION" in
    export) exec "$BEM_PYTHON" -B "$BEM_HERE/export_reference.py" "$@" ;;
    plot) exec "$BEM_PYTHON" -B "$BEM_HERE/plot_results.py" "$@" ;;
    verify) exec "$BEM_PYTHON" -B "$BEM_HERE/verify_reference.py" "$@" ;;
esac
if [ -z "$BEM_JULIA" ]; then
    echo 'Set BEM_JULIA to a Julia 1.10 executable.' >&2
    exit 1
fi
case "$BEM_ACTION" in
    setup) exec "$BEM_JULIA" --startup-file=no --project="$BEM_HERE/environment" -e 'using Pkg; Pkg.instantiate()' ;;
    test) exec "$BEM_JULIA" --startup-file=no --project="$BEM_HERE/environment" "$BEM_HERE/test.jl" "$@" ;;
    replay|evolve) exec "$BEM_JULIA" --startup-file=no --project="$BEM_HERE/environment" "$BEM_HERE/run.jl" --mode "$BEM_ACTION" "$@" ;;
    *) echo 'Usage: run.sh {setup|export|replay|evolve|test|verify|plot} [options]' >&2; exit 1 ;;
esac
