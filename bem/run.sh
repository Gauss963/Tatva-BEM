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
# Frequency workers provide CPU parallelism. Set BLAS limits before import,
# including Accelerate, which threadpoolctl may not discover at runtime.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
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
elif [ "${1:-}" = condensation-benchmark ]; then
    shift
    exec "$FDBEM_PYTHON" -B "$FDBEM_ROOT/tools/benchmark_condensation.py" "$@"
elif [ "${1:-}" = elastic-onset ]; then
    shift
    exec "$FDBEM_PYTHON" -B "$FDBEM_ROOT/tools/elastic_onset.py" "$@"
elif [ "${1:-}" = passivity ]; then
    shift
    exec "$FDBEM_PYTHON" -B "$FDBEM_ROOT/tools/audit_passivity.py" "$@"
elif [ "${1:-}" = galerkin-audit ]; then
    shift
    exec "$FDBEM_PYTHON" -B "$FDBEM_ROOT/tools/galerkin_audit.py" "$@"
elif [ "${1:-}" = galerkin ]; then
    shift
    exec "$FDBEM_PYTHON" -B "$FDBEM_ROOT/tools/run_galerkin.py" "$@"
elif [ "${1:-}" = galerkin-wave ]; then
    shift
    exec "$FDBEM_PYTHON" -B "$FDBEM_ROOT/tools/galerkin_wave.py" "$@"
elif [ "${1:-}" = galerkin-benchmark ]; then
    shift
    exec "$FDBEM_PYTHON" -B "$FDBEM_ROOT/tools/benchmark_galerkin.py" "$@"
elif [ "${1:-}" = galerkin-quadrature ]; then
    shift
    exec "$FDBEM_PYTHON" -B "$FDBEM_ROOT/tools/galerkin_quadrature.py" "$@"
elif [ "${1:-}" = galerkin-report ]; then
    shift
    exec "$FDBEM_PYTHON" -B "$FDBEM_ROOT/tools/report_galerkin.py" "$@"
elif [ "${1:-}" = fem-loading-replay ]; then
    shift
    exec "$FDBEM_PYTHON" -B "$FDBEM_ROOT/tools/replay_fem_loading.py" "$@"
elif [ "${1:-}" = fem-reference ]; then
    shift
    exec "$FDBEM_PYTHON" -B "$FDBEM_ROOT/tools/run_fem_reference.py" "$@"
elif [ "${1:-}" = weakening-audit ]; then
    shift
    exec "$FDBEM_PYTHON" -B "$FDBEM_ROOT/tools/audit_weakening_resolution.py" "$@"
elif [ "${1:-}" = disk-spectrum-benchmark ]; then
    shift
    exec "$FDBEM_PYTHON" -B "$FDBEM_ROOT/tools/benchmark_disk_spectrum.py" "$@"
elif [ "${1:-}" = galerkin-work ]; then
    shift
    exec "$FDBEM_PYTHON" -B "$FDBEM_ROOT/tools/audit_nodal_work.py" "$@"
elif [ "${1:-}" = galerkin-density ]; then
    shift
    exec "$FDBEM_PYTHON" -B "$FDBEM_ROOT/tools/galerkin_density.py" "$@"
elif [ "${1:-}" = bdf2-domain ]; then
    shift
    exec "$FDBEM_PYTHON" -B "$FDBEM_ROOT/tools/bdf2_domain.py" "$@"
elif [ "${1:-}" = adjoint ]; then
    shift
    exec "$FDBEM_PYTHON" -B "$FDBEM_ROOT/tools/run_adjoint.py" "$@"
elif [ "${1:-}" = forward-sensitivity-check ]; then
    shift
    exec "$FDBEM_PYTHON" -B "$FDBEM_ROOT/tools/check_forward_sensitivity.py" "$@"
elif [ "${1:-}" = radau-experiment ]; then
    shift
    exec "$FDBEM_PYTHON" -B "$FDBEM_ROOT/tools/radau_experiment.py" "$@"
elif [ "${1:-}" = radau-ad-check ]; then
    shift
    exec "$FDBEM_PYTHON" -B "$FDBEM_ROOT/tools/check_radau_ad.py" "$@"
elif [ "${1:-}" = resolvent-replay ]; then
    shift
    exec "$FDBEM_PYTHON" -B "$FDBEM_ROOT/tools/replay_resolvent.py" "$@"
elif [ "${1:-}" = resolvent-simulate ]; then
    shift
    exec "$FDBEM_PYTHON" -B "$FDBEM_ROOT/tools/simulate_resolvent.py" "$@"
elif [ "${1:-}" = resolvent-adjoint ]; then
    shift
    exec "$FDBEM_PYTHON" -B "$FDBEM_ROOT/tools/adjoint_resolvent.py" "$@"
elif [ "${1:-}" = native-resolvent-kernel ]; then
    shift
    exec "$FDBEM_PYTHON" -B "$FDBEM_ROOT/tools/prepare_native_resolvent.py" "$@"
elif [ "${1:-}" = mixed-resolvent-kernel ]; then
    shift
    exec "$FDBEM_PYTHON" -B "$FDBEM_ROOT/tools/prepare_mixed_resolvent.py" "$@"
elif [ "${1:-}" = mixed-quadrature-audit ]; then
    shift
    exec "$FDBEM_PYTHON" -B "$FDBEM_ROOT/tools/audit_mixed_quadrature.py" "$@"
elif [ "${1:-}" = cauchy-reuse-audit ]; then
    shift
    exec "$FDBEM_PYTHON" -B "$FDBEM_ROOT/tools/audit_cauchy_reuse.py" "$@"
elif [ "${1:-}" = reused-resolvent-kernel ]; then
    shift
    exec "$FDBEM_PYTHON" -B "$FDBEM_ROOT/tools/prepare_reused_resolvent.py" "$@"
elif [ "${1:-}" = fmm2d-library ]; then
    shift
    exec "$FDBEM_PYTHON" -B "$FDBEM_ROOT/tools/build_fmm2d_library.py" "$@"
elif [ "${1:-}" = native-spectrum-check ]; then
    shift
    exec "$FDBEM_PYTHON" -B "$FDBEM_ROOT/tools/check_radial_rollout.py" "$@"
elif [ "${1:-}" = static-refinement ]; then
    shift
    exec "$FDBEM_PYTHON" -B "$FDBEM_ROOT/tools/static_refinement.py" "$@"
elif [ "${1:-}" = compare ]; then
    shift
    exec "$FDBEM_PYTHON" -B "$FDBEM_ROOT/tools/compare_event16.py" "$@"
elif [ "${1:-}" = front-compare ]; then
    shift
    exec "$FDBEM_PYTHON" -B "$FDBEM_ROOT/tools/compare_fronts.py" "$@"
fi
exec "$FDBEM_PYTHON" -B -m finiteblock_bem.cli "$@"
