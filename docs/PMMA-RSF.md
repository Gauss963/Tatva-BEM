# PMMA-RSF

Dynamic rupture simulations for a two-block PMMA fault using Tatva and a
regularized rate-and-state friction law. The repository contains the complete
simulation, checkpoint, HDF5 dump, HPC launch, and post-processing workflow.

The Tatva core is based on upstream `v0.11.5`. The RSF contact projection was
validated separately against SCEC TPV101 and TPV102 before being used by the
PMMA model.

## Repository layout

```text
cases/                 TOML simulation inputs and coarse pilot scans
CohesiveZoneModel/     Cohesive-zone anchor and LSW-to-RSF calibration
scripts/run_case.py    Simulation entry point
scripts/               Analysis and animation programs
slurm/                 NANO4 GPU and F1 CPU job scripts
tatva/pmma/            PMMA model, RSF profiles, dynamics, and runner
tatva/                 Tatva core plus validated friction laws
tests/                 Tatva, friction, PMMA, and post-processing tests
runs/                  Generated TS#### case directories (not committed)
```

Each case is self-contained and does not read external geometry, material, or
solver input files. The consistent PMMA unit system is millimeter, second,
megapascal, and tonne per cubic millimeter.

## Installation

Python 3.11 or newer is required. In the existing `tatva` conda environment:

```bash
python -m pip install -e '.[analysis,dev]'
```

For GPU execution, install the CUDA-enabled JAX build appropriate for the HPC
system before installing this project.

## Run a case

Validate mesh size, time step, frame counts, and dump budget without creating a
run:

```bash
python scripts/run_case.py cases/rsf_0124_q4_vs30_long_transition_12h.toml --preflight
```

Launch locally:

```bash
python scripts/run_case.py cases/rsf_0124_q4_vs30_long_transition_12h.toml
```

Every new invocation atomically allocates the next label-free directory,
beginning at `runs/TS0117/`. A run contains the resolved input, HDF5 dump,
checkpoint, logs, statistics, and status. Resuming uses the same directory and
does not consume another sequence number.

Submit the default single-H200 production job on NANO4:

```bash
sbatch slurm/PMMA-RSF-GPU.slurm
```

See [`docs/HPC_CASES.md`](HPC_CASES.md) for case overrides, checkpoint
resume, pilot arrays, and CPU post-processing.

## Post-processing

Simulation jobs intentionally skip plotting and animation. Run the complete
analysis suite later on a CPU node:

```bash
python scripts/postprocess_velocity_weakening_run.py \
  --input runs/TS0120/data/simulation.h5
```

The F1 Slurm scripts can render frames with many workers while retaining bounded
per-worker memory use.

## Upstream Tatva

`origin` is the PMMA-RSF repository. `upstream` tracks
`https://github.com/smec-ethz/tatva`. Upstream updates should be merged only
after the friction, PMMA, and full Tatva test suites pass.

## License

Tatva and this derivative remain licensed under LGPL-3.0-or-later. See
`COPYING` and `COPYING.LESSER`.
