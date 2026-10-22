# Tatva-BEM

Fully elastodynamic boundary elements for finite-block frictional contact,
developed alongside the Tatva-based PMMA finite-element model.

**Research implementation: the full 45 ms event-16 FEM rupture has not yet
been reproduced or shown to converge.** The current BEM engine has analytic
wave/reflection verification and short nonlinear two-block diagnostics.

This repository preserves the existing Tatva/PMMA Git history. Its upstream
is [smec-ethz/tatva](https://github.com/smec-ethz/tatva); the inherited PMMA
baseline comes from [Gauss963/PMMA-RSF](https://github.com/Gauss963/PMMA-RSF).

## BEM implementation

- Direct 2D plane-strain elastodynamic boundary integral operators.
- Laplace-domain P/S wave kernels and singular-panel integration.
- BDF2 convolution quadrature retaining the complete boundary history.
- Two finite blocks with chamfer, original supports, rigid platen and spring.
- Unilateral penalty contact and cumulative linear slip weakening.
- Static-lift initialization using a local snapshot of event-16 plastic slip.
- Explicit failure states, memory preflight, local artifacts and provenance.

The BEM package is independent of the Tatva FEM time integrator. A shared
backend interface and a production-scale accelerated history solver remain
future work. No files in the external original Tatva checkout are modified.

## Quick start

Python 3.11 or newer:

```sh
python -m venv .venv
.venv/bin/python -m pip install -e './bem[test,plot]'
.venv/bin/python -m pytest bem/tests
.venv/bin/finiteblock-bem benchmark --name bar-reference
.venv/bin/finiteblock-bem smoke --name smoke-reference
.venv/bin/finiteblock-bem event16 --name event16-short
```

The helper `bem/run.sh` also runs directly from a checkout with an existing
compatible Python. Set `FDBEM_PYTHON` if needed. For scientific plots:

```sh
FDBEM_PYTHON=.venv/bin/python bem/run.sh plot --name bar-reference
```

`benchmark` checks an analytic finite-bar reflection problem. `smoke` uses
accelerated diagnostic loading, not the original event history. `event16`
starts from a BEM static lift and uses the original actuator speed, but the
default run is only 0.512 ms and does not reach rupture nucleation.

See [BEM usage and current results](bem/README.md),
[formulation](bem/docs/formulation.md),
[validation](bem/docs/validation.md), and
[remaining acceptance gates](bem/docs/roadmap.md).

## Repository layout

```text
bem/                       Independent finiteblock-bem Python package
validation/static_bem2d/   Earlier static DDM comparison and small FEM snapshots
tatva/                     Inherited Tatva FEM core and PMMA model
cases/, scripts/, slurm/   Inherited FEM inputs and execution workflow
tests/                     Inherited Tatva/FEM tests
docs/PMMA-RSF.md            Preserved FEM usage documentation
docs/UPSTREAM.md            Repository ancestry and remote conventions
.github/workflows/bem.yml   BEM tests and style checks on Python 3.11/3.12
```

The legacy static comparison vendors the original MIT-licensed Bem2d.jl
kernel so its Julia environment does not point outside this repository.
It is a separate validation experiment, not the dynamic BEM backend.

Generated trajectories, caches, run logs and large HDF5 outputs are ignored.
Only small model snapshots needed to reproduce the diagnostics are included.

## FEM baseline

The inherited finite-element workflow is documented in
[docs/PMMA-RSF.md](docs/PMMA-RSF.md). Its dependencies and test suite are
separate from the minimal BEM package. The BEM workflow does not certify
the inherited FEM suite or the complete coupled event.

## Licensing

The inherited Tatva code retains LGPL-3.0-or-later and its existing notices.
The independent BEM package has its own [MIT license](bem/LICENSE).
The vendored Bem2d.jl kernel retains its original MIT notice.
