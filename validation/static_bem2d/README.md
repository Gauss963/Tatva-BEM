# Legacy static Bem2d.jl comparison

This preserves the earlier static DDM reconstruction of
`local-mus08-muk06-gap0-v2-ev16`. The active fully elastodynamic implementation
is in [../../bem](../../bem/README.md).

This experiment supplies the measured FEM plastic slip and actuator
displacement to a static two-domain solve. It is not an independent slip or
dynamic rupture prediction. The original implementation showed local
friction-bound violations when transferring the restart; autonomous static
evolution therefore stops unless state-changing relaxation is explicitly
requested. The new direct BEM formulation has a different, independently
tested static-lift path.

## Run

Requires Julia 1.10, Python 3.11+ with NumPy, h5py and Matplotlib, and a Python
installation usable by Julia PyCall. Set `BEM_JULIA`, `BEM_PYTHON`, and
`PYTHON` if they are not found on PATH.

```sh
./run.sh setup
./run.sh test
./run.sh replay --convergence
./run.sh plot
```

Outputs stay in `output/`. The small `input/` and `reference/` snapshots are
self-contained. `export` and `verify` are optional read-only tools for the
original external run; ordinary replay does not access that checkout.

`vendor/Bem2d/` contains the unchanged original source kernel and MIT license
from brendanjmeade/Bem2d.jl. The minimal dependency environment omits unrelated
FEM and plotting tools that this experiment does not use. The environment's
source link points inside this directory.
