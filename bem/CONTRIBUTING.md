# Development

Use Python 3.11 or newer and an editable installation:

```sh
python -m pip install -e '.[test,plot,jax]' ruff==0.17.1
ruff check .
ruff format --check .
python -m pytest
```

Keep model units consistent. Source and target normals, physical boundary
tractions, and fictitious displacement discontinuities are different
quantities. Every change to wave kernels, singular integration, convolution
timing, contact updates, or state initialization needs a physical or analytic
verification appropriate to that change.

Do not replace wave history with radiation damping under a fully dynamic
label. Do not silently truncate failed runs, relax restart states, change
friction updates, or ignore memory preflight failures. Preserve failure
metadata and valid prefixes.

Generated arrays, plots, wheels and run logs stay in `outputs/`, which is
ignored by Git. Small packaged model snapshots are intentional inputs.
Never write into the external Tatva source/run directory.
