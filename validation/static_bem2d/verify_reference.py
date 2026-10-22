"""Verify input provenance and compare the Julia friction law with Tatva's source.

Only the two inspected, pure LSW functions are compiled from their AST, with
NumPy substituted for jax.numpy. No Tatva package import or run is performed.
"""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import types

import numpy as np

HERE=Path(__file__).resolve().parent


def verify():
    provenance=json.loads((HERE/'reference/provenance.json').read_text())
    for filename, record in provenance['files'].items():
        path=Path(filename)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256'], filename
    h5=provenance['hdf5']
    stat=Path(h5['path']).stat()
    assert stat.st_size==h5['bytes'] and stat.st_mtime_ns==h5['mtime_ns']
    source=Path(next(p for p in provenance['files'] if p.endswith('/tatva/friction.py')))
    tree=ast.parse(source.read_text())
    names={'linear_slip_weakening_coefficient','linear_slip_weakening_strength'}
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(functions)==2
    # The source functions contain no imports, file operations or callbacks.
    namespace={'jnp':np,'jax':types.SimpleNamespace(Array=np.ndarray)}
    subset=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+functions,type_ignores=[])
    exec(compile(ast.fix_missing_locations(subset),str(source),'exec'),namespace)
    samples=np.genfromtxt(HERE/'output/law_samples.csv',delimiter=',',names=True)
    mu=namespace['linear_slip_weakening_coefficient'](
        samples['new_cumulative'],static_friction=.8,kinetic_friction=.6,
        characteristic_slip=.0125959)
    np.testing.assert_allclose(samples['mu'],mu,atol=1e-14,rtol=1e-14)
    assert np.all(abs(samples['tau'])<=samples['p']*mu+1e-10)
    result={'source_fingerprints_unchanged':True,'lsw_matches_tatva_source':True,
            'lsw_samples':len(samples),'hdf5_size_and_mtime_unchanged':True,
            'scope':'Checks the recorded source files and main HDF5, not every file on the external volume.'}
    (HERE/'output/reference_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    verify()
