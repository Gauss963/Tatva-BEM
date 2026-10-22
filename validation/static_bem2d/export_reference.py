"""Read a Tatva run without importing or executing any Tatva code.

Only selected small HDF5 slices are read. All writes stay inside this folder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tomllib

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT = Path('/Volumes/Gauss-T7/tatva/runs/local-mus08-muk06-gap0-v2-ev16')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(run=DEFAULT):
    run = Path(run).resolve()
    out = HERE / 'reference'
    out.mkdir(exist_ok=True)
    (HERE / 'input').mkdir(exist_ok=True)
    files = [run / 'input/case.toml', run / 'input/resolved.json']
    for path in files:
        shutil.copyfile(path, HERE / 'input' / path.name)
    case = tomllib.loads(files[0].read_text())
    if case['solver']['friction_law'] != 'slip-weakening':
        raise ValueError('This reproduction currently implements slip-weakening only.')
    source_root = run.parent.parent
    files += [source_root / 'tatva/pmma/dynamics.py', source_root / 'tatva/friction.py']
    provenance = {'read_only_source': str(run), 'files': {
        str(p): {'sha256': sha256(p), 'bytes': p.stat().st_size} for p in files}}
    h5 = run / 'data/simulation.h5'
    provenance['hdf5'] = {'path': str(h5), 'bytes': h5.stat().st_size,
                          'mtime_ns': h5.stat().st_mtime_ns}
    with h5py.File(h5, 'r') as f:
        columns = [x.decode() for x in f['history_columns'][:]]
        history = f['history'][:]
        np.savetxt(out / 'fem_history.csv', history, delimiter=',', header=','.join(columns), comments='')
        phases = f['phase_id'][:]
        # The last normal-phase frame contains the restarted equilibrium and
        # the healed weakening state used at the beginning of event 16.
        normal = int(np.flatnonzero(phases == 1)[-1])
        slip = history[:, columns.index('max_slip')]
        onset_ids = np.flatnonzero((phases == 2) & (slip > 1e-7))
        onset = int(onset_ids[0]) if onset_ids.size else len(phases) - 1
        chosen = sorted(set([normal, normal + 1, 400, 2000, 4000, onset - 1, onset, len(phases) - 1]))
        chosen = [i for i in chosen if 0 <= i < len(phases)]
        y = f['interface/contact_line_y'][:].astype(float)
        master = f['interface/master_nodes'][:]
        slave = f['interface/slave_nodes'][:]
        kn = float(case['solver']['normal_penalty'])
        kt = f['interface/tangential_penalty_profile'][:].astype(float)
        for i in chosen:
            um = f['moving/displacement'][i, master, :].astype(float)
            us = f['stationary/displacement'][i, slave, :].astype(float)
            z = f['interface/plastic_slip'][i].astype(float)
            cumulative = f['interface/cumulative_slip'][i].astype(float)
            p = kn * np.maximum(um[:, 0] - us[:, 0], 0)
            trial = kt * (um[:, 1] - us[:, 1] - z)
            mu = f['interface/friction_coefficient'][i].astype(float)
            # Reconstruct the stored return-mapped shear traction; open nodes
            # carry no traction. Float32 u/z dumps lose some traction precision.
            tau = np.where(p > 0, trial, 0)
            values = np.column_stack((y, p, tau, z, cumulative, mu, um, us))
            order = np.argsort(y)
            np.savetxt(out / f'frame_{i:05d}.csv', values[order], delimiter=',',
                       header='y,p,tau,plastic_slip,cumulative_slip,mu,umx,umy,usx,usy', comments='')
        initial = np.loadtxt(out / f'frame_{normal:05d}.csv', delimiter=',', skiprows=1)
        np.savetxt(HERE / 'input/restart_interface.csv', initial, delimiter=',',
                   header='y,p,tau,plastic_slip,cumulative_slip,mu,umx,umy,usx,usy', comments='')
        attrs = {k: (v.tolist() if isinstance(v, np.ndarray) else v.item()
                     if isinstance(v, np.generic) else v) for k, v in f.attrs.items()}
        provenance['hdf5_attributes'] = attrs
        provenance['normal_frame'] = normal
        provenance['first_slip_frame_threshold_1e_7_mm'] = onset
        provenance['selected_frames'] = chosen
        d0 = history[normal, columns.index('applied_shear_displacement')]
        meta = [f'normal_frame = {normal}', f'first_slip_frame = {onset}',
                f'actuator_initial = {float(d0)!r}', f'frames = {chosen}',
                f'normal_time = {float(history[normal, 0])!r}',
                f'normal_phase_time = {float(case["solver"]["normal_phase_time"])!r}',
                f'restart_actuator_exact = {float(f.attrs["restart_actuator_displacement"])!r}']
        (HERE / 'input/reference.toml').write_text('\n'.join(meta) + '\n')
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(f'Exported normal frame {normal}, first slip frame {onset}, {len(chosen)} snapshots to {out}')
    return provenance


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=DEFAULT)
    export(parser.parse_args().run)
