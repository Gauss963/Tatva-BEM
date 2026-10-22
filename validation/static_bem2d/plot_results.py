"""Plot the exported FEM data and static BEM reconstruction; writes locally."""
from __future__ import annotations
import json
from pathlib import Path
import tomllib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
OUT=HERE/'output'


def plot():
    meta=tomllib.loads((HERE/'input/reference.toml').read_text())
    case=tomllib.loads((HERE/'input/case.toml').read_text())
    cmp=np.atleast_1d(np.genfromtxt(OUT/'comparison.csv',delimiter=',',names=True))
    history=np.genfromtxt(HERE/'reference/fem_history.csv',delimiter=',',names=True)
    frame=meta['normal_frame']
    reference=np.genfromtxt(HERE/f'reference/frame_{frame:05d}.csv',delimiter=',',names=True)
    hs=sorted(set(cmp['h_mm']),reverse=True)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    fig.suptitle('Event 16: static BEM reconstruction using FEM plastic slip\n'
                 'Measured actuator and plastic slip are inputs; this is not a dynamic rupture prediction.',fontsize=12)
    axes[0,0].plot(reference['y'],reference['p'],'k-',label='FEM, normal phase end',lw=1.3)
    axes[0,1].plot(reference['y'],reference['tau'],'k-',label='FEM, normal phase end',lw=1.3)
    for h in hs:
        profile=np.genfromtxt(OUT/f'h{h:g}/frame_{frame:05d}.csv',delimiter=',',names=True)
        axes[0,0].plot(profile['y'],profile['p'],label=f'BEM h={h:g} mm',lw=1)
        axes[0,1].plot(profile['y'],profile['tau'],label=f'BEM h={h:g} mm',lw=1)
    axes[0,0].set(xlabel='Along-fault coordinate y (mm)',ylabel='Contact pressure (MPa)',title='Restart: full pressure profile, including endpoints')
    axes[0,1].set(xlabel='Along-fault coordinate y (mm)',ylabel='Resisting shear traction (MPa)',title='Restart: full shear profile, including endpoints')
    finest=min(hs)
    rows=np.sort(cmp[cmp['h_mm']==finest],order='time_s')
    axes[1,0].plot(history['time_s'] if 'time_s' in history.dtype.names else history['time'],history['avg_tau'],'k-',lw=1,label='FEM dynamic history')
    axes[1,0].plot(rows['time_s'],rows['bem_mean_tau_MPa'],'o',label=f'Static replay, h={finest:g} mm')
    axes[1,0].set(xlabel='FEM time (s)',ylabel='Mean interface shear traction (MPa)',title='Static snapshots versus dynamic history')
    for h in hs:
        row=cmp[(cmp['h_mm']==h)&(cmp['frame']==frame)][0]
        axes[1,1].plot(h,100*row['normal_force_relative_error'],'bo')
        axes[1,1].plot(h,100*row['shear_force_relative_error'],'rs')
    axes[1,1].plot([],[],'bo',label='Normal force imbalance')
    axes[1,1].plot([],[],'rs',label='Shear force imbalance')
    axes[1,1].set(xlabel='Maximum BEM panel length (mm)',ylabel='Integrated force imbalance (%)',title='Collocation discretization check at restart')
    for ax in axes.flat:
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(OUT/'comparison.png',dpi=180)
    plt.close(fig)

    moving=np.array([[0,0],[200,0],[200,480],[195,500],[0,500],[0,0]])
    stationary=np.array([[200,0],[345,0],[345,550],[200,550],[200,0]])
    fig,ax=plt.subplots(figsize=(7,8),layout='constrained')
    ax.fill(moving[:,0],moving[:,1],color='#d5e9f7',edgecolor='#27618b',label='Moving block')
    ax.fill(stationary[:,0],stationary[:,1],color='#eee3d0',edgecolor='#83643b',label='Stationary block')
    ax.plot([200,200],[0,480],'r-',lw=2,label='Penalty contact + LSW')
    for y in np.linspace(40,460,6):
        ax.annotate('',xy=(0,y),xytext=(-35,y),arrowprops={'arrowstyle':'->','color':'#27618b'})
    ax.text(-42,250,'16 MPa',rotation=90,va='center')
    ax.plot([345,345],[0,550],'k--',label='ux=0')
    ax.plot([200,345],[550,550],'k:',label='uy=0')
    ax.plot([0,200],[0,0],color='purple',lw=3,label='Rigid platen + spring')
    ax.text(35,-35,'k = 25 MPa/mm; actuator D',color='purple')
    ax.annotate('20 × 5 mm chamfer',xy=(198,491),xytext=(-15,535),arrowprops={'arrowstyle':'->'})
    ax.set(xlim=(-60,390),ylim=(-60,580),xlabel='x (mm)',ylabel='y (mm)',aspect='equal',
           title='local-mus08-muk06-gap0-v2-ev16\nPlane strain, E=7662 MPa, ν=0.2')
    ax.legend(loc='center left',bbox_to_anchor=(1.0,.5),fontsize=8)
    fig.savefig(OUT/'geometry.png',dpi=160,bbox_inches='tight');plt.close(fig)

    base=cmp[(cmp['h_mm']==finest)&(cmp['frame']==frame)][0]
    tauerr=abs(base['bem_mean_tau_MPa']/base['fem_mean_tau_MPa']-1)*100
    perr=abs(base['bem_mean_p_MPa']/base['fem_mean_p_MPa']-1)*100
    lines=[
        '# Event 16 BEM comparison report','',
        '**Status: static reconstruction runs; full dynamic event is not reproduced.**','',
        'The HDF5 source was opened read-only. All generated files are in this folder.',
        'This calculation supplies each selected FEM snapshot’s plastic slip and actuator displacement to a static two-domain BEM solve. It is an elasticity/contact diagnostic, not an independent prediction of those inputs.','',
        f'Finest maximum panel length: {finest:g} mm. Normal-phase reference: frame {frame}.','',
        '| Quantity at restart | FEM | Static BEM |',
        '|---|---:|---:|',
        f'| Mean pressure (MPa) | {base["fem_mean_p_MPa"]:.6f} | {base["bem_mean_p_MPa"]:.6f} |',
        f'| Mean shear traction (MPa) | {base["fem_mean_tau_MPa"]:.6f} | {base["bem_mean_tau_MPa"]:.6f} |',
        f'| Platen displacement (mm) | {base["fem_platen_mm"]:.6f} | {base["bem_platen_mm"]:.6f} |','',
        f'Mean-pressure difference: {perr:.4f}%. Mean-shear difference: {tauerr:.4f}%.',
        f'Integrated normal-force imbalance: {100*base["normal_force_relative_error"]:.4f}%; shear-force imbalance: {100*base["shear_force_relative_error"]:.4f}%.',
        f'Full-profile relative L2 errors: pressure {100*base["p_profile_rel_l2_all"]:.2f}%, shear {100*base["tau_profile_rel_l2_all"]:.2f}%.',
        f'After excluding 20 mm at each end: pressure {100*base["p_profile_rel_l2_trim20"]:.2f}%, shear {100*base["tau_profile_rel_l2_trim20"]:.2f}%. Both full and trimmed metrics are retained in comparison.csv.','',
        '## Unresolved differences','',
        f'- Frozen-plastic restart maximum |tau|/(mu*p) = {base["max_strength_ratio"]:.3f}. Values above 1 mean that the transferred FEM state violates the BEM friction bound locally. Replay deliberately reports this instead of clipping it.',
        '- Endpoint fields and transferred plastic-slip gradients do not match. Small global errors do not establish local contact/friction convergence.',
        '- The autonomous quasistatic mode refuses an inadmissible restart by default. Allowing restart relaxation changes the initial state and is a separate diagnostic.',
        '- Tatva uses Quad4 bulk FEM and nodal penalty contact; this implementation uses constant boundary elements and collocated contact.',
        '- Tatva includes inertia and stress waves. Static BEM does not. Late dynamic snapshots are explicitly not expected to satisfy static equilibrium.',
        '- The implicit local return map uses the same continuous LSW law but differs from Tatva’s explicit old-cumulative-slip microstep update.',
        '- FEM interface tractions are reconstructed from float32 displacement/plastic-slip output; that subtraction loses traction precision.',
        '- The run records a Git revision and rigid-platen/restart features not fully represented by the current source snapshot. Archived run configuration and HDF5 attributes take precedence.','',
        '## Artifacts','',
        '- comparison.png: full endpoint profiles, dynamic history with static snapshots, force refinement.',
        '- geometry.png: geometry, supports and loading.',
        '- comparison.csv: all snapshots, mesh sizes and error metrics.',
        '- reference/provenance.json: original paths, source hashes and recorded HDF5 metadata.',
        '- reference_verification.json and tests.log: generated after validation.','',
        'A small residual verifies the assembled equations only. It does not verify dynamic equivalence or a mesh-independent rupture solution.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    print(f'Wrote plots and report to {OUT}')


if __name__=='__main__':
    plot()
