# Repository ancestry and remotes

```text
origin    git@github.com:Gauss963/Tatva-BEM.git
upstream  https://github.com/smec-ethz/tatva.git
```

The repository was initialized from the committed Tatva/PMMA checkout at
`c309f21` (`Point the cohesive-calibration test at rsf_0121`). Its existing
history was preserved. Untracked simulation outputs, private notes and
working files from the external checkout were not imported.

The PMMA repository began with an imported source snapshot rather than an
upstream Git parent. Its documented core baseline is Tatva `v0.11.5`; the
core diff against that tag consists of the PMMA/friction additions and two
small core-file changes. An ancestry-only merge records that exact tag as
an additional parent while preserving the downstream tree. It does not
claim that the latest upstream `main` has been integrated.

That baseline is also developed in Gauss963/PMMA-RSF. The BEM model's archived
event-16 reference records revision
`933ae1f3fe1c634391bdbd316e4efa5dbf0e6d5d`; the small archived input snapshot,
rather than the current FEM implementation, defines that comparison case.

When cloning this repository, Git only installs `origin`; add the upstream
remote explicitly:

```sh
git remote add upstream https://github.com/smec-ethz/tatva.git
git fetch upstream
```

Review upstream changes before merging. Updating inherited FEM code does not
automatically update the independent BEM boundary operators, contact adapter,
initial-condition assumptions or validation results.

The original `/Volumes/Gauss-T7/tatva` repository was used read-only. This
checkout and all BEM artifacts are independent of it.
