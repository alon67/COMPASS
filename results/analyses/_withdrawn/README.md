# Withdrawn artifacts

Nothing in this directory is release evidence. It is excluded from the release
inventory (`configs/release_inventory.json`, `analyses.exclude`), and
`tools/build_release_manifest.py` **fails the release** if any file matching
`orphan_policy.forbidden_basename_patterns` is found outside it.

## `*_n4_projected` (2026-08-09, post-response-3 audit P0-4C)

`n4_projected` was removed as a sensitivity arm when it was found to be a
duplicate of `primary`: the N4 projection now happens inside the primary arm's
source table, so a separate arm selected the same models and its agreement with
the primary conclusion was arithmetic rather than evidential.

The **outputs** were not removed at the same time, and the broad
`results/analyses/**/*.json|parquet` inventory rule kept shipping them. One of
them, `decision_n4_projected.json`, still asserts an overall `H-G` conclusion
with H-B scored `0/2` — which directly contradicts the current decision, where
H-B is `0/0` and no overall label is asserted. A reader who opened it would find
a released, hash-verified artifact stating a conclusion the project does not
hold.

Retained here so the history is legible; excluded so it cannot be read as
current.

## `*fast5*` (same date, same finding)

A five-model fast pass used while developing the perturbation battery. The
current `tools/run_all.sh` does not generate it, so it is an orphan: an output
with no producer in the declared generation graph.

## `readout_resolution` (2026-08-06)

`readout_resolution.json` / `readout_resolution.parquet` and the tool that
produced them (`tools/_withdrawn_readout_resolution.py`) are retained here as a
record, **not** as evidence. The 2026-08-06 audit (section 4) showed the analysis
was invalid:

* the statistic it tested (`mean_nonadditivity_sd`) is a mean over eight
  backgrounds, whose quantisation grid is 2.5/8 = 0.3125 Hz, while the "floor"
  it was compared against was one *unaveraged* 2.5 Hz single-observation
  quantum, and the per-model median over six tuple means can be finer still;
* a deterministic quantisation grid is not a stochastic detection limit -- a
  mean below one quantum is exactly observable and exactly repeatable; and
* the boolean `resolvable` column tested neither measurement error, simulation
  variability, background variability, nor the uncertainty of the mean.

Every conclusion drawn from it is withdrawn, in particular "four models are
unresolvable", "the effect is carried by the two toy models", and "both cortical
reconstructions are below a hard floor".

The replacement is `tools/perturbation_inference.py` ->
`results/analyses/perturbation_inference.json`, which separates the two
questions the withdrawn tool conflated: **numerical repeatability** (rerun at a
finer time step, compare per-background eta) and **inferential precision**
(bootstrap and exact sign-flip inference over the background distribution).
