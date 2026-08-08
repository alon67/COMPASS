# Withdrawn artifacts

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
