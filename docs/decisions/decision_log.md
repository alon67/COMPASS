# COMPASS decision log

Design decisions that shaped the study, with the evidence that motivated each.
Blueprint WP0 deliverable ("decision log template") and §26 ("record failed
simulations and exclusions").

Format: **what was decided**, *why*, and what would have happened otherwise.

---

## D1 — Parameters are multiplicative scale factors, not absolute densities

**Decided:** every free parameter is a log-uniform multiplicative factor on the
model's canonical per-segment value; the published model sits at 1.0.

**Why:** blueprint §14.2 forbids comparing raw parameter magnitudes across
models. A scale factor is unit-free and semantically identical across a
1-compartment point model and a 1,106-compartment reconstruction. It also
preserves each model's published spatial channel distributions.

**Otherwise:** overwriting Hay's apical `Ih` gradient, Mainen's axonal Na⁺
hotspot or Migliore's distance-dependent A-current with uniform densities would
have confounded "neuron identity" with "we deleted the channel gradient", making
the morphology/distribution ablation of §13.3 impossible to interpret.

## D2 — One model per process

**Decided:** NEURON is initialised fresh in every worker and each worker builds
exactly one model; `NeuronAdapter.build()` raises if a second model is requested.

**Why:** NEURON's section list is process-global. During development, building
Migliore after Mainen in one process silently gave Migliore 1,312 "compartments"
(it has 559 as distributed) and inflated its runtime from 4.6 s to 12 s per
simulation, because `h.allsec()` returned the previous model's sections too and
Migliore's mechanisms were inserted into them.

**Otherwise:** cross-model contamination that is invisible in the output tables.

## D3 — Deterministic fixed time step for the campaign, not CVODE

**Decided:** each model runs at the largest fixed step its convergence audit
accepted (`CAMPAIGN_SOLVER` in `src/compass/workflows/simulate.py`).

**Why:** the audit selects CVODE for most models at the *canonical* point, but
under a broad prior CVODE's cost tracks each sample's stiffness. Profiling
`mainen_l5pc` over six prior samples: median 11 s, individual samples 233 s and
307 s, mean 96 s. Fixed-step cost is constant and the results are
bit-reproducible.

**Otherwise:** an unbudgetable campaign, and — if slow samples were ever
abandoned — a sampling bias toward well-behaved regions of parameter space.

**Evidence:** `data/provenance/numerical_qc.json`.

## D4 — The rheobase search inherits the model's solver settings

**Decided:** `find_rheobase` forwards the full solver configuration to every
probe.

**Why:** the bisection costs ~9 simulations per sample. An early version passed
only `dt_ms`, so the search silently ran fixed-step while the protocols ran under
CVODE, making threshold-finding ~10× more expensive than all six protocols
combined (13.5 s vs 1.5 s for `bbp_l5pc`).

## D5 — Migliore's initial-state procedure is reproduced per sample

**Decided:** `MiglioreCA1._post_finitialize` solves
`e_pas(x) = v(x) + (ina(x)+ik(x))/g_pas(x)` after every `finitialize`.

**Why:** this is the published model's own procedure (`proc init` in
`fig_1a.hoc`). Without it the cell rests at −74.9 mV instead of the published
−65 mV, and resting potential becomes an artefact of the conductance draw rather
than the model's compensation. Because the standing currents depend on the
sampled conductances, it must be redone for every sample, not once at build time.

## D6 — Migliore runs on the d_lambda mesh at dt = 0.025 ms  *(revised; see D15)*

**Decided:** d_lambda(100 Hz) segmentation (1,106 segments), dt = 0.025 ms,
N = 512.

**Why:** the as-distributed 559-segment mesh differs by one spike from finer
meshes on a knife-edge 0.5 nA step.

**Revised 2026-08-05.** This entry originally justified dt = 0.025 ms as a
knowingly-unconverged compromise and declared "a small systematic bias in this
model's absolute rate features" as a limitation. The feature-level audit (D15)
shows that was **too pessimistic**: dt = 0.025 ms on the 1,106-segment mesh
reproduces all 13 conditioning features on all 6 prior samples with agreement
1.00. The original `numerical_qc` rejection came from comparing exact spike
counts at one canonical point. No bias is carried; the limitation is withdrawn.

## D7 — Traces are copied out of NEURON, never viewed

**Decided:** `np.array(vec, copy=True)` rather than `np.asarray(vec)`.

**Why:** numpy can wrap a NEURON `Vector`'s buffer without copying. When the
Vector is garbage-collected at the end of `simulate()`, the array points at freed
memory that the next run can reuse, so two `Recording` objects can silently
alias. Caught by `test_parameter_injection_and_reset_are_exact`, which found a
parameter injection that "had no effect" because both traces were the same
buffer.

**Impact on results:** none — every recording is consumed before the next
simulation begins — but the hazard is removed.

## D8 — The synergy endpoint is an area, not a mean over negative tuples

**Decided:** `area_negative = mean_t min(Ω_t, 0)` over *all* tuples examined,
divided by tuple order.

**Why:** the mean over only the negative tuples can be dominated by a single
extreme tuple. On `hh_point`'s tightest ensemble the two definitions differ by a
factor of 2.8 (0.036 vs 0.013).

## D9 — E7 uses a preregistered per-sample covariance score, stratified by fitness

**Decided:** rank samples by the fraction of their squared deviation lying in the
ensemble's leading two principal directions, and take the top-scoring samples
*within target-distance deciles*.

**Why:** an earlier greedy random search raised mean |r| only from 0.098 to 0.105
— too weak a contrast to test the covariance hypothesis. The subspace score
raises it to 0.197 against a size-matched random subset at 0.098, and flips the
net O-information from synergy to redundancy. Stratifying by target distance is
essential: an unstratified high-covariance subset would also differ in phenotype
quality (blueprint risk table).

## D10 — A fourth, tighter tolerance level was added

**Decided:** tolerance series extended to `very_strict` = 0.5 z.

**Why:** solution-manifold contraction is the mechanism the generic-geometry
hypothesis proposes. At the blueprint's three levels the accepted set is still a
mildly trimmed prior (33% acceptance at permissive for `hh_point`); the fourth
level reaches 1.3% acceptance, where the manifold is genuinely thin.

**Frozen before** the screening campaign was analysed.

## D11 — The two most expensive models run the E1 ensemble only

**Decided:** `mainen_l5pc` and `migliore_ca1` are simulated for ensemble regime
E1 only; the other six models run E1, E2 and E6.

**Why:** measured throughput on the actual campaign, from `bbp_l5pc`'s own shard
timestamps:

| regime | shards | wall span | rate |
|---|---|---|---|
| E1 (independent prior) | 82 | 40.4 min | **2.03 blocks/min** |
| E2 (covariance prior) | 41 | 22.8 min | **1.80 blocks/min** |
| E6 (clustered prior) | 41 | 211.7 min | **0.18 blocks/min** |

E6 is ~11× slower than E1 at identical simulation settings. The cause is
sample composition, not numerics: a broad independent prior produces many samples
that never spike, for which the rheobase bisection exits after a single probe and
`extract()` skips eFEL spike-shape analysis entirely, whereas an ensemble
clustered on viable basins produces almost exclusively spiking samples that pay
the full ~11-probe search and five eFEL extractions each.

Extrapolated to `mainen_l5pc` (≈2.4× the per-simulation cost of `bbp_l5pc`),
E1+E2+E6 would have taken a further ~3.6 h, and `migliore_ca1` ~1.5 h beyond
that.

**What is preserved:** E1 carries the **primary** endpoints. Every conditioned
ensemble (E3/E3b/E4/E4b), the size-matched control (E3m), the high-covariance
subset (E7) and both statistical nulls (E8/E9) are *derived from E1* without
further simulation, so all four primary endpoints and both primary contrasts
(conditioning effect, covariance effect) retain full 8-model coverage.

**What is lost:** the E2 (covariance-preserving prior) and E6 (clustered prior)
contrasts, which support *secondary* endpoints, drop from 8-model to 6-model
coverage.

**Also:** `mainen_l5pc` is excluded from the Sobol campaign. A direct timing of
its per-sample cost under the campaign's fixed-step settings gave **122-143 s**
(rheobase bisection 102-125 s of that: ~13 probes on a 754-segment model), so
even the smallest usable Saltelli design (N=32 base → 704 evaluations) would
have needed ~2 h of wall time on 12 workers. Sobol therefore covers 6 of 8
models, including one multicompartment BBP-lineage model (`bbp_l5pc`).

## D12 — Sample sizes for the two expensive multicompartment models were cut to N=256

**Decided:** `mainen_l5pc` and `migliore_ca1` run E1 at N=256 rather than 1024
and 512.

**Why:** the measurement above. At 122-143 s per sample a 25-sample block needs
~55 min, so N=1024 (41 blocks on 12 workers) is ~3.7 h for one model. N=256 fits
a single 12-worker wave.

**What this costs:** these two models have the smallest ensembles in the panel,
and their strictly-conditioned subsets (E4/E4b) may fall below the 60-sample
floor and be dropped from the analysis. `n_samples` is carried on every endpoint
row and enters the hierarchical model as `log_n_samples`, so the reduced
precision is visible in every table rather than hidden. Cross-model endpoints are
additionally computed on size-matched subsamples.

**What this preserves:** the panel keeps all eight model families, including the
`mainen_l5pc` / `bbp_l5pc` replication pair — an independently built cortical
pyramidal model versus one from the Blue Brain pipeline — which is the single
most important contrast in the design and the reason not to drop either.

This decision was taken on compute grounds while the campaign was running, before
any cross-model result had been inspected, and changes no endpoint, tolerance,
estimator or decision threshold.

## D13 — Source must not be edited while a campaign is running

**What happened:** `base.py` was edited to add a call to a new
`_post_finitialize()` hook, and the hook's default implementation was added in a
*second* edit moments later. Simulation workers are spawned fresh per block and
re-import the package, so the blocks that happened to start inside that window
imported a source tree in which the call existed but the method did not. They
raised `AttributeError: 'PospischilRS' object has no attribute
'_post_finitialize'`.

**How it was caught:** by the campaign's own audit, not by luck. Failures are
recorded rather than discarded (`simulation_status`, `qc_flags`), so the
post-campaign summary showed 425 errored samples in `pospischil_rs` (2.6%) with
an exception code, and the traceback was retained in the `error` column.

**Repair:** the 17 shards containing any errored row were deleted and
regenerated (0.2 min — `pospischil_rs` is a point model). A verification stage
now runs after every campaign and asserts that zero non-`ok` rows remain across
all shards. Final count: **0 failed simulations in 78,336**.

**Rule adopted:** `tools/run_pipeline.sh` carries an explicit warning, and the
pipeline begins with a repair pass followed by a hard verification stage. Had
the errors been silently dropped instead of recorded, this would have shown up
only as an unexplained 2.6% shortfall in one model's sample size.

## D14 — Verification tooling needs verifying too

Four defects surfaced during the 2026-08-05 review response, and **three were in
code written to check the science**, not in the pipeline:

1. `module_stability` pooled top-tuple lists across models, which share no
   parameter names — reporting 0.074 where the correct per-ensemble value is
   0.813.
2. `convergence_audit.set_mesh` recomputed the mesh from *perturbed* `cm`/`Ra`
   (both are sampled parameters) and re-snapshotted them as canonical, so the
   mesh drifted 1106 -> 10546 segments across settings and agreement degraded
   monotonically with refinement. Two hours of compute produced six meaningless
   rejections.
3. The point-model branch of the same tool raised `KeyError` on models with no
   `"all"` section list, and the caller filtered stdout, so four models were
   silently skipped.
4. The decision code filtered `kind == "control"` after the control groups were
   renamed, so the perturbation criterion silently dropped from the tally.

**Rule adopted:** any tool that gates a scientific claim gets a self-consistency
check (mesh stability across repeated calls; non-empty group assertions; explicit
counts printed in the decision table). A verification result that *reverses* a
prior conclusion is re-derived before it is acted on.

## D15 — The original numerical QC was too narrow, and it mattered

`numerical_qc.py` accepted a time step if spike count matched at the single
canonical parameter point. Re-testing all 13 conditioning features across 6
prior samples showed **four of eight models running at unconverged settings**
(`hh_point`, `khaliq_purkinje`, `bbp_l5pc`, `mainen_l5pc`) and one model
(`migliore_ca1`) wrongly rejected by the narrow test. `mainen_l5pc` needed both
a finer step and a 3x finer mesh.

All four were re-simulated. Superseded shards are archived under
`results/features/screening_superseded_dt/`, never deleted.

## D16 — The perturbation null was an artefact, and fixing it reversed the result

The first release concluded that statistical synergy modules are "no more
functionally interactive than randomly chosen ones" (1.40x, r(Omega,|eta|) =
-0.064). That experiment drew backgrounds from E1-filtered-for-firing rather than
the accepted manifold (49 of 64 were E3 members), pushed 279 evaluations outside
the declared prior, normalised against the unconditioned ensemble, tested one
feature, and used middle-Omega tuples while the prose claimed random ones.

Corrected: backgrounds from E3, displacements sign-flipped inward (0 out-of-prior
evaluations), normalisation on the tested population, four features, both control
families, paired and bootstrap uncertainty. Result: **2.34x vs matched controls
and 4.05x vs random controls**, both CIs excluding 1.0.

**The earlier negative claim was wrong, not merely underpowered.** The H-G
override argued in the first release rested on it and has been withdrawn; the
conclusion is now H-M on the blueprint's own 19.3 definition.

## D17 — Sobol's headline feature was the one with the least data

Saltelli mean-imputes undefined outputs. That missingness is phenotype-dependent,
not random, and biases `1 - sum(S1)` upward. Spike-frequency adaptation had the
lowest valid fraction in the panel (0.53) and the highest apparent interaction
(0.99). Quoting is now gated at a valid fraction of 0.90; the mean over quotable
analyses is 0.363, and the "adaptation is ~95% interaction-driven" claim is
withdrawn.

---

## Exclusions

| asset | intended slot | decision |
|---|---|---|
| ModelDB 97747 (Golomb FS interneuron) | FS interneuron replication | **excluded** — distributed as an XPPAUT `.ode` file, not an executable NEURON model; fails blueprint §6.2 |
| ModelDB 17664, 116835, 226010, 267306, 2017402, 267621 | Purkinje/granule/dopaminergic/STG alternates | downloaded and archived with provenance; **not included** in the confirmatory panel on simulation-cost grounds only |

No model was excluded, and none was included, on the basis of the result it
produced. Panel composition was frozen before the screening campaign
(`preregistration.md` §1).
