# COMPASS decision log

Design decisions that shaped the study, with the evidence that motivated each.
Blueprint WP0 deliverable ("decision log template") and §26 ("record failed
simulations and exclusions").

Format: **what was decided**, *why*, and what would have happened otherwise.

**This log is append-only and entries are not rewritten.** An entry records the
decision *and the numbers that motivated it at the time*. Where a later repair
changed those numbers, the entry carries a `> **Superseded:**` line pointing to
the entry that revised it — the reasoning stays as it was, but no reader should
mistake a historical figure for a current one. **D31 tabulates every superseded
value in one place**; `docs/RESULTS.md` and `README.md` always carry the current
ones, and `tools/verify_report_numbers.py` fails the build if they drift.

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

> **Superseded (D31):** recomputed against the corrected conditioning targets
> these are **1.87x** and **2.77x**, with p = 0.195 and 0.078 and the matched
> CI starting at 1.03 — both now inconclusive. The design correction described
> here was real; the strength of the result it produced was overstated.

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

> **Superseded (D31):** after the stale Sobol campaigns were re-run (D25) the
> mean over the same 24 quotable analyses is **0.375**.

## D18 — The reported "ICC" was not an ICC, and it no longer votes

`np.var(m.fittedvalues)` was used as the fixed-effect variance, but statsmodels'
MixedLM `fittedvalues` already contain the random-effect predictions, so the
between-family variance entered the denominator twice. The published **0.048**
was neither the adjusted ICC nor the variance-partition coefficient.

Corrected: fixed-effect variance is computed from `exog @ fe_params` alone. The
variance-partition coefficient is **0.050**; the standard **adjusted ICC is
0.289** — six times the published figure.

> **Superseded (D31):** on the corrected conditioning targets the adjusted ICC
> is **0.165** and the VPC **0.023**. The arithmetic fix stands; the values do
> not.

The criterion has also been **withdrawn from the decision table**, independently
of the arithmetic. The fixed covariates conditioned on are regime, geometry and
sample size, which *are* the H-G hypothesis; regressing them out and then asking
whether the residual favours H-B tests H-B against a model already handed the
alternative. The value is reported descriptively in `mixed_effects.json` and
flows through a `note()` path that the tally cannot read.

Guarded by `tests/numerical/test_variance_decomposition.py`.

## D19 — The decision tally moves for procedural reasons; the label is demoted

Withdrawing the ICC criterion moved H-B from **4/6 to 4/5** — a failing
criterion was removed, so the pass rate rose from 0.67 to 0.80 with no number
about neurons changing. Earlier, the random-control criterion was **added after
preregistration**, once the corrected battery's result was known, raising both
the H-B denominator and its pass count.

Each change was defensible alone. Together they establish that a five-criteria-
per-side vote has no resolution finer than 0.2 and is sensitive to post-hoc
choices. `RESULTS.md` §12.1 now states this, and the H-G/H-B/H-M label is
presented as a summary of the criteria table rather than as a result. The
load-bearing content is the individual effect sizes.

## D20 — Effects below the readout's measurement floor are not reported as effects

eta is quoted in ensemble-SD units, which hides an absolute floor: the primary
readout is a spike **count** over a 400 ms step, so one extra spike is an
irreducible 2.5 Hz regardless of dt. `tools/readout_resolution.py` now computes
each model's floor in its own SD units and writes `readout_resolution.json`.

Consequence: the rate readout resolves the effect in 6 of 8 models, but **not in
`bbp_l5pc` (0.023 SD vs a 0.074 SD floor) or `mainen_l5pc` (0.050 vs 0.073)** —
the two multicompartment cortical reconstructions. Four more sit within a factor
of two of their floor, and the panel mean is dominated by `hh_cable` (1.280) and
`hh_point` (0.919). The functional-validation clause of H-M is therefore
supported for the toy models and unsupported for the models with real morphology.

> **Superseded (D31):** on the corrected targets the rate readout resolves the
> effect in **4 of 8** models, not 6. `khaliq_purkinje` and `pospischil_rs`
> also fall below their floors, and the largest median |eta| is `hh_point` at
> 0.717 SD (not `hh_cable` at 1.280).

## D21 — Module identity is readout-dependent; this is the project's central result

On identical simulations, identical tuples and identical perturbations,
r(Omega, |eta|) is **−0.452** (p = 1.3e-8) for mean firing rate, **−0.332** for
first ISI, **−0.032** for AP half-width and **−0.003** for AP amplitude. The
link between statistical synergy and functional non-additivity is strong for
spike-timing readouts and absent for waveform readouts. Pooling all four gives
−0.221, describing none of them.

Two explanations remain confounded: readout *representation* (different readouts
are sensitive to different parameters) and *counting nonlinearity* (threshold
crossing is itself non-additive, so any tuple looks non-additive in a counted
readout). `RESULTS.md` §14.1 specifies the discriminating experiment — repeat
selection and perturbation under disjoint waveform-only and rate-only
conditioning vectors, plus a graded non-counted rate readout as a third arm.
Until it is run, module claims are stated with the readout named.

## D22 — Pipelines must fail loudly

`run_pipeline.sh` and `run_phase2.sh` ran under `set -u` with 8 and 13 stages
piped through `tail`. The shell therefore saw `tail`'s exit status, so a stage
could die and the pipeline would run every downstream stage on stale inputs and
print "complete". Both now use `set -euo pipefail`, tee each stage to
`results/reports/*_logs/<run-id>_<stage>.log`, and propagate the real exit
status; the "no failed simulations" check exits non-zero instead of only
printing a count.

## D23 — The repressilator had an inert parameter, and it was hiding a sign flip

The protein equation read `-KP * (p - m) * 0 + BETA * (m - p)`. The `* 0` made
`k_p` an inert coordinate: it was drawn from the prior, entered every tuple
statistic, and changed nothing. The "six-parameter" non-neuronal comparator had
five active parameters and one nuisance axis diluting every quantity computed
from it.

With `k_p` active the comparator's behaviour changes qualitatively. The previous
table reported **+0.0697** at the very-strict tolerance and the prose described
the gene circuit "flipping to redundancy" as the accepted set became a thin
clustered sheet. There is **no flip**: Ω is monotone negative,
−0.0018 → −0.0112 → −0.0681 → **−0.183**, with effective rank falling 6.00 →
5.06. The reversal narrative is withdrawn.

Separately, both README and `RESULTS.md` quoted a bare **−0.068**, which is the
*strict* value, as if it were the comparator's result. Ω spans two orders of
magnitude across tolerances here; a single figure without its tolerance is not
a number anyone can check. Now quoted as a range.

Guarded by a parameter-influence test in `tests/numerical/` that fails if any
comparator parameter leaves every feature unchanged.

## D24 — One authoritative solver contract

`tools/canonical_audit.py` took its integrator settings from
`data/provenance/numerical_qc.json` (`chosen_kwargs`), the first-pass QC sweep,
which selected **CVODE at dt = 0.025 ms** for six of eight models. The campaign
runs **fixed-step at per-model time steps** from `CAMPAIGN_SOLVER`. The
repository therefore held two contradictory solver contracts, and the canonical
manifests — which define the phenotype target every ensemble is accepted
against — were built under the integrator the campaign does not use.

This is not a bookkeeping detail. A target produced by an adaptive-step
integrator anchors ensembles simulated at a fixed step, so "accepted" meant
"close to a phenotype this model never produces under its own campaign
settings".

`CAMPAIGN_SOLVER` is now the single source of truth and `canonical_audit`
raises rather than guessing for an unlisted model. `numerical_qc.json` is
retained as the historical record of the first pass and is no longer consulted
for production settings.

A second staleness was found in the same layer: the `mainen_l5pc` manifest
recorded **754 compartments**, the pre-refinement "no compartment longer than
50 µm" mesh, after the adapter had been changed to refine to twice the
d_lambda(100 Hz) mesh (**2,304**). Every geometry-derived covariate in the
cross-model analysis read the stale count.

Guarded by `tests/unit/test_solver_contract.py`, which compares the stored
artefacts — not the code that writes them — against `CAMPAIGN_SOLVER` and
against the mesh the adapters actually build.

**Measured impact.** Rebuilding under the campaign contract moved every model's
canonical spike count *downward*, by 17–24%: `pospischil_fs` 81 → 64,
`hh_point` 39 → 31, `hh_cable` 41 → 33, `khaliq_purkinje` 67 → 54,
`pospischil_rs` 41 → 34, `mainen_l5pc` 17 → 13. A one-directional bias of that
size is not scatter — the old target described a cell firing about a fifth
faster than the campaign's own integrator produces.

Most features were untouched: the median shift is 0.000–0.018 E1-SD and only
3 of 96 feature-model pairs exceed 0.5 E1-SD. The effect is concentrated in
spike count and quantities derived from it. What makes it matter is the
comparison with the conditioning tolerances (3.0 / 2.0 / 1.0 / 0.5 z): a
0.44–0.53 SD target error is ~15% of the permissive tolerance and comparable to
the *entire* very-strict tolerance.

Re-deriving the ensembles confirms exactly that pattern. Acceptance at the
permissive tolerance is unchanged to three decimals (`hh_cable` 0.335,
`hh_point` 0.321, `pospischil_fs` 0.811), while the tight tolerances move:
`hh_cable` E4b 0.024 → 0.036, `hh_point` E4b 0.013 → 0.019, `pospischil_fs` E4
0.089 → 0.108 and it now yields a very-strict ensemble where it previously
yielded none. The conclusions most affected are therefore the ones drawn at the
tightest tolerances — already the thinnest on surviving models (D-note in
RESULTS 6.1).

## D25 — A cached simulation is only reusable if its identity is recorded

`run_sobol.py` decided whether to reuse a stored shard with
`if not out.exists()`. Nothing else. When `hh_point`, `khaliq_purkinje` and
`bbp_l5pc` were re-simulated at corrected time steps for the screening
campaign, their Sobol shards — produced under the old settings — were silently
kept. Nothing failed; the Sobol indices were simply computed on a different
model than the one the rest of the project describes.

Each shard now carries a sidecar `S_XXXXXX.identity.json` recording the solver
contract, the design (kind, n_base, seed, second-order flag, block size), the
feature list, and a **simulation-scoped** code hash. The hash deliberately
covers only `adapters`, `protocols`, `features`, `sampling` and
`workflows/simulate.py`: hashing all of `src/` (as the campaign manifest does)
would invalidate every stored simulation whenever a plotting or statistics
module changed, which is an incentive to stop stamping.

Shards whose identity does not match are **moved to
`results/features/sobol/_superseded/<timestamp>/`, never deleted** — overwriting
an artefact is how the Mainen convergence evidence was lost (D26).

**Existing shards were not stamped retroactively.** Writing a stamp onto an
artefact whose provenance cannot be read from the artefact asserts precisely the
thing in question. `--verify-stored` instead re-simulates one real block per
model under the current code and compares it bit-for-bit; a stamp is written
only where the stored data reproduce exactly, and a model that differs has its
whole design re-run.

## D26 — The Mainen convergence evidence was overwritten by a later run

`src/compass/adapters/models.py` justifies refining the Mainen mesh with: "Twice
the d_lambda(100 Hz) mesh at dt = 0.0125 ms reproduces all 13 conditioning
features on all 6 samples. See `convergence_audit_mainen_l5pc.json`." The file
it cites contained only `as_built` settings, **all three rejected**, with
`chosen: null` — a later `--mesh as_built` invocation had overwritten the run
that produced the evidence.

The artefact was also stale in a second way: it recorded `as_built` as 754
segments, from before the adapter's own refinement made `as_built` mean 2,304.

Re-run against a reference that is finer in **both** time and space
(dt = 0.00625 ms at 4× d_lambda), testing the production combination explicitly.
The general lesson is D25's: audit outputs are evidence, and evidence must be
appended, not replaced.

## D27 — Mainen is not spatially converged, and the original audit could not have found out

Re-running the Mainen audit against a reference refined in **both** time and
space (dt = 0.00625 ms at 4× d_lambda, 4,440 segments, 75 min on 7 cores) gives
a worse answer than the overwritten artefact claimed, for a structural reason.

**The original audit only ever tested time.** Its reference was dt = 0.00625 ms
at the *same* 2,304-segment mesh as production. A same-mesh reference cannot
detect spatial error, so "twice the d_lambda mesh reproduces all 13 features"
was never a test of the mesh — it was a test of the time step at that mesh.

Against a properly refined reference:

| setting | segments | mean agreement | failing features |
|---|---:|---:|---|
| dt 0.0125, d_lambda mesh | 1,066 | 0.824 | 6 of 13 |
| dt 0.025, d_lambda mesh | 1,066 | 0.824 | 6 of 13 |
| **dt 0.0125, production** | **2,304** | **0.989** | **`adaptation_index` only** |
| dt 0.025, production | 2,304 | 0.989 | `ap_amplitude_mV` only |

Two conclusions, and they point opposite ways.

*The refinement was strongly justified.* Going from the d_lambda mesh to twice
it moves agreement from 0.824 to 0.989 and clears five of six failing features.
The adapter is right to refine.

*It is still not converged.* `chosen` is **null**: at production, one of the 13
conditioning features fails to reproduce on one of seven prior samples. The
cause is a single threshold-adjacent spike — `adaptation_index` is a ratio of
ISIs, and losing one spike from a 9-spike train flips it from −0.045 to +0.068.

`adaptation_index` is a conditioning feature for every model, so Mainen's
accepted ensemble is defined partly by a quantity that is not mesh-converged
for it. This is recorded as a limitation rather than repaired: refining Mainen
to 4,440 segments would roughly double the cost of the most expensive model in
the panel, and the same threshold-adjacent behaviour would reappear at any mesh.
It is also the second independent reason to distrust this particular feature —
it was already withdrawn from the Sobol results for mean-imputation bias (D17).

## D28 — A convergence tolerance may not be finer than the readout's quantum

`ABS_TOL["step_abs_hi.mean_firing_rate_Hz"]` was **2.0 Hz**. Firing rate here is
a spike count over a 400 ms window, so its quantum is **2.5 Hz**. The criterion
demanded a precision the measurement cannot express, and no mesh or time step
can satisfy it: one threshold-adjacent spike scored as a convergence failure at
every rate below 40 Hz. The identical difference expressed as `spike_count`
(tolerance 1.0, one spike) scored as agreement — the same physical event was
simultaneously a pass and a fail depending on its units.

Raised to exactly one quantum, from the protocol geometry, before looking at any
model's result. **It does not change the verdict**: `mainen_l5pc` still fails,
now on `adaptation_index` instead of firing rate. That is the check that this
was a correction and not a route to a wanted answer — compare D19, where two
defensible criterion changes each moved the H-G/H-B tally.

Two supporting changes:

* `convergence_audit.py` now **supersedes rather than overwrites**: an existing
  artefact is moved to `data/provenance/_superseded/` with its write timestamp.
  This is the direct fix for the loss described in D26.
* `--rescore` re-evaluates a stored audit under current tolerances using its
  saved feature rows. The 75-minute Mainen run did not have to be repeated to
  apply this correction, which is precisely why the raw rows are stored.

## D29 — The manifests were stale because the tool that builds them was broken

Fixing the solver contract (D24) was not enough to regenerate the manifests:
`tools/canonical_audit.py` died after the first model.

Its `main()` had always looped over `MODEL_REGISTRY` building every model in one
process. That was fine until the adapters gained a guard against cross-model
section contamination — NEURON keeps one global section list per process, and
models loaded from published `.hoc` morphologies discover their own sections by
differencing `h.allsec()`, so a second model built in the same process mixes the
lists. The guard was correct and necessary. It also, silently, made the manifest
builder unable to finish.

So the canonical manifests were not stale because nobody thought to re-run the
tool. **They were stale because the tool could not complete**, and had not been
able to since the guard landed. That is why CVODE-era manifests survived every
subsequent correction to the campaign: each fix regenerated what it could reach,
and this artefact was unreachable.

Fixed by auditing one model per spawned process, matching what
`convergence_audit.py` and the campaign workers already do.

The general lesson is the one D25 states for cached data, applied to code: a
regeneration path that is never exercised end-to-end is not a regeneration path.
`tests/unit/test_solver_contract.py` now checks the stored artefacts against the
contract, so the next time this breaks the test fails rather than the artefact
quietly ageing.

## D30 — Every "sign reversal" in the conditioning results was sampling noise

Correcting the conditioning targets (D24) admitted a very-strict ensemble for
`pospischil_fs` of **66 samples in 7 dimensions** whose Δ net O-information is
**+0.0152** — opposite in sign to the two well-resolved models at that tolerance
and large enough to flip the panel mean from −0.0060 to +0.0020.

The question "is that a real reversal or noise?" must not be settled by choosing
a cutoff that excludes it. `tools/estimator_reliability.py` measures the
estimator instead, by subsampling ensembles that *are* large enough down to
matched n — a procedure that never sees the disputed value. At n = 66 this
statistic flips sign in up to **33%** of resamples of ensembles whose full-sample
value is clearly negative, and reaches **+0.0066** by chance. The disputed value
sits inside that distribution.

Extending the analysis to a per-model bootstrap at each ensemble's own size gives
the result that matters:

* **Five of eight models resolve the conditioning effect at 4–5σ** at the
  permissive tolerance. The three that do not (`migliore_ca1` 140,
  `mainen_l5pc` 153, `bbp_l5pc` 321 samples) are sample-size limited, not effect
  limited, and their direction still agrees.
* **The estimator is unbiased under this procedure**: E1 gives
  −0.0000000 ± 0.000009 and the shuffled E8 null +0.000008 ± 0.000025 over eight
  models. Negative shifts are not manufactured by conditioning.
* **Where the effect is resolvable it is monotone.** `hh_cable` is significant at
  all four tolerances (5.1 → 8.1 → 8.9 → 5.3σ) with magnitude rising 17-fold.
* **All four positive entries in the panel are ≤2.0σ.** The earlier correction
  claiming "three models reverse sign, including both cortical reconstructions"
  described scatter around zero and is itself withdrawn.

This supersedes both the original "monotonically" claim and the first correction
to it. Resolvable coverage falls 5 → 6 → 2 → 1 across the tolerances, so the
panel-level dose-response is still not estimable — but for a sharper reason than
"models disagree": at tight tolerances most models cannot measure anything.

**No minimum-sample threshold was applied retroactively.** The conditioning path
has no analogue of Sobol's `VALID_FRACTION_FLOOR`, which is a real gap and is
recorded as one. Introducing a floor now, having seen which values it would
remove, is the failure mode of D19; it belongs in the next preregistration, not
in this analysis.

## D31 — Every number superseded by the C1–C4 repair, in one place

The repair of the four audit blockers changed results as well as provenance.
Correcting the conditioning targets (D24) moved the accepted ensembles that
almost everything downstream is computed from, and re-running the three stale
Sobol campaigns (D25) changed the sensitivity indices. This is the single index
of what moved; the entries above keep their original figures with a
`> **Superseded**` pointer here.

| quantity | earlier | current | why it moved |
|---|---:|---:|---|
| perturbation, top vs matched | 2.34× (p = 0.109) | **1.87× (p = 0.195)** | corrected targets → new E3 backgrounds |
| perturbation, top vs random | 4.05× (p = 0.023) | **2.77× (p = 0.078)** | as above |
| functionally supported (top) | 25.0% | **20.8%** | as above |
| r(Ω,\|η\|), primary readout | −0.452 | **−0.401** | as above |
| models resolving the rate readout | 6 / 8 | **4 / 8** | as above |
| largest median \|η\| | `hh_cable` 1.280 | `hh_point` **0.717** | as above |
| Sobol mean interaction fraction | 0.363 | **0.375** | 3 stale campaigns re-run |
| `bbp_l5pc` interaction fraction | 0.482 | **0.538** | as above |
| adjusted ICC | 0.289 | **0.165** | corrected targets |
| variance-partition coefficient | 0.050 | **0.023** | corrected targets |
| unique-R² ratio, size-matched | 8.5× [3.0, 44.2] | **9.3× [3.2, 54.2]** | corrected targets |
| LOMO median R² | 0.694 | **0.818** | corrected targets |
| LOMO worst fold (`migliore_ca1`) | −4.468 | **−0.748** | corrected targets |
| E8 naive synergy fraction | 0.437 | **0.454** | corrected targets |
| within/between regime variability | 2.7× | **1.8×** | corrected targets |
| direction agreement (scored) | 1.000 | **0.875** | one 2.0σ, 66-sample ensemble (D30) |

Two patterns are worth naming rather than leaving implicit.

**The functional-validation evidence weakened; the readout split did not.** The
enrichment ratios fell by 30–40% and both paired tests became non-significant,
but the difference between spike-timing and waveform readouts survived almost
unchanged (r = −0.401/−0.278 versus −0.091/+0.063). The finding that depends on
comparing readouts is more robust than the finding that depends on comparing
tuples to controls.

**Transfer improved.** LOMO median rose 0.694 → 0.818 and the worst fold
improved six-fold. That is the expected signature of removing a per-model error:
a target computed by the wrong integrator is a model-specific offset, and a
held-out fold cannot learn model-specific offsets. It is weak corroboration that
the C2 repair fixed something real rather than merely relabelling it.

---

## Exclusions

| asset | intended slot | decision |
|---|---|---|
| ModelDB 97747 (Golomb FS interneuron) | FS interneuron replication | **excluded** — distributed as an XPPAUT `.ode` file, not an executable NEURON model; fails blueprint §6.2. Its slot is now filled by ModelDB 237595 (Tzilivaki FS basket cell). |
| ModelDB 17664 (Miyasho Purkinje) | Purkinje replication | **included** in the `screening_v2_replication` cohort (D31) |
| ModelDB 267306 (Moubarak SNc DA) | first dopaminergic lineage | **included** (D31) |
| ModelDB 226010 (Dougalis vlPAG/DRN DA) | second dopaminergic lineage | **included** (D31) |
| ModelDB 266901 (Tomko CA1) | CA1 implementation replication | **included** as a nested implementation of the Migliore lineage (D31) |
| ModelDB 237595 (Tzilivaki FS basket) | FS replication | **included** (D31) |
| ModelDB 2017402 (Ortner SNc DA) | DA implementation replication | **adapted, audited, NOT simulated** — fails gate N3 at every tested time step and yields ≤3 spikes in the 400 ms window (D33) |
| ModelDB 20212 (Poirazi CA1) | independent CA1 lineage | **excluded** — the published biophysics is only expressible through the package's own `ExperimentControl.defvar()` harness; adapting it would import a large third-party experiment scaffold across the adapter boundary (D33) |
| ModelDB 116835 (Diwakar granule) | cerebellar granule class | **excluded** — the canonical channel distribution is applied by the GUI-only `Parametri.hoc`; the bare template rests at −32.7 mV instead of −70 mV (D33) |
| ModelDB 267621 (Zang & Marder STG LP) | invertebrate stress test | downloaded and archived with provenance; **not evaluated**, deferred on cost |

No model was excluded, and none was included, on the basis of the result it
produced. Panel composition was frozen before the screening campaign
(`preregistration.md` §1).

---

# Round 3 — 2026-08-06 post-revision technical and scientific audit

Five blockers were raised (`COMPASS_POST_REVISION_TECHNICAL_SCIENTIFIC_AUDIT.md`).
Each decision below states what was chosen **and what was rejected**, because in
three of them the cheaper option was to keep the result and add a caveat.

## D31 — Extend the panel from six source lineages to ten

*Problem.* Eight `model_id` values represented six independently developed
sources. `hh_point`/`hh_cable` are one HH lineage; `pospischil_rs`/`pospischil_fs`
one Destexhe framework. Every fold, bootstrap and sign test treated eight as
eight. Three biological classes had exactly one lineage, so no class-specific
claim was supportable even in principle.

*Decision.* Add a separately versioned `screening_v2_replication` cohort of five
models chosen for **replication structure**, not breadth:

| model | lineage | fills |
|---|---|---|
| `miyasho_purkinje` (ModelDB 17664) | `miyasho_2001` | second Purkinje lineage, with `khaliq_2003` |
| `tzilivaki_fsbc` (ModelDB 237595) | `tzilivaki_2019` | second fast-spiking lineage, with `pospischil_2008` |
| `moubarak_snc_da` (ModelDB 267306) | `moubarak_2022` | first dopaminergic lineage |
| `dougalis_vlpag_da` (ModelDB 226010) | `dougalis_2017` | second dopaminergic lineage |
| `tomko_ca1` (ModelDB 266901) | `migliore_1999` | second CA1 *implementation* |

Ten lineages across fourteen implementations. Lineage attribution is
deliberately **conservative**, because the failure being corrected is
overstatement: `ortner_snc_da` shares an author's NEURON code with
`dougalis_vlpag_da` and is filed as one lineage with two implementations, and
`tomko_ca1` inherits Migliore's channel kinetics and is filed as a nested
implementation rather than as an independent CA1 source.

*Rejected.* Counting every new `model_id` as a new family, which would have
reported fourteen "independent models" — the exact inflation the audit
identified.

## D32 — Exclude `mainen_l5pc` from primary inference rather than footnote it

*Problem.* At its exact production setting (dt = 0.0125 ms, 2,304 segments)
Mainen reproduces 12 of 13 conditioning features on all seven audit points and
fails `adaptation_index` on one. The previous revision characterised this
accurately and kept the model in every headline number.

*Decision.* Exclusion, enforced in code. `compass.analysis.cohorts.PRIMARY_EXCLUSIONS`
drives every analysis, and `tests/integration/test_convergence_contract.py`
fails if a model listed as an N3 failure is *not* excluded — and also fails if a
listed failure starts passing, so a stale exclusion is caught as readily as a
stale inclusion. A prespecified sensitivity cascade
(`decision_sensitivity.json`) reports six arms: primary, Mainen retained, the
failing feature dropped from the phenotype for **every** model, partial-coverage
retained, all models, and discovery/replication separately.

*Rejected.* A publication-wide footnote. The audit's release rule 7 is explicit:
"an exception requires exclusion of the affected model/data from primary
inference and regeneration of dependent artifacts; it does not convert the gate
to a pass."

## D33 — Reject two candidate models and one adapted model on frozen gates

`poirazi_ca1` (ModelDB 20212) and `diwakar_granule` (ModelDB 116835) were
fetched, built and rejected on gate 9.5.2 and 9.5.4 respectively;
`ortner_snc_da` (ModelDB 2017402) was fully adapted, its mechanisms compiled and
its convergence audit run, and then **not simulated** because it fails gate N3
at 0.025, 0.0125 and 0.00625 ms and produces ≤3 spikes in the frozen 400 ms
window. All three exclusions were decided before any information endpoint was
computed for them, and all three are recorded with their evidence
(`EXCLUDED_CANDIDATES`, `convergence_audit_ortner_snc_da.json`).

Rejecting a model *after* paying to adapt it is the outcome an outcome-blind
gate is supposed to be able to produce. If a gate never rejects anything that
cost something, it is not doing any work.

## D34 — Withdraw the 2.5 Hz "measurement floor" rather than reinterpret it

*Problem.* The claim compared a median over six tuple means of eight backgrounds
against one unaveraged single-observation quantum, and treated a deterministic
grid as a stochastic detection limit.

*Decision.* Full withdrawal of the analysis and of all four conclusions drawn
from it, with the tool and its outputs preserved under
`results/analyses/_withdrawn/` so the record survives. The replacement separates
numerical repeatability (rerun at half dt, compare per-background η) from
inferential precision (bootstrap and exact sign-flip inference over backgrounds,
and exact enumeration of all 2⁸ sign flips for the integer spike count).

*Rejected.* Rescaling the floor by √8 or by the number of backgrounds. That
would have preserved the framing while fixing the arithmetic, and the framing is
what is wrong: quantisation is not noise.

## D35 — Fail closed on shard identity, and backfill rather than re-simulate

*Problem.* Screening reused a shard because its path existed. Nothing checked
code, solver, mechanism build, configuration or design.

*Decision.* Reuse now requires an exact match of a computed job identity
(simulation-relevant code hash, per-job configuration hash, external asset hash,
compiled mechanism hash, environment hash, solver contract, feature schema
version, protocol hash, seed, prior and row count) plus a checksum match.
Non-matching shards are **quarantined, never overwritten**.

Two scoping decisions matter and were not obvious. The identity uses a
*simulation-relevant* code hash rather than all of `src/`, and a *per-job*
configuration hash rather than the whole config file — otherwise editing a
plotting module or correcting a note in an unrelated campaign entry invalidates
3,057 correct shards, and a validation rule that cries wolf gets switched off.

Existing shards were **backfilled** with reconstructed identities stamped
`identity_provenance: "backfilled"`, not re-simulated. A backfilled identity
asserts only "these fields describe the tree at reconstruction time", which is
why the deterministic replay gate exists and why backfilled shards are exactly
the ones it targets.

*Immediate payoff.* The check found that 425 `pospischil_rs` E1 rows (sample ids
925–1349) had been simulated from a **different Sobol design** than the other
7,767: seed 20262103 against 20262247. The stored E1 table was a mixture of two
scrambles and no artifact recorded it. Those 17 shards were quarantined and
regenerated.

## D36 — Report intrinsic dimension as a sensitivity covariate, not a measurement

The TwoNN implementation was biased upward by 45–50%. Fixing it was
straightforward; the interesting decision was what to do afterwards. Rather than
substituting a corrected number and continuing, a second estimator with
different bias behaviour (Levina–Bickel MLE) is computed alongside it, both
enter the hierarchical model's geometry block, and the estimator version is
stamped into every endpoint row so a stored covariate can never be silently
compared across algorithm versions.

*Rejected.* Trusting one estimator because it is now unbiased on Gaussian test
data. The covariate is load-bearing for the headline variance ratio; a single
algorithm's finite-sample behaviour on non-Gaussian accepted ensembles is not
something a synthetic calibration establishes.

## D37 — Split "valid" into six row-level fields

`simulation_status == "ok"` was doing the work of at least four different
questions. Rows now carry `execution_valid`, `protocol_valid`, `feature_valid`,
`physiology_class`, `analysis_eligible`, `validator_version` and
`failure_reason`. The distinction that motivated it: under broad Tier C/D priors
a finite but silent or depolarisation-blocked draw is a **prior-predictive
outcome** to be classified, while a numerically failed rheobase probe is
**corrupt** even though the surviving protocols would still yield a finite
feature row. The old single field could not tell those apart, and the rheobase
search actively confused them by returning `None` ("does not spike") for a
solver failure.

## D38 — What the new gates rejected, and what that cost

Four defects were found by gates rather than by reading, and each one cost
something real to act on. They are listed because a gate suite that only ever
confirms what you already believed is not evidence that the data is sound.

| found by | defect | action |
|---|---|---|
| identity-validated reuse | 425 `pospischil_rs` E1 rows (ids 925-1349) were simulated from a **different Sobol design** (seed 20262103 vs 20262247), so the stored E1 table was a mixture of two scrambles | 17 shards quarantined and regenerated |
| gate D3 | `cv_isi` was **manufactured as 0.0** on 2,127 rows: two spikes give one interval, whose standard deviation is exactly zero | extractor now requires 3 spikes; stored rows enumerated as documented non-load-bearing drift, with the "non-load-bearing" claim re-derived from the analysis code rather than trusted |
| gate N6 | `dougalis_vlpag_da`'s hyperpolarising step (-0.002 nA, from -0.5 x its 0.0039 nA rheobase) was **below the model's own baseline drift**: the membrane depolarised by +2.8 mV and input resistance came out at **-223 MOhm** | amplitude rule given a floor at -0.1 x the `hi` step; affected shards quarantined on `protocol_hash` and regenerated automatically |
| gate N6 | project and eFEL spike counts differ by 4 and 3 on the two spontaneously active models | diagnosed as a peak-vs-onset windowing convention on a continuously firing cell; recorded with the measured value so a *change* in the gap still fails |

## D39 — Scope the reuse criterion to semantic versions, not a byte hash

*Problem discovered while building D35.* Hashing the simulation source into the
shard identity meant that editing a comment in `extract.py` invalidated all
98,048 stored simulations. A validation rule that discards correct data on a
no-op change gets switched off, and then it protects nothing.

*Decision.* The reuse criterion is the set of **declared semantic versions** --
feature schema, solver contract, protocol hash, model-asset hash, compiled
mechanism hash, per-job configuration hash, seed, prior, row count and file
checksum. `sim_code_hash` is still written into every sidecar as
`recorded_only.sim_code_hash`, so drift stays visible and auditable, but it does
not by itself invalidate a shard.

*What replaces the lost detection.* An undeclared change to the extractor is now
a **code-review failure** rather than a data-discarding one:
`tests/unit/test_solver_contract.py` records the content digest of
`extract.py` against the declared `FEATURE_SCHEMA_VERSION` and fails if the
module changes without the version changing. The check moved earlier in the
pipeline, which is the direction you want it to move.

*Rejected.* Keeping the byte hash and re-running the campaign on every source
edit. That is correct in principle and unusable in practice, and an unusable
gate is worse than a scoped one because nobody can tell it has been disabled.

## D40 — The two highest-priority replication models could not be run

The audit's section 9.4 named an alternate Purkinje and an executable
independent fast-spiking model as the two most valuable additions. Both were
fetched, adapted, had their mechanisms compiled and their f-I curves calibrated,
and both then **failed gate N3**:

* `tzilivaki_fsbc` (ModelDB 237595) — `mean_firing_rate_Hz` reproduces on
  25-50% of audit points at dt = 0.05 and 0.025 ms, on the published
  9,875-segment mesh *and* on a 4,096-segment d_lambda mesh. A ~145 Hz cell puts
  one or two spikes per 400 ms window inside the tolerance, and resolving them
  needs a finer step than a 9,875-section axonal arbor allows within this
  budget.
* `miyasho_purkinje` (ModelDB 17664) — `ap_amplitude_mV` reproduces on 40% of
  points at 0.025 ms on both meshes; a finer-step audit was run and its outcome
  is recorded in `convergence_audit_miyasho_purkinje.json`.

*Decision.* Report the failure and its consequence rather than lowering the
tolerance or quoting the models anyway. The fast-spiking class therefore still
has one lineage, and the audit's Priority-2 recommendation is **partially
unmet** — not because the models are unavailable but because a
convergence-accepted campaign on morphologically detailed fast cells costs more
compute than this release had.

That is a substantive finding about the audit's own recommendation: "add six
lineages" is not a purely curatorial task. Two of the six candidates were
excluded on frozen technical gates before any endpoint was computed, one more
was excluded for protocol coverage, and of the remainder the detailed
morphologies are the expensive ones precisely because detail and stiffness go
together.

