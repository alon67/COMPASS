# COMPASS preregistration

**Frozen:** before the screening campaign was analysed. Only the canonical-model
audit, the pilot timing runs, and the estimator/benchmark validation had been
inspected at the time of writing. No endpoint, tolerance, tuple rule or decision
threshold below was chosen after seeing a screening result.

Blueprint reference: §20.2.

---

## 1. Model inclusion list (frozen)

| model_id | blueprint slot (§6.1) | source | lab / pipeline | free params |
|---|---|---|---|---|
| `hh_point` | minimal Hodgkin-Huxley point model | Hodgkin & Huxley 1952; NEURON built-in `hh` | Hodgkin-Huxley / independent | 6 |
| `hh_cable` | minimal HH, cable variant | as above, soma+dendrite | Hodgkin-Huxley / independent | 8 |
| `pospischil_rs` | cortical pyramidal (regular spiking) | ModelDB 123623 | Destexhe / independent | 8 |
| `pospischil_fs` | fast-spiking cortical interneuron | ModelDB 123623 | Destexhe / independent | 7 |
| `khaliq_purkinje` | cerebellar Purkinje | ModelDB 48332 | Raman / independent | 10 |
| `mainen_l5pc` | cortical pyramidal, multicompartment | ModelDB 2488 | Mainen & Sejnowski / independent | 10 |
| `bbp_l5pc` | cortical pyramidal, BBP lineage | ModelDB 139653 + BlueBrain/BluePyOpt | Blue Brain Project / **bbp** | 15 |
| `migliore_ca1` | hippocampal CA1 pyramidal | ModelDB 2796 | Migliore / independent | 8 |

**Replication structure (§6.4).** Two independently developed cortical pyramidal
models (`mainen_l5pc`, an independent 1996 implementation, vs `bbp_l5pc`, the
Hay/Blue Brain lineage optimised by the BBP pipeline) separate neuronal identity
from laboratory implementation. Two models from a single framework and lab
(`pospischil_rs`, `pospischil_fs`) separate cell class from implementation in the
opposite direction. Two morphological variants of the same channel set
(`hh_point`, `hh_cable`) isolate morphology (§13.3).

**Exclusions applied.** ModelDB 97747 (Golomb FS interneuron) was excluded before
use: it is distributed as an XPPAUT `.ode` file, not an executable NEURON model,
and therefore fails the §6.2 requirement for "executable source or a reproducible
simulator definition". ModelDB 17664, 116835, 226010, 267306, 2017402 and 267621
were downloaded and are archived with provenance but were not brought into the
confirmatory panel, on grounds of simulation cost only.

## 2. Parameter priors and evidence tiers

Every parameter is sampled as a **multiplicative scale factor on the model's
canonical per-segment value**, log-uniform, with the canonical model at scale
1.0. Absolute canonical values are retained in the parameter table.

| tier | models | basis |
|---|---|---|
| C - published optimisation range | `bbp_l5pc` | bounds derived from `BluePyOpt/examples/l5pc/config/params.json` |
| D - expert-defined plausible range | all others | default ×/÷4 for conductances; narrower for `cm`, `Ra`, reversal potentials and morphology |

**No parameter in this study is tier A or B.** No result may therefore be
described as measured biological variability (§7.2, §26). Results are stratified
by tier in the report.

## 3. Protocols (frozen)

Common core per model: `rest`, `hyperpol_abs`, `step_abs_lo`, `step_abs_hi`
(absolute family) and `step_rheo_1p5x`, `step_rheo_3p0x` (rheobase-normalised
family), plus a bisection rheobase search. Timing: 120 ms delay, 400 ms step,
550 ms total; rest 300 ms. Absolute amplitudes are pinned per model from the
canonical f-I curves measured in `results/manifests/canonical_*.json`.

The two protocol families are analysed **separately and never pooled** (§14.4).
`khaliq_purkinje` fires spontaneously, so its rheobase is 0 and the normalised
family is undefined for it; this is reported as a coverage limitation rather
than imputed.

## 4. Conditioning (frozen)

Target: the canonical model's own feature vector. Dispersion: the standard
deviation of the unconditioned E1 ensemble. Distance metrics: maximum absolute
z-score and weighted Euclidean (both reported; §15.2 forbids relying on max-z
alone). Tolerance series (§15.3), in z units:

| name | threshold | ensemble code |
|---|---|---|
| permissive | 3.0 | E3 |
| moderate | 2.0 | E3b |
| strict | 1.0 | E4 |
| very strict | 0.5 | E4b |

A sample with any non-finite conditioning feature is **rejected**, not imputed.

## 5. Ensemble regimes (frozen)

Simulated: E1 (independent Sobol prior), E2 (Gaussian-copula prior, ρ=0.6),
E6 (clustered/oversampled). Derived without further simulation: E3/E3b/E4/E4b
(conditioned), E3m (size-matched random subset of E3), E7 (high-covariance
subset of E3, matched on size and stratified by target distance), E8 (column
shuffle), E9 (Gaussian-copula resample).

## 6. Tuple rules (frozen)

Orders 3, 4 and 5. Exhaustive enumeration when the number of tuples ≤ 1200,
otherwise deterministic seeded sampling of 1200 tuples (seed 20260804). Every
order is accompanied by a family-wise surrogate distribution computed from 25
column-shuffled surrogates **evaluated over the same number of tuples**, so
extreme values are compared against a search of matched size.

## 7. Primary endpoints (§18.1)

1. `synergy_area`, `redundancy_area` — order-normalised mean negative / positive
   O-information across orders 3-5.
2. `frac_synergy_o3` — fraction of synergy-dominated order-3 tuples.
3. Shift in (1) and (2) from unconditioned E1 to each conditioned ensemble.
4. Shift from the size-matched diverse subset E3m to E7 / E6.
5. Variance in (1) attributable to sampling regime and statistical geometry
   versus model identity.
6. Agreement between statistical modules and direct perturbation non-additivity.

## 8. Estimators

Gaussian (covariance-based, the estimator used in the published literature) and
Gaussian-copula (rank-to-normal-score, invariant to monotone reparameterisation),
both with the analytic `log det` small-sample bias correction. A
Kozachenko-Leonenko k-NN estimator is used in the synthetic benchmark suite.

Conclusions that depend on the estimator are reported as outcome D (§29).

## 9. Null models

Column-shuffled surrogates per order (family-wise); E8 shuffled ensembles;
E9 copula-matched ensembles; size-matched random subsets (E3m) as the comparator
for E7.

## 10. Hierarchical model (§18.2)

Endpoint ~ sampling block (regime, tolerance, log N, acceptance rate)
+ geometry block (mean |r|, effective rank / D, intrinsic dimension / D,
participation ratio / D, PC1 fraction)
+ architecture block (n parameters, log n compartments, point-model flag)
+ model identity (fixed, for commonality analysis; random, for the mixed model).

Reported as commonality analysis (unique and shared R²), a mixed model with
model family as a random effect, and leave-one-model-family-out prediction.

## 11. Decision thresholds (§19.4) — frozen

| criterion | supports | threshold |
|---|---|---|
| cross-model agreement in the direction of the conditioning effect | H-G | ≥ 0.80 |
| cross-model agreement in the direction of the covariance effect | H-G | ≥ 0.80 |
| unique R²(sampling + geometry) / unique R²(model identity) | H-G | ≥ 2.0 |
| out-of-sample R² for a held-out model family, sampling+geometry predictors | H-G | > 0 |
| sampling regime classified across unseen families vs baseline | H-G | > 1.5 × baseline |
| model identity classified from information summaries vs baseline | H-B | > 1.5 × baseline |
| intraclass correlation of model identity (mixed model) | H-B | ≥ 0.50 |
| top-tuple module stability across estimators (mean Jaccard) | H-B | ≥ 0.50 |
| fraction of top-synergy modules functionally supported by perturbation | H-B | ≥ 0.50 |
| top-synergy vs control tuple non-additivity ratio | H-B | > 1.5 |

Module functional support requires mean |η| ≥ 0.25 ensemble SD **and** sign
consistency ≥ 0.75 across ≥ 8 background points.

**Conclusion rule.** H-G if ≥60% of H-G criteria pass and <40% of H-B criteria
pass; H-B in the mirror-image case; H-M if ≥50% of H-G criteria pass alongside
≥40% of H-B criteria; otherwise inconclusive.

## 12. Declared deviations from the blueprint

These are limitations of scale, declared in advance, not changes of endpoint.

1. **Scale.** The blueprint envisages 10⁷ simulations on a cluster. This
   implementation runs ≈10⁵ simulations on one 14-core workstation. Sample sizes
   are therefore per-model and unequal (8192 for cheap point models down to 512
   for `migliore_ca1`); every cross-model endpoint is additionally recomputed on
   a size-matched subsample.
2. **`migliore_ca1` time step.** The convergence audit accepts only dt = 0.0125 ms
   for this model; it is run at 0.025 ms because 0.0125 ms would cost more wall
   time than the other seven models combined. The step is identical for every
   sample, so within-model dependence structure — the object of study — is
   unaffected; absolute rate features carry a small systematic bias.
3. **`migliore_ca1` discretisation.** Its as-distributed segmentation is not
   spatially converged; it is run on the d_lambda(100 Hz) mesh instead.
4. **No tier A/B priors, no per-cell targets.** E5 (per-cell conditioning) is not
   executed: no cell-level feature distributions were obtained for these models.
   Conditioning is on the canonical model's own phenotype.
5. **Sobol is not run for `migliore_ca1`** (design size × per-sample cost).
6. **Confirmatory scale** in the blueprint's sense (50,000-100,000 effective
   samples per condition) is not reached. This release is therefore reported as
   a **screening** study, and its conclusions are stated with that scope.
7. **E2 and E6 coverage is 6 of 8 models.** `mainen_l5pc` and `migliore_ca1` run
   the E1 ensemble only, on measured-throughput grounds (decision log D11).
   Because every conditioned and null ensemble is *derived* from E1, all four
   primary endpoints and both primary contrasts keep full 8-model coverage; only
   the secondary covariance-prior and clustered-prior contrasts are reduced.
   Decided while the campaign was running and before any cross-model result was
   inspected.

---

## Addendum 3 — 2026-08-06 audit response (registered before the replication cohort was analysed)

The following were fixed in advance of any analysis of the
`screening_v2_replication` cohort. They are recorded here, not only in the
decision log, because they change what the preregistered criteria *mean*.

1. **The unit of cross-model inference is `source_lineage_id`, not `model_id`.**
   Direction-agreement, leave-one-out folds, cluster bootstraps and random
   effects all group on lineage. The frozen discovery cohort has eight
   implementations over six lineages; the panel as a whole has fourteen over
   ten.
2. **Gate N3 is binding.** A model whose exact production `(dt, segment count)`
   is not a directly tested, accepted row of its own feature-level convergence
   audit is excluded from primary inference. Exclusions are listed in
   `compass.analysis.cohorts.PRIMARY_EXCLUSIONS` and are enforced by a test that
   also fails if a listed model starts passing.
3. **A prespecified sensitivity cascade accompanies every exclusion.** Six arms:
   primary, `retain_mainen`, `drop_failed_feature` (the failing feature removed
   from the phenotype for every model), `retain_partial_coverage`, `all_models`,
   and discovery/replication separately.
4. **Two decision criteria are withdrawn and become descriptive**: the
   Gaussian-vs-copula module Jaccard (both are Gaussian estimators on the same
   samples) and the model-identity classifier (random within-table splits of one
   campaign's derived rows). The two perturbation control ratios are collapsed to
   **one** scored criterion, because they are the same measurement with different
   denominators.
5. **Sobol quotability requires completeness AND precision AND physical
   plausibility**, not completeness alone.
6. **New models enter a new campaign namespace.** `screening` is frozen; adding a
   model to it is rejected by `tools/run_campaign.py`.
7. **Inclusion gates for the replication cohort were frozen before any
   information endpoint was computed** for any candidate: source and licence
   recorded, executes without importing third-party experiment scaffolding, all
   sampled parameters active (gate N4), canonical behaviour inside frozen
   source-justified bounds (gate N1), exact production setting convergence-accepted
   (gate N3), and common protocol coverage defined with explicit missingness.
   Three candidates failed these gates and were excluded; two of them after their
   mechanisms had been compiled and their convergence audited.

---

## Addendum 4 — 2026-08-09 independent audit response (POST HOC; not preregistered)

Everything below was decided **after** the results it acts on were known. It is
recorded separately from Addendum 3 for that reason, and nothing here should be
read with the weight of a rule fixed in advance.

1. **Item 7 of Addendum 3 was not enforced as written.** It lists "all sampled
   parameters active (gate N4)" among the inclusion gates frozen before any
   information endpoint was computed for a replication candidate. `tomko_ca1`
   was admitted to the replication cohort and **fails N4** on `gkdr_soma`,
   `gcal_soma` and `gcat_soma`. `bbp_l5pc`, a discovery-cohort model that
   predates that gate, fails it on `gskv3_apical` and `gcalva_apical`. Neither
   was excluded. This is a departure from a registered inclusion rule, it was
   found by an external reviewer rather than by us, and it is not repaired by
   relabelling: the correct action under Addendum 3 item 2's own logic would
   have been exclusion. What is done instead is stated in item 2.

2. **The N4 failures are handled by arms, not by exclusion.** The primary cohort
   retains both and is relabelled **N3-passing, N4 unresolved** wherever it is
   described. Two arms are added: `n4_strict` (both excluded — the headline
   under N1/N3/N4 enforcement — not "full N1–N7", since N2 is evaluable
   only for the six models with a sourceable published scalar) and the primary
   arm itself, which now retains both with their
   unidentifiable parameters projected out of the sampled space before ensembles
   are built). The projection is outcome-blind in the sense that the dropped set
   comes from a gate-N4 measurement made before any endpoint was inspected; the
   *decision to run it* is not.

3. **Both perturbation-derived H-B criteria acquire a numerical precondition.**
   They are scored only on implementations whose per-background η is repeatable
   at half the campaign time step, and are **not evaluable** — reported, casting
   no vote — if the repeatable subset spans fewer than three lineages. The
   repeatability rule itself (median |Δη| ≤ 0.05 SD and ≥ 80% sign agreement)
   *was* prespecified, in `tools/perturbation_inference.py`, before the
   half-step run. The minimum lineage count is new, and 3 is chosen because the
   criterion is scored on a lineage-clustered bootstrap interval that cannot be
   estimated from two clusters.

4. **Cluster bootstraps weight lineages equally.** Resampling lineages while
   pooling their implementations' rows gave two-implementation lineages double
   weight. Statistics are now formed within lineage and then averaged across
   lineages. This changed a published significance claim: the paired
   perturbation tests move from p = 0.047 and 0.016 over seven implementations
   to p = 0.188 and 0.063 over five lineages, and **the claim that both are
   below 0.05 is withdrawn**. With five lineages the smallest attainable
   two-sided Wilcoxon p is 0.0625, so this panel cannot produce a significant
   paired result at the lineage level at any effect size.

5. **An arm that selects the same rows as another arm is not independent
   evidence.** Duplicate arms are detected from the data and labelled; the
   cascade reports how many arms are substantively distinct.

6. **`preregistered: false` is recorded in the artifacts** for item 3, so the
   distinction survives being read out of the JSON rather than out of this file.

---

## Addendum 5 — 2026-08-09 post-response-2 audit response (POST HOC; not preregistered)

The second independent post-response audit
(`COMPASS_INDEPENDENT_POST_RESPONSE_2_TECHNICAL_SCIENTIFIC_AUDIT_2026-08-09.md`)
found that the fixes registered in Addendum 4 were real and that four further
problems changed what the data mean. The resulting deviations are recorded here
**after the outcome of the previous round was known**, which is the strongest
reason to state them plainly rather than fold them into the frozen text above.

1. **The primary arm is now N4-compliant by projection.** Addendum 4 created
   `n4_strict` and `n4_projected` as sensitivity arms while the headline
   continued to be scored on a cohort containing two implementations whose
   sampled parameters the frozen protocols cannot observably perturb. The audit
   called that a primary scientific blocker and allowed two dispositions:
   exclude the failing implementations, or project the failed dimensions out
   before ensemble construction. **Projection is now the primary rule**, and the
   former primary rule survives as the `n4_retained` arm so the cost of the
   change is measured rather than asserted.

   *Why projection and not exclusion.* Exclusion removes a whole source lineage
   and moves both direction criteria across an exact 0.80 threshold on the
   denominator alone (4/5 → 3/4). Choosing between two dispositions **after**
   seeing that one of them flips two criteria is exactly the situation
   preregistration exists to prevent, so the reasoning is stated in full: the
   projection is preferred because it preserves the inferential unit count and
   because the dropped set was measured by gate N4 before any endpoint was
   inspected, not because of which way the criteria fall. Both arms are reported
   side by side in every summary, and a reader who prefers exclusion can read
   `n4_strict` directly.

2. **The rheobase measurement procedure changed, and campaign rows changed with
   it (`probe-2` → `probe-3`).** The search returned "does not spike" whenever
   the ceiling current failed to evoke a spike, which equates depolarisation
   block at high current with silence; and the rheobase-normalised protocol
   family was suppressed at rheobase 0 only for models on a *model-level*
   spontaneity list, so a draw that became spontaneous in an ordinarily silent
   model ran "1.5× rheobase" at 0 nA. Both are measurement defects rather than
   analysis choices, so the affected rows were regenerated. The affected set is
   enumerable by construction and was enumerated by re-probing every stored
   non-spiking row with the real solver; the numbers are in
   `data/provenance/rheobase_probe3_scan.json` and
   `data/provenance/rheobase_probe3_repair.json`.

3. **Gate N2 is implemented for the subset with a sourceable published scalar.**
   It reported `0/0` through three audits. It now compares each model's canonical
   operating point against a published value declared in
   `data/provenance/n2_source_targets.json`, and **explicitly withdraws any claim
   of source-level reproduction** for the eight models where no such scalar
   exists. This is not trace-level reproduction and is not described as such.

4. **All learned cross-validation preprocessing moved inside the training
   fold.** Dummy levels, imputation medians, standardisation constants, the
   zero-variance drop list and the endpoint's own centring were previously fitted
   on the pooled data before the held-out lineage was separated. Two of those
   leaks are inert for an OLS R² and two are not; the distinction is documented
   in `compass.analysis.hierarchical.FoldPipeline` rather than glossed as a
   blanket fix.

5. **Cross-validation is reported as a fold distribution, not a median.** The
   median lineage-transfer R² concealed one severely negative fold. Every fold,
   the worst fold, the interquartile range and the number of folds beating the
   intercept-only null are reported together.

6. **The label `H-G` is reported as an exploratory decision-framework outcome.**
   The five H-G criteria are correlated derivatives of one small panel — four are
   computed from the same endpoint table and four from the same endpoint — so the
   vote count is not five independent confirmations. `decision.json` now carries
   `criterion_dependence` and `interpretation` blocks stating what the label may
   and may not be described as. **H-B is recorded as UNRESOLVED**, not refuted.

---

## Addendum 6 — post-hoc changes made in response to the 2026-08-09 post-response-3 audit

Same rule as Addendum 5: everything here was decided **after** outcomes were
visible and is therefore exploratory. It is listed so a reader can discount it,
not to make it preregistered retrospectively.

1. **The rheobase search was found to be sparse, and the whole corpus was
   regenerated (`probe-3` → `probe-4`).** `probe-3` walked a geometric ladder
   `hi/2**k` and only when the ceiling did not spike. A reviewer's deterministic
   counterexample — a cell firing from 0.30 to 0.40 nA under a 1.0 nA ceiling —
   returned `None`, because the ladder probes 0.25 and 0.50 and nothing between.
   Running the ladder only for ceiling-silent cells was a second defect: a cell
   with a low disconnected window *and* an upper firing branch was bisected on
   the upper branch, so its rheobase was overstated rather than missed.

   `probe-4` scans a **uniform grid of declared resolution** — 64 points over
   `(0, ceiling]` — upward from the bottom, on **every** sample.
   `compass.adapters.base.RHEOBASE_SEARCH_CONTRACT` states both halves of what
   that provides: any firing window at least `ceiling/64` wide is found; a
   narrower one may be stepped over, so `None` means *no spike at any probed
   amplitude at this resolution* and never *silent everywhere*.
   `tools/audit_rheobase_resolution.py` measures the observed firing-window
   widths on a stratified sample at 8× that resolution, so whether 64 is fine
   enough is answered with evidence rather than assertion.

   **No enumeration argument is made this time.** `probe-3`'s targeted repair
   rested on the claim that the change touched a strictly enumerable set of
   rows; the claim was false, its equivalence check retained a `NOT equivalent`
   verdict, and its affected-row counts could not be reconciled. The corpus is
   now regenerated in full from the stored parameter draws — the measurement
   schema is part of every shard's reuse identity, so bumping it invalidates
   every shard — and `tools/repair_rheobase.py` was deleted. In-place mutation
   of feature shards is no longer an operation this repository offers. The
   before/after difference is measured by diffing two immutable corpus snapshots
   (`tools/snapshot_corpus.py`, `tools/reconcile_corpus.py`), and every count
   quoted anywhere carries its inclusion criterion.

2. **One voltage band, applied to the corpus rather than to a report.** The
   waveform battery flagged traces outside `[-150, 80] mV` as physiologically
   impossible while the corpus rule excluded only outside `[-160, 120] mV`, so a
   trace could be "impossible" in the report and fully analysis-eligible in the
   inference. There is now one band (`PHYSICAL_V_MIN_MV` / `PHYSICAL_V_MAX_MV`),
   set at the tighter pair, used by both. Measured cost of the tightening on the
   previous retained sample: 4 of 914 protocol traces move from flagged-only to
   excluded, against 17 already excluded under the wider band.

3. **Every waveform finding class now has a disposition, fixed before the run.**
   `retain_waveforms.FINDING_DISPOSITION` assigns each class to one of
   *integrity failure*, *numerical invalid*, *prior predictive*, *detector
   limit* or *protocol settling*, and names the mechanism that carries it into
   the analysis. A class judged invalid is excluded by a corpus-wide rule
   applying to all rows, not repaired in a report covering a sample. A finding
   class with no declared disposition fails the WAVEFORM gate.

4. **The N4 projection applies to every reported arm, not only the discovery
   headline.** `replication_only` was reading the unprojected replication
   endpoint table while the manifest reported N4 as globally projected on the
   strength of the *primary* arm's filename. The replication arm now consumes
   the projected table and `replication_n4_retained` is added so the cost of the
   projection is measured on that arm too, exactly as `n4_retained` does for
   discovery. Disposition is computed per arm and per model by
   `compass.analysis.cohorts.n4_disposition_by_arm()`.

5. **H-B has an input gate.** `H_B_INPUT` is mandatory and has four subjects:
   the battery's measurement schema is current, its half-step twin exists, its
   repeatable subset clears the lineage floor, and H-B carries an explicit
   evaluable-or-withheld disposition. Previously nothing in the release logic
   would have stopped a stale battery if the other blockers were cleared, and
   the analysis tool could clear the staleness check by stamping its own
   schema — so the gate is paired with an identity sidecar that the analysis
   must read rather than write.

6. **N2's published summary is generated, and says 6 of 14.** The roll-up reads
   `7/7`, which is six model tests plus one declaration-coverage test. The
   sentence every document now quotes is
   `compass.validation.gates.n2_coverage()["statement"]`: *source-referenced
   canonical scalar agreement: 6/14 evaluable implementations*. This is a
   reporting change, not a change of evidence.
