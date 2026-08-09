# COMPASS

**Comparative Modeling of Parameter-Space Synergy and Sampling**

Are the higher-order parameter interactions reported in phenotype-constrained
neuronal models neuron-specific biological signatures, or generic consequences of
nonlinear model structure, parameter sampling, covariance, and conditioning on
selected outputs?

COMPASS applies one sampling and analysis framework to a panel of
conductance-based neuron models -- **fourteen implementations across ten source
lineages**, of which the frozen discovery cohort is eight implementations over
six lineages -- plus synthetic systems with analytically known dependence
structure and two non-neuronal nonlinear dynamical systems, under matched
sampling and conditioning regimes.

The distinction between *implementation* and *source lineage* is load-bearing
and was the subject of the third audit's finding 4: `hh_point`/`hh_cable` are one
Hodgkin-Huxley lineage and `pospischil_rs`/`pospischil_fs` one Destexhe-lab
framework, so counting model ids overstates the inferential sample. Every fold,
bootstrap and random effect clusters on `source_lineage_id`.

It is designed as a **falsification study**: a synergistic parameter tuple is not
treated as biological merely because it appears in a neuron model. See
`docs/preregistration.md` for the frozen hypotheses, endpoints and decision
thresholds, and `docs/RESULTS.md` for the findings.

---

## Headline result

**Conclusion: mixed (H-M)**, on 7 implementations over 5 source lineages after
**four** models were excluded by the numerical-convergence gate (`mainen_l5pc`,
`miyasho_purkinje`, `ortner_snc_da`, `tzilivaki_fsbc` — the four entries of
`cohorts.PRIMARY_EXCLUSIONS`, each failing N3). An earlier version of this line
said three.

Read [`docs/RESULTS.md` §0](docs/RESULTS.md) first: the third audit
(2026-08-06) invalidated two published results — a biased intrinsic-dimension
estimator and a spike-count "measurement floor" — and both are withdrawn and
recomputed rather than softened.

The **sign** of higher-order structure is generic — set by how the ensemble was
built, not by which neuron it came from. The **identity** of the parameters
involved is not: the extreme tuples are reproducible and do interact in the model
equations.

| evidence that the sign is generic | value |
|---|---|
| cross-model agreement, conditioning → synergy | **4 / 5 source lineages** in the primary cohort (5/6 with Mainen retained); exact two-sided sign p = 0.375, or 0.219 with Mainen. Unanimous across implementations at the permissive tolerance. |
| cross-model agreement, high-covariance → redundancy | **4 / 5 source lineages**; exact two-sided sign p = 0.375. **Fails when Mainen is retained** (4/6, p = 0.6875). Weakened from 5/5 (p = 0.0625) once E7 selection was made genuinely held out by cross-fitting: the earlier subset was ranked using axes partly fitted on the rows it then selected. |
| unique R²(sampling+geometry) ÷ unique R²(identity) | **>1,800×** (a lower bound: with lineage as the identity block, unique R²(identity) falls below the 1e-4 resolution floor); **12.6×** size-matched, CI [3.6, 60.5], 100% of resamples above threshold |
| effect exceeds each model's own conditional bootstrap noise | **5 / 8 implementations**, SNR 4–5 (a conditional bootstrap ratio, *not* a sigma significance — see below) |
| effective inferential sample | **7 implementations / 5 source lineages** in the primary cohort; 14 implementations / 10 lineages registered |
| same effect in a non-neuronal gene circuit | Ω 0.0000 → **−0.0018 … −0.183** (permissive → very strict) |
| shuffled null still "synergy-dominated" by the naive statistic | **45%** (surrogate-controlled: 0.000) |

| evidence that the identities are not generic | value |
|---|---|
| top-tuple agreement, Gaussian vs Gaussian-copula | Jaccard **0.83** (min 0.67) — descriptive only; both are Gaussian estimators on the same samples, so this measures estimator similarity, not module stability |
| within-campaign implementation identifiability | **0.566** vs 0.151 baseline — descriptive only; random within-table splits of derived rows from one campaign measure dataset fingerprinting, not biological identity |
| top-synergy vs **random** controls, non-additivity | 4.74× point, CI [1.90, 9.65], paired p = 0.016 over 7 models |
| top-synergy vs **matched** controls | 2.97× point, CI [1.35, 7.71], paired p = 0.047; lineage-clustered paired difference +0.203 SD, CI [+0.018, +0.439] over 5 lineages |

**The last two rows are stronger than the previous release reported, but still
bounded.** The point ratios are 2.97
(matched) and 4.74 (random), with
paired Wilcoxon p = 0.047 and
p = 0.016 — both now below 0.05, where the
previous release had 0.195 and 0.078 and neither reached it. Every one of the
seven models shows top > random, and six of seven show top > matched.

Three limits keep this short of established. The bootstrap upper bounds (7.71
and 9.65) sit far above the point estimates, because seven models is a small
unit. `mainen_l5pc` is absent from the perturbation battery entirely, so this
arm carries the partial-coverage caveat the decision table prints and cannot be
compared with an arm that has complete coverage. And the lineage-clustered
interval — the quantity actually scored — rests on five lineages.
**Only one of the two control comparisons is scored**:
they come from the same perturbation result on the same top tuples and differ
only in the denominator, so scoring both would count one measurement twice.
Two further limits matter as much as the point estimates.

*It depends on which quantity you measure.* On identical simulations, identical
tuples and identical perturbations, enrichment over matched controls is:

| readout | enrichment | r(Ω, \|η\|) |
|---|---:|---:|
| mean firing rate (spike count) | **2.97×** | −0.515 |
| first ISI (spike time) | **2.79×** | −0.318 |
| AP amplitude (waveform) | 1.32× | −0.162 |
| AP half-width (waveform) | 0.85× | −0.005 |

*A previous "measurement floor" claim here has been withdrawn.* The 2026-08-06
audit (§4) showed the analysis was invalid: it compared a **median over six
tuple means of eight backgrounds** against **one unaveraged single-observation
2.5 Hz quantum**. The mean's own grid is 2.5/8 = 0.3125 Hz, and a deterministic
quantisation step is not a stochastic detection limit — a mean below one quantum
is exactly observable and exactly repeatable. Everything that followed from it
is retired: "four of eight models unresolvable", "the effect is carried by the
two toy models", and "both cortical reconstructions are below a hard floor".
The replacement separates the two questions the old analysis conflated —
numerical repeatability (rerun at half the time step, compare per-background η)
and inferential precision (bootstrap and exact sign-flip inference over the
background distribution) — in
[`results/analyses/perturbation_inference.json`](results/analyses/perturbation_inference.json).
Raw per-background outcomes are now archived in `perturbations_raw.parquet`;
the earlier run stored only summaries, which is why no reanalysis was possible.

*The direction criterion is scored below 1.000.* It averages the conditioning
shift across all four tolerances, giving a 66-sample very-strict ensemble the
same weight as a 6,642-sample permissive one. One model's mean turns positive on
the strength of a 2.0σ fluctuation. At the permissive tolerance alone the
agreement is unanimous. The per-model bootstrap ratios reported in
`estimator_reliability.json` are **conditional bootstrap signal-to-noise
ratios**, not sigma significances: they bootstrap only the already-selected E3
parameters, hold the E1 endpoint fixed, do not repeat conditioning or target
estimation, and use 40 replicates. The criterion is left as preregistered rather
than rescored after the fact.

Three methodological results stand independently of the panel:

* A **shuffled null** with all dependence destroyed still reports 45%
  "synergy-dominated" tuples; matched-search surrogate control returns **0.000**.
* Gaussian and copula O-information are **structurally blind** to product-type
  synergy — exactly 0.000 where a k-NN estimator returns −1.57 nats.
* Two ensembles at **the same mean |r| (0.27 vs 0.29)** land on **opposite
  sides** of zero. Covariance magnitude does not determine the balance.

---

## Model panel

| model_id | class | source | pipeline | params | compartments |
|---|---|---|---|---|---|
| `hh_point` | minimal Hodgkin-Huxley | Hodgkin & Huxley 1952 (NEURON `hh`) | independent | 6 | 1 |
| `hh_cable` | minimal HH + dendrite | as above | independent | 8 | 11 |
| `pospischil_rs` | cortical pyramidal, regular spiking | ModelDB 123623 | independent | 8 | 1 |
| `pospischil_fs` | cortical fast-spiking interneuron | ModelDB 123623 | independent | 7 | 1 |
| `khaliq_purkinje` | cerebellar Purkinje | ModelDB 48332 | independent | 10 | 1 |
| `mainen_l5pc` | cortical L5 pyramidal | ModelDB 2488 | independent | 10 | 2304 |
| `migliore_ca1` | hippocampal CA1 pyramidal | ModelDB 2796 | independent | 8 | 1106 |
| `bbp_l5pc` | cortical L5b pyramidal | ModelDB 139653 + BlueBrain/BluePyOpt | **BBP** | 15 | 642 |

Two independently developed cortical pyramidal models (`mainen_l5pc` vs
`bbp_l5pc`) separate neuronal identity from laboratory implementation; two models
from one lab and framework (`pospischil_rs`, `pospischil_fs`) separate cell class
from implementation; two morphological variants of one channel set (`hh_point`,
`hh_cable`) isolate morphology.

## Design

Parameters are sampled as **multiplicative scale factors on each model's
canonical per-segment conductance profile**, log-uniform, canonical = 1.0. This
makes a "factor of 2" mean the same thing in a point model and in a
1000-compartment reconstruction, and it preserves each model's published spatial
channel distributions rather than flattening them.

Ensemble regimes (blueprint §13.2):

| code | regime | simulated? |
|---|---|---|
| E1 | unconditioned independent (Sobol) prior | yes |
| E2 | unconditioned covariance-preserving prior | yes |
| E6 | clustered / oversampled basins | yes |
| E3 / E3b / E4 / E4b | phenotype-conditioned at 4 tolerances | derived |
| E3m | size-matched random subset of E3 | derived |
| E7 | high-covariance subset, size- and fitness-matched | derived |
| E8 / E9 | shuffled and copula-matched null controls | derived |

## Install

```bash
pip install -e .
```

Requires NEURON ≥ 8.2 with a working `nrnivmodl`, and a C compiler.

## Reproduce

**One command rebuilds everything** — assets, mechanisms, calibration, campaigns,
analyses, gates, notebook and report — with per-stage logs and fail-closed
behaviour:

```bash
./tools/run_all.sh
```

It supports `--from <stage>`, `--until <stage>`, `--only <stage>`, `--dry-run`
and `--analysis-only`. Stages that are not run are reported as **skipped**, never
as passed. (The three older partial drivers — `run_pipeline.sh`, `run_phase2.sh`,
`run_repair.sh` — each reproduced a different subset and no single one recreated
all headline artifacts; `run_all.sh` replaces them.)

Individual stages, if you want them separately:

```bash
python tools/fetch_modeldb_index.py     # full ModelDB metadata index (model selection audit)
python tools/fetch_models.py            # download the panel, hash and record provenance
python tools/build_mechanisms.py        # compile per-model NEURON mechanism libraries
python tools/calibrate_amplitudes.py    # measure f-I curves, propose protocol amplitudes
python tools/convergence_audit.py <model>  # feature-level convergence over prior samples
python tools/canonical_audit.py         # canonical reproduction, f-I curves, reference traces
python tools/run_campaign.py --experiment screening
python tools/run_campaign.py --experiment screening_v2_replication
```

Verification:

```bash
pytest -m "not slow"
```

```bash
ruff check src tests tools
```

Release validation (slow; every gate fails closed, and a missing gate record is
a failure rather than a pass):

```bash
python tools/validate_campaign.py --all && python tools/validate_replay.py --experiment screening && python tools/build_release_manifest.py
```

```bash
pytest -m slow
```

## Layout

```
configs/    experiment, prior, protocol and conditioning configuration
data/
  raw/        downloaded model archives (unmodified)
  models/     compiled per-model mechanism libraries
  provenance/ download hashes, ModelDB index, numerical-QC evidence
src/compass/
  registry/   typed manifest / parameter / protocol / experiment schemas
  adapters/   NEURON adapter base + the eight model builders
  sampling/   transforms, independent / correlated / clustered priors, null controls
  protocols/  the frozen common-core protocol library
  features/   eFEL-based + project-owned feature extraction and QC
  conditioning/ target distances, tolerance series, ensemble construction
  analysis/   entropy, O-information, RSI, geometry, tuples, Sobol, perturbation,
              hierarchical inference, synthetic and non-neuronal benchmarks
  workflows/  block-based simulation with atomic result shards
  reporting/  figure generation
tools/      the executable pipeline stages
tests/      unit, numerical and integration tests
results/    manifests, feature shards, derived analysis tables, traces
docs/       preregistration, results, model audits
notebooks/  COMPASS_results.ipynb
```

## What this study can and cannot establish

Following blueprint §30, four levels of claim are kept distinct:

* **Observed** — a metric has a value in a declared ensemble. *Established.*
* **Statistical** — an ensemble is synergy- or redundancy-dominated under a
  stated estimator. *Established.*
* **Functional** — joint parameter perturbations are non-additive in the model
  equations. *Established, for the tested modules.*
* **Biological** — the same interaction occurs in real neurons. **Not established
  and not claimed.** Every prior here is evidence tier C (a published
  optimisation range) or tier D (an expert-defined range); none is a measured
  biological distribution.

## References

* Reva M. et al. (2025) *Higher-order interactions in neuronal function.* PNAS. https://doi.org/10.1073/pnas.2500048122
* Rosas F. E. et al. (2019) *Quantifying high-order interdependencies via multivariate extensions of the mutual information.* Phys Rev E 100:032305.
* Chechik G. et al. (2001) *Group redundancy measures reveal redundancy reduction in the auditory pathway.* NIPS 14.
* Prinz A. A., Bucher D., Marder E. (2004) *Similar network activity from disparate circuit parameters.* Nat Neurosci 7:1345.
* Ince R. A. A. et al. (2017) *A statistical framework for neuroimaging data analysis based on mutual information estimated via a Gaussian copula.* Hum Brain Mapp 38:1541.
* Facco E. et al. (2017) *Estimating the intrinsic dimension of datasets by a minimal neighborhood information.* Sci Rep 7:12140.
* Hay E. et al. (2011) *Models of neocortical layer 5b pyramidal cells...* PLoS Comput Biol 7:e1002107.
* Mainen Z. F., Sejnowski T. J. (1996) *Influence of dendritic structure on firing pattern in model neocortical neurons.* Nature 382:363.
* Pospischil M. et al. (2008) *Minimal Hodgkin-Huxley type models for different classes of cortical and thalamic neurons.* Biol Cybern 99:427.
* Khaliq Z. M., Gouwens N. W., Raman I. M. (2003) *The contribution of resurgent sodium current to high-frequency firing in Purkinje neurons.* J Neurosci 23:4899.
* Migliore M. et al. (1999) *Role of an A-type K+ conductance in the back-propagation of action potentials in the dendrites of hippocampal pyramidal neurons.* J Comput Neurosci 7:5.
* Elowitz M. B., Leibler S. (2000) *A synthetic oscillatory network of transcriptional regulators.* Nature 403:335.

## License

MIT for the COMPASS code. Downloaded models retain their original licenses; see
`data/provenance/downloads.json`. No third-party model source is redistributed in
this repository — `tools/fetch_models.py` retrieves it from the original archive.
