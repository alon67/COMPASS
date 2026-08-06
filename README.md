# COMPASS

**Comparative Modeling of Parameter-Space Synergy and Sampling**

Are the higher-order parameter interactions reported in phenotype-constrained
neuronal models neuron-specific biological signatures, or generic consequences of
nonlinear model structure, parameter sampling, covariance, and conditioning on
selected outputs?

COMPASS applies one sampling and analysis framework to a panel of eight
independently developed conductance-based neuron models, plus synthetic systems
with analytically known dependence structure and two non-neuronal nonlinear
dynamical systems, under matched sampling and conditioning regimes.

It is designed as a **falsification study**: a synergistic parameter tuple is not
treated as biological merely because it appears in a neuron model. See
`docs/preregistration.md` for the frozen hypotheses, endpoints and decision
thresholds, and `docs/RESULTS.md` for the findings.

---

## Headline result

**98,048 simulations across 8 model families. Conclusion: mixed (H-M).**

The **sign** of higher-order structure is generic — set by how the ensemble was
built, not by which neuron it came from. The **identity** of the parameters
involved is not: the extreme tuples are reproducible and do interact in the model
equations.

| evidence that the sign is generic | value |
|---|---|
| cross-model agreement, conditioning → synergy | **8 / 8 models** |
| cross-model agreement, high-covariance → redundancy | **8 / 8 models** |
| unique R²(sampling+geometry) ÷ unique R²(identity), size-matched | **8.5×**, 95% CI [3.0, 44.2] |
| intraclass correlation of model identity | **0.048** |
| same effect in a non-neuronal gene circuit | Ω 0.0000 → **−0.068** |
| shuffled null still "synergy-dominated" by the naive statistic | **44%** (surrogate-controlled: 0.000) |

| evidence that the identities are not generic | value |
|---|---|
| top-synergy tuples vs **random** controls, non-additivity | **4.05×**, CI [1.68, 7.83], p = 0.023 |
| top-synergy tuples vs **matched** controls | **2.34×**, CI [1.11, 4.60], p = 0.109 |
| meeting the functional-support bar | **25%** vs 6% (matched), 4% (random) |
| top-tuple stability across estimators | Jaccard **0.81** |

Three methodological results stand independently of the panel:

* A **shuffled null** with all dependence destroyed still reports 44%
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

```bash
python tools/fetch_modeldb_index.py     # full ModelDB metadata index (model selection audit)
python tools/fetch_models.py            # download the panel, hash and record provenance
python tools/build_mechanisms.py        # compile per-model NEURON mechanism libraries
python tools/convergence_audit.py <model>  # feature-level convergence over prior samples
python tools/canonical_audit.py         # canonical reproduction, f-I curves, reference traces

python tools/run_campaign.py --experiment screening   # the simulation campaign
bash  tools/run_pipeline.sh                            # analysis -> Sobol -> perturbations -> decision

python tools/build_notebook.py          # regenerate the notebook
jupyter nbconvert --execute --to notebook --inplace notebooks/COMPASS_results.ipynb
```

Verification:

```bash
pytest            # 86 tests: schema, transforms, features, estimators, Sobol, perturbation
ruff check .
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
