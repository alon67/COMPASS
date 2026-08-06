"""Generate the final COMPASS analysis notebook.

The notebook is *built*, not hand-edited, so that it is always consistent with
the analysis modules and can be regenerated from clean derived data with no
hidden state (blueprint Task 15 verification criterion).
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "COMPASS_results.ipynb"

MD = nbf.v4.new_markdown_cell
CODE = nbf.v4.new_code_cell

CELLS: list = []


def md(text: str) -> None:
    CELLS.append(MD(text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(CODE(text.strip("\n")))


# --------------------------------------------------------------------------- #
md(r"""
# COMPASS — Comparative Modeling of Parameter-Space Synergy and Sampling

**Are higher-order parameter interactions in phenotype-constrained neuronal models
neuron-specific biological signatures, or generic consequences of nonlinear model
structure, parameter sampling, and conditioning on selected outputs?**

This notebook regenerates every figure and number in `docs/RESULTS.md` from the
frozen derived tables in `results/analyses/`. It contains **no analysis logic**:
all estimators, ensembles and statistics live in `src/compass/` and are executed
by the `tools/run_*.py` scripts. Re-running this notebook from a clean kernel
reproduces the report exactly.

Pipeline that produced the inputs:

```bash
python tools/fetch_models.py          # download panel, record provenance + hashes
python tools/build_mechanisms.py      # compile per-model NEURON mechanism libraries
python tools/numerical_qc.py --spatial  # time-step + spatial convergence audit
python tools/canonical_audit.py       # canonical reproduction, f-I curves, traces
python tools/run_campaign.py --experiment screening
python tools/run_analysis.py --experiment screening
python tools/run_benchmarks.py
python tools/run_sobol.py
python tools/run_perturbations.py
python tools/run_crossmodel.py
```

**Reporting discipline (blueprint §30).** Four kinds of statement are kept apart
throughout:

| level | meaning | established here? |
|---|---|---|
| **Observed** | a metric has a value in a declared ensemble | yes |
| **Statistical** | the ensemble is synergy- or redundancy-dominated under a stated estimator | yes |
| **Functional** | joint perturbations are non-additive in the model equations | yes |
| **Biological** | the same interaction occurs in real neurons | **no — out of scope** |
""")

code(r"""
%matplotlib inline
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from compass.reporting import figures as FG
FG.apply_style()

AN  = ROOT / "results" / "analyses"
MAN = ROOT / "results" / "manifests"
TR  = ROOT / "results" / "traces"
PROV = ROOT / "data" / "provenance"
EXP = "screening"

def load(name, kind="parquet"):
    p = AN / name
    if not p.exists():
        print(f"  (missing {name})"); return None
    return pd.read_parquet(p) if kind == "parquet" else json.loads(p.read_text())

endpoints  = load(f"endpoints_annotated_{EXP}.parquet")
if endpoints is None:
    endpoints = load(f"endpoints_{EXP}.parquet")
curves     = load(f"order_curves_{EXP}.parquet")
tuple_sc   = load(f"tuple_scores_{EXP}.parquet")
rsi_tab    = load(f"rsi_{EXP}.parquet")
cond_shift = load("conditioning_shift.parquet")
cov_shift  = load("covariance_shift.parquet")
vpart      = load("variance_partition.parquet")
lomo       = load("lomo.parquet")
clf        = load("classification.parquet")
stability  = load("module_stability.parquet")
sobol_sum  = load("sobol_summary.parquet")
sobol_idx  = load("sobol_indices.parquet")
sobol_conv = load("sobol_convergence.parquet")
pert       = load("perturbations.parquet")
bench      = load("benchmarks.json", "json")
decision   = load("decision.json", "json")
mixed      = load("mixed_effects.json", "json")

print(f"endpoints rows      : {0 if endpoints is None else len(endpoints)}")
print(f"order-curve rows    : {0 if curves is None else len(curves)}")
print(f"per-tuple scores    : {0 if tuple_sc is None else len(tuple_sc)}")
""")

# --------------------------------------------------------------------------- #
md(r"""
---
## 1. Provenance and the model panel

Every model was obtained from a citable archive and hashed. The panel spans the
blueprint §6.1 slots and contains **two independently developed cortical
pyramidal models** — one from the Blue Brain Project lineage and one that was
never touched by the BBP pipeline — which is what separates neuronal identity
from laboratory implementation (§6.4).
""")

code(r"""
dl = json.loads((PROV / "downloads.json").read_text())
prov = pd.DataFrame([{
    "source": d["source"], "id": d["id"], "slot": d["slot"],
    "status": d["status"], "sha256/commit": (d.get("sha256") or d.get("commit",""))[:16],
    "n_mod": d.get("n_mod"),
} for d in dl])
display(prov)

rows = []
for p in sorted(MAN.glob("canonical_*.json")):
    r = json.loads(p.read_text()); m = r["manifest"]
    rows.append({
        "model_id": r["model_id"], "class": m["neuron_class"], "lab": m["lab"],
        "pipeline": m["pipeline"], "compartments": r["n_compartments"],
        "params": r["n_parameters"], "temp_C": m["temperature_c"],
        "V_rest_mV": round(r["resting_potential_mV"], 1),
        "rheobase_nA": None if r["rheobase_nA"] is None else round(r["rheobase_nA"], 4),
        "spont_Hz": round(r["spontaneous_rate_Hz"], 1),
        "s/protocol": r["mean_protocol_s"],
    })
panel = pd.DataFrame(rows).set_index("model_id").loc[
    [m for m in FG.MODEL_ORDER if m in {r["model_id"] for r in rows}]]
display(panel)
""")

md(r"""
### Numerical quality control

Each model's time step was chosen by a convergence audit against the finest
fixed step, and multicompartment models were additionally checked for spatial
convergence. The campaign runs **deterministic fixed steps**, not CVODE: under a
broad prior CVODE's cost depends on how stiff each individual sample happens to
be (profiling `mainen_l5pc` gave a median of 11 s per sample but individual
samples of 233 s and 307 s), which cannot be budgeted and risks biasing the
ensemble if any sample were abandoned.
""")

code(r"""
qc = json.loads((PROV / "numerical_qc.json").read_text())
rows = []
for r in qc:
    accepted = [k for k, v in r["accepted"].items() if v]
    sp = r.get("spatial", {})
    rows.append({
        "model_id": r["model_id"],
        "accepted_timesteps": ", ".join(accepted),
        "audit_choice": r["chosen"],
        "spatial_converged_as_published":
            None if not sp.get("applicable") else
            sp["levels"]["as_published"]["matches_finest"],
        "spatial_choice": sp.get("chosen", "-"),
    })
qcdf = pd.DataFrame(rows).set_index("model_id")
from compass.workflows.simulate import CAMPAIGN_SOLVER
qcdf["campaign_dt_ms"] = [CAMPAIGN_SOLVER.get(m, {}).get("dt_ms") for m in qcdf.index]
display(qcdf)
""")

code(r"""
fig = FG.fig_model_panel(MAN, TR); plt.show()
""")

# --------------------------------------------------------------------------- #
md(r"""
---
## 2. Estimator validation (blueprint WP5)

No neuronal number is believed until the estimators are shown to recover known
structure and the sign conventions are verified against analytic systems.

Conventions, frozen and asserted in `tests/numerical/`:

* **O-information** $\Omega>0$ → redundancy, $\Omega<0$ → synergy.
* **RSI** $= I(X_{1..n};Y)-\sum_i I(X_i;Y)$: $>0$ → synergy, $<0$ → redundancy.

Note the opposite polarity of the two — the most common sign error in this
literature.
""")

code(r"""
sign = pd.DataFrame(bench["rsi_sign_validation"])
display(sign[["case","expected","rsi","joint","sum_individual","correct"]].round(4))
print("all RSI sign conventions correct:", bool(sign["correct"].all()))
fig = FG.fig_estimator_validation(bench); plt.show()
""")

md(r"""
**Finding E1 — the Gaussian and copula estimators are structurally blind to
product-type synergy.** Both return $\Omega \approx 0.000$ for `xor_like`
($x_3=x_1x_2$) and `multiplicative` ($x_4=x_1x_2x_3$), where the k-NN estimator
returns $-0.84$ and $-1.57$ nats. Both *do* detect sum-type synergy
(`sum_constrained`, $\Omega=-3.27$), and all three agree on every redundancy
case. So the covariance-based estimator used throughout the published literature
can only see synergy that is already expressed in second-order statistics. This
is a limitation of the instrument, not of the systems, and it bounds what any
Gaussian O-information result — including the ones below — can claim.
""")

# --------------------------------------------------------------------------- #
md(r"""
---
## 3. Does conditioning manufacture higher-order structure?

The first secondary question of the blueprint (§2.2.1): *does phenotype
conditioning create higher-order dependence when the original prior is
independent?*

Every ensemble here starts from the **same** independent quasi-random prior. The
only thing that changes across the tolerance series is how tightly the accepted
samples are required to reproduce the canonical model's own phenotype.
""")

code(r"""
fig = FG.fig_conditioning_effect(endpoints, bench); plt.show()
""")

code(r"""
cs = cond_shift[(cond_shift.estimator=="gaussian") &
                (cond_shift.protocol_family=="absolute") &
                (cond_shift.metric=="max_z")]
tbl = (cs.pivot_table(index="model_id", columns="tolerance",
                      values=["d_net_o_information","d_mean_abs_pearson","acceptance_rate"])
         .reindex([m for m in FG.MODEL_ORDER if m in set(cs.model_id)]))
display(tbl.round(4))
g = cs.groupby("model_id")["d_net_o_information"].mean()
print(f"models shifting toward synergy after conditioning: {(g<0).sum()}/{len(g)}")
print(f"models gaining parameter correlation           : "
      f"{(cs.groupby('model_id')['d_mean_abs_pearson'].mean()>0).sum()}/{len(g)}")
""")

# --------------------------------------------------------------------------- #
md(r"""
---
## 4. O-information order curves with matched-search surrogate controls

Extreme O-information values grow simply because the number of tuples grows with
order. Every order is therefore accompanied by a family-wise surrogate band built
from column-shuffled ensembles evaluated over the **same number of tuples**
(blueprint §16.2). Only excursions outside that band are interpretable.
""")

code(r"""
fig = FG.fig_order_curves(curves); plt.show()
""")

code(r"""
cu = curves[(curves.estimator=="gaussian") & (curves.regime.isin(["E1","E3","E4"]))]
summary = (cu.groupby(["regime","order"])
             .agg(median_omega=("median","mean"),
                  frac_synergy=("frac_synergy","mean"),
                  frac_synergy_beyond_surrogate=("frac_synergy_sig","mean"),
                  frac_redundancy_beyond_surrogate=("frac_redundancy_sig","mean"),
                  n_tuples=("n_tuples","mean")))
display(summary.round(4))
""")

# --------------------------------------------------------------------------- #
md(r"""
---
## 5. Regime versus model identity

The blueprint's central comparison. If the balance between synergy and redundancy
were a property of the neuron, rows of the heat map below would be internally
consistent and differ from each other. If it is a property of the sampling and
conditioning design, columns would be consistent instead.
""")

code(r"""
fig = FG.fig_regime_heatmap(endpoints); plt.show()
""")

code(r"""
ep = endpoints[(endpoints.estimator=="gaussian") &
               (endpoints.protocol_family=="absolute") &
               (endpoints.metric.isin(["none","max_z"]))]
within_model  = ep.groupby("model_id")["net_o_information"].std().mean()
within_regime = ep.groupby("regime")["net_o_information"].std().mean()
print(f"mean SD of net O-information within a MODEL  (varying regime): {within_model:.4f}")
print(f"mean SD of net O-information within a REGIME (varying model) : {within_regime:.4f}")
print(f"ratio (regime effect / model effect): {within_model/within_regime:.2f}")
display(ep.groupby("regime")["net_o_information"]
          .agg(["mean","std","count"]).round(4).sort_values("mean"))
""")

md(r"""
### Statistical geometry, not neuron identity, tracks the endpoints
""")

code(r"""
fig = FG.fig_geometry_vs_endpoint(endpoints); plt.show()
""")

code(r"""
cvs = cov_shift[(cov_shift.estimator=="gaussian") & (cov_shift.protocol_family=="absolute")]
display(cvs.groupby("contrast")[["d_net_o_information","d_frac_synergy_o3",
                                 "d_mean_abs_pearson"]].mean().round(4))
""")

# --------------------------------------------------------------------------- #
md(r"""
---
## 6. Target-specific information (RSI)

O-information describes dependence among parameters. RSI asks a different
question: for a given electrical feature, do parameter tuples carry more
information jointly than separately? Computed for every common feature, on both
the unconditioned and the conditioned ensembles.
""")

code(r"""
if rsi_tab is not None and len(rsi_tab):
    fig = FG.fig_rsi(rsi_tab); plt.show()
    display(rsi_tab[rsi_tab.estimator=="gaussian"]
            .groupby(["regime","target"])[["frac_synergistic","median_rsi"]]
            .mean().round(4))
""")

# --------------------------------------------------------------------------- #
md(r"""
---
## 7. Cross-model variance partitioning and generalisation

Commonality analysis over four predictor blocks. `r2_unique` is each block's
semi-partial contribution — the variance it explains that no other block can —
so a block that is merely collinear with geometry cannot masquerade as an
independent effect.
""")

code(r"""
fig = FG.fig_variance_partition(vpart); plt.show()
if vpart is not None and len(vpart) and "endpoint" in vpart.columns:
    display(vpart.pivot_table(index="endpoint", columns="block",
                              values=["r2_unique","r2_alone"]).round(4))
print(json.dumps(mixed, indent=1))
""")

code(r"""
fig = FG.fig_generalisation(lomo, clf); plt.show()
if lomo is not None and len(lomo):
    display(lomo[lomo.endpoint=="synergy_area"]
            .groupby("predictors")[["r2_out_of_sample","rmse"]].median().round(4))
if clf is not None and len(clf):
    display(clf.round(4))
""")

# --------------------------------------------------------------------------- #
md(r"""
---
## 8. Functional sensitivity (Sobol)

Sobol indices are computed **only on the unconditioned Saltelli design**.
Applying a variance-based estimator after outcome filtering would violate the
independence assumption it rests on, and the code refuses to do it
(blueprint §16.4). The interaction fraction $1-\sum_i S_1^{(i)}$ is the
functional counterpart of statistical synergy.
""")

code(r"""
if sobol_sum is not None and len(sobol_sum):
    fig = FG.fig_sobol(sobol_sum, sobol_idx); plt.show()
    display(sobol_sum.pivot_table(index="model_id", columns="feature",
                                  values="interaction_fraction").round(3))
    display(sobol_conv.groupby(["model_id","n_base"])[["interaction_fraction",
                                                       "max_S1_conf"]].mean().round(4))
""")

# --------------------------------------------------------------------------- #
md(r"""
---
## 9. Direct perturbation: are statistical modules functionally real?

The decisive test. The most synergistic order-3 tuples in each model's
phenotype-conditioned ensemble are perturbed singly and jointly at eight
background points spread across the accepted manifold, and the non-additivity

$$\eta = \big[f(x_0+\textstyle\sum_i \delta_i)-f(x_0)\big]-\sum_i\big[f(x_0+\delta_i)-f(x_0)\big]$$

is measured in ensemble-SD units. A matched set of *control* tuples from the
middle of the O-information distribution is run identically, so the question is
not "do extreme tuples interact at all" but "do they interact **more than
ordinary tuples do**".
""")

code(r"""
if pert is not None and len(pert):
    fig = FG.fig_perturbation(pert); plt.show()
    display(pert.groupby("kind").agg(
        n=("tuple","size"),
        mean_abs_eta_SD=("mean_nonadditivity_sd", lambda s: np.abs(s).mean()),
        median_abs_eta_SD=("mean_nonadditivity_sd", lambda s: np.abs(s).median()),
        mean_sign_consistency=("sign_consistency","mean"),
        frac_functionally_supported=("functionally_supported","mean")).round(4))
    display(pert.groupby(["model_id","kind"])["mean_nonadditivity_sd"]
            .apply(lambda s: np.abs(s).mean()).unstack().round(3))
""")

# --------------------------------------------------------------------------- #
md(r"""
---
## 10. Module stability

A top-5% tuple list from a single run is not evidence (blueprint §16.6). Stability
is measured as the Jaccard overlap of the top-10 tuples across estimators, and as
the per-tuple agreement between the Gaussian and copula estimates.
""")

code(r"""
if stability is not None and len(stability):
    fig = FG.fig_module_stability(stability, tuple_sc); plt.show()
    display(stability.round(3))
""")

# --------------------------------------------------------------------------- #
md(r"""
---
## 11. The preregistered decision

Criteria and thresholds were fixed in `docs/preregistration.md` before the
screening campaign was analysed. This cell only evaluates them.

**Reading the counting rule.** Blueprint 19.3 defines H-M as holding when
"synergy or redundancy signs are largely governed by sampling regime **but**
stable, *functionally validated* module identities differ systematically across
neuronal classes." After the 2026-08-05 review both clauses are now satisfied:

* the sign is regime-governed (all five H-G criteria pass, 8/8 direction
  agreement on both the conditioning and the covariance contrast);
* the modules are stable across estimators (per-model Jaccard 0.81) **and**
  functionally validated -- top-synergy tuples are 2.3x more non-additive than
  matched middle-Omega controls and 4.1x more than random ones, both clearing
  the preregistered 1.5x threshold.

An earlier version of this notebook argued for overriding the rule toward H-G on
the grounds that "statistical modules frequently fail direct perturbation
validation". That claim came from a perturbation experiment with three design
faults (backgrounds drawn from E1-that-fires rather than the accepted manifold;
279 evaluations outside the declared prior; normalisation against the
unconditioned ensemble). With those fixed the finding reverses, so the override
no longer stands and H-M is reported as the conclusion.

""")

code(r"""
if decision:
    fig = FG.fig_decision(decision); plt.show()
    print(f"H-G criteria passed : {decision['n_hg_passed']}/{decision['n_hg_total']}")
    print(f"H-B criteria passed : {decision['n_hb_passed']}/{decision['n_hb_total']}")
    print(f"CONCLUSION          : {decision['conclusion']}")
    display(pd.DataFrame(decision["criteria"])[
        ["criterion","supports","value","passed","detail"]])
""")

md(r"""
---
## 12. Audit tables — failures are reported, not hidden

Blueprint §26: *never silently remove numerical failures*. Every simulation that
errored, produced a non-physical voltage, or failed to spike is retained in the
result table with its reason code.
""")

code(r"""
import glob
rows = []
for d in sorted((ROOT / "results" / "features" / EXP).glob("*")):
    fs = sorted(glob.glob(str(d / "*.parquet")))
    if not fs: continue
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    flags = (df.qc_flags.astype("string").replace("", pd.NA).dropna()
               .str.split(";").explode().str.split(":").str[-1].value_counts())
    rows.append({
        "model_id": d.name, "n_simulated": len(df),
        "n_error": int((df.simulation_status=="error").sum()),
        "n_non_spiking": int(df.non_spiking.sum()),
        "pct_non_spiking": round(100*df.non_spiking.mean(), 1),
        "unstable_baseline": int(flags.get("unstable_baseline", 0)),
        "nonphysical_voltage": int(flags.get("nonphysical_voltage", 0)),
        "solver_failure": int(flags.get("solver_failure", 0)),
    })
audit = pd.DataFrame(rows).set_index("model_id")
display(audit)
print(f"TOTAL simulations in the screening campaign: {audit.n_simulated.sum():,}")
print(f"hard failures: {audit.n_error.sum()}  "
      f"({100*audit.n_error.sum()/max(1,audit.n_simulated.sum()):.3f}%)")
""")

code(r"""
ep = endpoints[endpoints.estimator=="gaussian"]
display(ep.groupby("evidence_tier").agg(
    n_ensembles=("model_id","size"),
    models=("model_id", lambda s: s.nunique()),
    mean_synergy_area=("synergy_area","mean"),
    mean_net_o_information=("net_o_information","mean")).round(4))
print("Tier C = published optimisation range; Tier D = expert-defined range.")
print("Neither may be described as measured biological variability (blueprint 7.2).")
""")

md(r"""
---
## 13. Conclusions

The **sign** of higher-order structure -- whether an ensemble looks synergistic
or redundant -- is set by how the ensemble was built, not by which neuron it
came from. The **identity** of the parameters involved is not arbitrary: the
extreme tuples are reproducible across estimators and do interact measurably in
the model equations. That combination is H-M.

Evidence that the sign is generic:

1. It appears in a **gene circuit** with no neurons in it, from an exactly
   independent prior (section 3).
2. Its sign is set by the **ensemble regime**, identically across eight models
   spanning 1 to 2,304 compartments (section 5).
3. Sampling and geometry explain **8.5x** more unique endpoint variance than
   model identity (size-matched, model-cluster bootstrap 95% CI [3.0, 44.2],
   exceeding the preregistered threshold of 2 in 100% of resamples).
4. A shuffled null still reports **44%** "synergy-dominated" tuples by the naive
   statistic; surrogate control returns exactly 0 (section 4).

Evidence that the identities are not generic:

5. Top tuples are **2.3x** more non-additive than matched middle-Omega controls
   and **4.1x** more than random controls, with 25% meeting the absolute
   functional-support bar versus 6% of controls (section 9).
6. Model identity is decodable from information summaries at 0.58 against a
   0.15 baseline, and top-tuple membership is stable across estimators
   (Jaccard 0.81).

The honest limits: only 25% of top tuples clear the absolute functional-support
threshold, so enrichment is clear but most individual tuples still fail it; the
paired test against the harder middle-Omega comparator is p = 0.11 with n = 8
models; and transfer to a held-out model family fails badly for one of the eight
(migliore_ca1, R^2 = -4.5).

### Methodological recommendations

1. **Report a matched-search surrogate control.** Without one, ~46% of tuples
   look "synergy-dominated" in an ensemble with no dependence at all.
2. **Report the unconditioned ensemble.** The conditioned-vs-unconditioned
   contrast is the interpretable quantity; the conditioned value alone is not.
3. **Do not read covariance magnitude as redundancy.** Two ensembles here at
   mean |r| = 0.28 sit on opposite sides of zero.
4. **Validate modules by perturbation before interpreting them.** Estimator
   stability (Jaccard 0.84 here) does not imply mechanism.
5. **State the estimator's blind spot.** Gaussian and copula O-information
   return exactly 0.000 for product-type synergy that k-NN scores at −1.57 nats.
6. **Keep O-information and RSI separate.** Here they respond to conditioning in
   opposite directions.

---
## 14. Scope of the claims

* **Observed / Statistical / Functional** results above stand as reported, for the
  declared ensembles, estimators and models.
* **Biological** claims are not made. No parameter in this study carries evidence
  tier A or B; the priors are optimisation ranges (tier C) or expert-defined
  ranges (tier D), and the conditioning targets are the models' own canonical
  phenotypes rather than measured cell-level feature distributions.
* This is a **screening**-scale study (≈10⁵ simulations on one workstation), not
  the confirmatory scale (≈10⁷) the blueprint envisages. Declared deviations are
  listed in `docs/preregistration.md` §12.

See `docs/RESULTS.md` for the full narrative interpretation.
""")

# --------------------------------------------------------------------------- #
def main() -> int:
    nb = nbf.v4.new_notebook(cells=CELLS)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python",
                                 "name": "python3"}
    nb.metadata["language_info"] = {"name": "python", "version": sys.version.split()[0]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT)
    print(f"wrote {OUT} ({len(CELLS)} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
