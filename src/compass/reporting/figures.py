"""Figure generation for the COMPASS report.

Every function takes already-derived tables and returns a matplotlib Figure, so
the notebook contains no analysis logic and the figures can be regenerated from
``results/analyses/`` without any notebook state (blueprint Task 15).

Design conventions (validated categorical palette; see docs/):
 * O-information is a **polarity** quantity, so it uses a diverging blue/red
   scale with a neutral grey midpoint: blue = synergy (negative), red =
   redundancy (positive).  Categorical series use the fixed slot order and are
   never cycled.
 * No dual-axis charts anywhere.
 * Legends whenever two or more series are present; direct labels on the low
   contrast slots.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---- palette ------------------------------------------------------------- #
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
INK3 = "#8a8985"
GRID = "#e3e2de"
SYNERGY = "#2a78d6"      # diverging cool pole
REDUNDANCY = "#e34948"   # diverging warm pole
MIDPOINT = "#f0efec"
STATUS = {"good": "#0ca30c", "warning": "#fab219",
          "serious": "#ec835a", "critical": "#d03b3b"}

DIVERGING = mpl.colors.LinearSegmentedColormap.from_list(
    "compass_div", [SYNERGY, MIDPOINT, REDUNDANCY])
SEQUENTIAL = mpl.colors.LinearSegmentedColormap.from_list(
    "compass_seq", ["#cde2fb", "#3987e5", "#0d366b"])

MODEL_ORDER = ["hh_point", "hh_cable", "pospischil_rs", "pospischil_fs",
               "khaliq_purkinje", "mainen_l5pc", "migliore_ca1", "bbp_l5pc"]
MODEL_LABEL = {
    "hh_point": "HH point", "hh_cable": "HH cable",
    "pospischil_rs": "Cortical RS", "pospischil_fs": "Cortical FS",
    "khaliq_purkinje": "Purkinje", "mainen_l5pc": "L5PC (Mainen)",
    "migliore_ca1": "CA1 (Migliore)", "bbp_l5pc": "L5PC (BBP)",
}
REGIME_LABEL = {
    "E1": "E1 unconditioned", "E2": "E2 covariance prior",
    "E3": "E3 permissive", "E3b": "E3b moderate", "E4": "E4 strict",
    "E4b": "E4b very strict", "E3m": "E3m matched subset",
    "E6": "E6 clustered", "E7": "E7 high covariance",
    "E8": "E8 shuffled null", "E9": "E9 copula null",
}


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK3, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "legend.frameon": False, "figure.dpi": 120, "lines.linewidth": 2.0,
        "lines.markersize": 5,
    })


def _order_models(vals) -> list:
    known = [m for m in MODEL_ORDER if m in set(vals)]
    return known + sorted(set(vals) - set(known))


def _lab(m: str) -> str:
    return MODEL_LABEL.get(m, m)


# --------------------------------------------------------------------------- #
# Figure 1 - model panel
# --------------------------------------------------------------------------- #
def fig_model_panel(manifest_dir: Path, trace_dir: Path) -> plt.Figure:
    """Canonical traces and f-I curves for every model in the panel."""
    recs = {}
    for p in sorted(manifest_dir.glob("canonical_*.json")):
        r = json.loads(p.read_text())
        recs[r["model_id"]] = r
    models = _order_models(recs)
    n = len(models)
    fig, axes = plt.subplots(2, n, figsize=(2.05 * n, 4.6))
    for j, mid in enumerate(models):
        r = recs[mid]
        ax = axes[0, j]
        npz = trace_dir / f"canonical_{mid}.npz"
        if npz.exists():
            d = np.load(npz)
            t, v = d["step_hi_t"], d["step_hi_v"]
            # a 260 ms window from just before stimulus onset, so that individual
            # action potentials remain resolvable even at 300 Hz
            m = (t >= 90.0) & (t <= 350.0)
            ax.plot(t[m], v[m], color=SERIES[0], lw=0.8)
        ax.set_title(_lab(mid), fontsize=9)
        ax.set_xlabel("time (ms)")
        if j == 0:
            ax.set_ylabel("V (mV)")
        ax.set_ylim(-100, 60)

        ax = axes[1, j]
        fi = pd.DataFrame(r["fi_curve"])
        rate = pd.to_numeric(fi["rate_Hz"], errors="coerce").fillna(0.0)
        ax.plot(fi["amplitude_nA"], rate, color=SERIES[1], marker="o", ms=3, lw=1.6)
        ax.set_xlabel("I (nA)")
        if j == 0:
            ax.set_ylabel("rate (Hz)")
        rb = r.get("rheobase_nA")
        if rb:
            ax.axvline(rb, color=INK3, ls=":", lw=1.2)
    fig.suptitle("Figure 1  Model panel: canonical somatic response and f-I curve",
                 fontsize=11, fontweight="bold", y=1.0)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Figure 2 - estimator validation
# --------------------------------------------------------------------------- #
def fig_estimator_validation(benchmarks: dict) -> plt.Figure:
    """O-information on systems with analytically known structure."""
    df = pd.DataFrame(benchmarks["estimator_validation"])
    df = df[df["n"] == df["n"].max()]
    order = ["independent", "duplicated", "linear_correlated", "mixture_pooled",
             "sum_constrained", "xor_like", "multiplicative"]
    order = [o for o in order if o in set(df["system"])]
    ests = ["gaussian", "copula", "knn"]
    piv = df.pivot_table(index="system", columns="estimator", values="o_information")
    piv = piv.reindex(order)[ests]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.0),
                                  gridspec_kw={"width_ratios": [1.35, 1]})
    x = np.arange(len(order))
    w = 0.26
    for i, est in enumerate(ests):
        ax.bar(x + (i - 1) * w, piv[est].to_numpy(), width=w * 0.9,
               color=SERIES[i], label=est, edgecolor=SURFACE, linewidth=1.2)
    ax.axhline(0, color=INK3, lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([o.replace("_", "\n") for o in order], fontsize=8)
    ax.set_ylabel("O-information (nats)")
    ax.set_title("A  Estimator response to known dependence structure")
    ax.legend(title="estimator", fontsize=8, title_fontsize=8)
    ax.text(0.01, 0.04, "negative = synergy", transform=ax.transAxes,
            color=SYNERGY, fontsize=8, fontweight="bold")
    ax.text(0.01, 0.93, "positive = redundancy", transform=ax.transAxes,
            color=REDUNDANCY, fontsize=8, fontweight="bold")

    exp = df.drop_duplicates("system").set_index("system")["expected"].reindex(order)
    blind = piv.copy()
    for c in blind.columns:
        blind[c] = np.where(exp.to_numpy() == "synergy",
                            (piv[c] < -0.02).astype(float),
                            np.where(exp.to_numpy() == "redundancy",
                                     (piv[c] > 0.02).astype(float),
                                     (piv[c].abs() < 0.02).astype(float)))
    im = ax2.imshow(blind.to_numpy(), cmap=mpl.colors.ListedColormap([MIDPOINT, STATUS["good"]]),
                    vmin=0, vmax=1, aspect="auto")
    ax2.set_xticks(range(len(ests)))
    ax2.set_xticklabels(ests, fontsize=8)
    ax2.set_yticks(range(len(order)))
    ax2.set_yticklabels([f"{o}\n({e})" for o, e in zip(order, exp, strict=True)], fontsize=7)
    for i in range(len(order)):
        for j in range(len(ests)):
            ax2.text(j, i, "detected" if blind.to_numpy()[i, j] else "BLIND",
                     ha="center", va="center", fontsize=7,
                     color="white" if blind.to_numpy()[i, j] else INK,
                     fontweight="bold")
    ax2.set_title("B  Correct classification (green = detected)")
    ax2.grid(False)
    del im
    fig.suptitle("Figure 2  Information-estimator validation on analytically known systems",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Figure 3 - conditioning creates dependence
# --------------------------------------------------------------------------- #
def fig_conditioning_effect(endpoints: pd.DataFrame, benchmarks: dict) -> plt.Figure:
    """The central contrast: unconditioned prior vs phenotype-conditioned ensembles."""
    ep = endpoints[(endpoints["estimator"] == "gaussian") &
                   (endpoints["protocol_family"] == "absolute") &
                   (endpoints["metric"].isin(["none", "max_z"]))]
    tol_order = ["E1", "E3", "E3b", "E4", "E4b"]
    models = _order_models(ep["model_id"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.0))

    ax = axes[0]
    for i, mid in enumerate(models):
        g = ep[ep["model_id"] == mid].set_index("regime")
        y = [g.loc[r, "net_o_information"] if r in g.index else np.nan for r in tol_order]
        y = [float(np.atleast_1d(v)[0]) if np.size(v) else np.nan for v in y]
        ax.plot(range(len(tol_order)), y, marker="o", color=SERIES[i % 8], label=_lab(mid))
    ax.axhline(0, color=INK3, lw=1.2)
    ax.set_xticks(range(len(tol_order)))
    ax.set_xticklabels(["uncond.", "perm.", "moder.", "strict", "v.strict"], fontsize=8)
    ax.set_ylabel("net O-information\n(redundancy - synergy)")
    ax.set_title("A  Neuron models: conditioning tolerance")
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1]
    for i, sysrec in enumerate(benchmarks["comparators"]):
        labels = [e["label"] for e in sysrec["ensembles"]]
        vals = [e["o_information_full"] for e in sysrec["ensembles"]]
        ax.plot(range(len(labels)), vals, marker="s", color=SERIES[i],
                label=sysrec["system"].replace("_", " "))
    ax.axhline(0, color=INK3, lw=1.2)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["uncond.", "perm.", "moder.", "strict", "v.strict"], fontsize=8)
    ax.set_ylabel("O-information (nats)")
    ax.set_title("B  Non-neuronal comparators")
    ax.legend(fontsize=8)

    ax = axes[2]
    for i, mid in enumerate(models):
        g = ep[ep["model_id"] == mid].set_index("regime")
        y = [g.loc[r, "mean_abs_pearson"] if r in g.index else np.nan for r in tol_order]
        y = [float(np.atleast_1d(v)[0]) if np.size(v) else np.nan for v in y]
        ax.plot(range(len(tol_order)), y, marker="o", color=SERIES[i % 8], label=_lab(mid))
    ax.set_xticks(range(len(tol_order)))
    ax.set_xticklabels(["uncond.", "perm.", "moder.", "strict", "v.strict"], fontsize=8)
    ax.set_ylabel("mean |Pearson r| among parameters")
    ax.set_title("C  Dependence induced in an independent prior")

    fig.suptitle("Figure 3  Phenotype conditioning manufactures parameter dependence",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Figure 4 - order curves
# --------------------------------------------------------------------------- #
def fig_order_curves(curves: pd.DataFrame) -> plt.Figure:
    """O-information versus tuple order with family-wise surrogate bands."""
    cu = curves[(curves["estimator"] == "gaussian") &
                (curves["regime"].isin(["E1", "E3", "E4"]))]
    models = _order_models(cu["model_id"].unique())
    n = len(models)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.7 * nrow),
                             squeeze=False)
    colours = {"E1": SERIES[6], "E3": SERIES[0], "E4": SERIES[1]}
    for k, mid in enumerate(models):
        ax = axes[k // ncol][k % ncol]
        g = cu[cu["model_id"] == mid]
        for regime in ("E1", "E3", "E4"):
            gg = g[g["regime"] == regime].sort_values("order")
            if gg.empty:
                continue
            ax.plot(gg["order"], gg["median"], marker="o", color=colours[regime],
                    label=REGIME_LABEL[regime])
            ax.fill_between(gg["order"], gg["q05"], gg["q95"],
                            color=colours[regime], alpha=0.13, linewidth=0)
        gg = g[g["regime"] == "E3"].sort_values("order")
        if not gg.empty:
            ax.fill_between(gg["order"], gg["surrogate_lo"], gg["surrogate_hi"],
                            color=INK3, alpha=0.18, linewidth=0,
                            label="surrogate 90% band")
        ax.axhline(0, color=INK3, lw=1.0)
        ax.set_title(_lab(mid), fontsize=9)
        ax.set_xticks([3, 4, 5])
        if k % ncol == 0:
            ax.set_ylabel(r"$\Omega$ (nats)")
        if k // ncol == nrow - 1:
            ax.set_xlabel("tuple order")
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    axes[0][0].legend(fontsize=7)
    fig.suptitle("Figure 4  O-information order curves with matched-search surrogate bands",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Figure 5 - regime comparison heatmap
# --------------------------------------------------------------------------- #
def fig_regime_heatmap(endpoints: pd.DataFrame, value: str = "net_o_information",
                       estimator: str = "gaussian") -> plt.Figure:
    ep = endpoints[(endpoints["estimator"] == estimator) &
                   (endpoints["protocol_family"] == "absolute") &
                   (endpoints["metric"].isin(["none", "max_z"]))]
    regimes = [r for r in ["E1", "E2", "E3", "E3b", "E4", "E4b", "E3m",
                           "E6", "E7", "E8", "E9"] if r in set(ep["regime"])]
    models = _order_models(ep["model_id"].unique())
    M = np.full((len(models), len(regimes)), np.nan)
    for i, mid in enumerate(models):
        for j, reg in enumerate(regimes):
            v = ep[(ep["model_id"] == mid) & (ep["regime"] == reg)][value]
            if len(v):
                M[i, j] = float(v.mean())
    lim = np.nanmax(np.abs(M)) if np.isfinite(M).any() else 1.0
    fig, ax = plt.subplots(figsize=(1.05 * len(regimes) + 3.4, 0.52 * len(models) + 2.2))
    im = ax.imshow(M, cmap=DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(regimes)))
    ax.set_xticklabels([REGIME_LABEL.get(r, r).replace(" ", "\n", 1) for r in regimes],
                       fontsize=7.5)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([_lab(m) for m in models], fontsize=8)
    for i in range(len(models)):
        for j in range(len(regimes)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=6.5,
                        color=INK if abs(M[i, j]) < 0.55 * lim else "white")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("net O-information\n(- synergy   +redundancy)", fontsize=8)
    ax.grid(False)
    ax.set_title("Figure 5  Ensemble regime determines the synergy/redundancy balance",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Figure 6 - variance partitioning
# --------------------------------------------------------------------------- #
def _empty_figure(message: str) -> plt.Figure:
    """Placeholder so a missing upstream table cannot break the whole notebook."""
    fig, ax = plt.subplots(figsize=(7.0, 1.6))
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=10, color=INK2)
    return fig


def fig_variance_partition(vp: pd.DataFrame) -> plt.Figure:
    if vp is None or len(vp) == 0 or "endpoint" not in vp.columns:
        return _empty_figure("variance partition unavailable "
                             "(needs >= 3 model families in the endpoint table)")
    endpoints = list(dict.fromkeys(vp["endpoint"]))
    blocks = ["sampling", "geometry", "architecture", "model_identity"]
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    x = np.arange(len(endpoints))
    w = 0.2
    for i, b in enumerate(blocks):
        vals = [float(vp[(vp["endpoint"] == e) & (vp["block"] == b)]["r2_unique"].iloc[0])
                if len(vp[(vp["endpoint"] == e) & (vp["block"] == b)]) else np.nan
                for e in endpoints]
        bars = ax.bar(x + (i - 1.5) * w, vals, width=w * 0.9, color=SERIES[i],
                      label=b.replace("_", " "), edgecolor=SURFACE, linewidth=1.2)
        for b_, v in zip(bars, vals, strict=True):
            if np.isfinite(v) and v > 0.005:
                ax.text(b_.get_x() + b_.get_width() / 2, v + 0.006, f"{v:.2f}",
                        ha="center", fontsize=7, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([e.replace("_", "\n") for e in endpoints], fontsize=8)
    ax.set_ylabel(r"unique $R^2$ (semi-partial)")
    ax.set_title("Figure 6  Variance in the information endpoints attributable to each block",
                 fontsize=11, fontweight="bold")
    ax.legend(title="predictor block", fontsize=8, title_fontsize=8)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Figure 7 - generalisation and classification
# --------------------------------------------------------------------------- #
def fig_generalisation(lomo: pd.DataFrame, clf: pd.DataFrame) -> plt.Figure:
    if lomo is None or len(lomo) == 0 or clf is None or len(clf) == 0:
        return _empty_figure("generalisation analysis unavailable "
                             "(needs >= 2 model families)")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.0),
                                  gridspec_kw={"width_ratios": [1.5, 1]})
    sub = lomo[lomo["endpoint"] == "synergy_area"]
    sets = list(dict.fromkeys(sub["predictors"]))
    models = _order_models(sub["held_out_model"].unique())
    x = np.arange(len(models))
    w = 0.8 / max(1, len(sets))
    for i, s in enumerate(sets):
        vals = [float(sub[(sub["predictors"] == s) & (sub["held_out_model"] == m)]
                      ["r2_out_of_sample"].mean()) for m in models]
        ax.bar(x + (i - (len(sets) - 1) / 2) * w, vals, width=w * 0.88,
               color=SERIES[i], label=s, edgecolor=SURFACE, linewidth=1.0)
    ax.axhline(0, color=INK3, lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([_lab(m) for m in models], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(r"out-of-sample $R^2$")
    ax.set_title("A  Predicting a held-out model family (endpoint: synergy area)")
    ax.legend(fontsize=7)

    tasks = list(clf["task"])
    x2 = np.arange(len(tasks))
    ax2.bar(x2 - 0.19, clf["accuracy"], width=0.36, color=SERIES[0], label="classifier",
            edgecolor=SURFACE, linewidth=1.2)
    ax2.bar(x2 + 0.19, clf["baseline"], width=0.36, color=INK3, label="majority baseline",
            edgecolor=SURFACE, linewidth=1.2)
    for i, (a, b) in enumerate(zip(clf["accuracy"], clf["baseline"], strict=True)):
        ax2.text(i - 0.19, a + 0.015, f"{a:.2f}", ha="center", fontsize=7.5, color=INK2)
        ax2.text(i + 0.19, b + 0.015, f"{b:.2f}", ha="center", fontsize=7.5, color=INK2)
    ax2.set_xticks(x2)
    ax2.set_xticklabels([t.replace("_", "\n") for t in tasks], fontsize=8)
    ax2.set_ylabel("accuracy")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("B  What can be recovered from information summaries?")
    ax2.legend(fontsize=8)
    fig.suptitle("Figure 7  Generalisation across model families",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Figure 8 - Sobol
# --------------------------------------------------------------------------- #
def fig_sobol(summary: pd.DataFrame, indices: pd.DataFrame) -> plt.Figure:
    if summary is None or len(summary) == 0:
        return _empty_figure("Sobol results unavailable")
    s = summary[summary["interaction_fraction"].notna()] if "interaction_fraction" in summary else summary
    models = _order_models(s["model_id"].unique())
    feats = list(dict.fromkeys(s["feature"]))
    M = np.full((len(models), len(feats)), np.nan)
    for i, m in enumerate(models):
        for j, f in enumerate(feats):
            v = s[(s["model_id"] == m) & (s["feature"] == f)]["interaction_fraction"]
            if len(v):
                M[i, j] = float(v.iloc[0])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.2),
                                  gridspec_kw={"width_ratios": [1.5, 1]})
    im = ax.imshow(M, cmap=SEQUENTIAL, vmin=0, vmax=max(0.6, np.nanmax(M) if np.isfinite(M).any() else 0.6),
                   aspect="auto")
    ax.set_xticks(range(len(feats)))
    ax.set_xticklabels([f.split(".")[-1].replace("_", "\n") for f in feats], fontsize=7)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([_lab(m) for m in models], fontsize=8)
    for i in range(len(models)):
        for j in range(len(feats)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if M[i, j] > 0.4 else INK)
    ax.grid(False)
    ax.set_title(r"A  Interaction fraction  $1-\sum S_1$")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    d = indices[indices["ST_minus_S1"].notna()] if "ST_minus_S1" in indices else pd.DataFrame()
    if len(d):
        for i, m in enumerate(models):
            g = d[d["model_id"] == m]
            ax2.scatter(g["S1"], g["ST"], s=16, color=SERIES[i % 8], label=_lab(m),
                        edgecolor=SURFACE, linewidth=0.6, alpha=0.85)
        lim = float(np.nanmax([d["ST"].max(), d["S1"].max(), 0.1])) * 1.05
        ax2.plot([0, lim], [0, lim], color=INK3, ls="--", lw=1.2)
        ax2.set_xlabel(r"first-order $S_1$")
        ax2.set_ylabel(r"total-order $S_T$")
        ax2.set_title(r"B  $S_T>S_1$ means the parameter acts through interactions")
        ax2.legend(fontsize=7, ncol=2)
    fig.suptitle("Figure 8  Variance-based functional sensitivity (unconditioned prior only)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Figure 9 - perturbation validation
# --------------------------------------------------------------------------- #
def fig_perturbation(pert: pd.DataFrame) -> plt.Figure:
    if pert is None or len(pert) == 0:
        return _empty_figure("perturbation results unavailable")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.0))
    kinds = ["top_synergy", "control"]
    colours = {"top_synergy": SERIES[0], "control": INK3}
    for i, k in enumerate(kinds):
        g = pert[pert["kind"] == k]
        vals = np.abs(pd.to_numeric(g["mean_nonadditivity_sd"], errors="coerce").dropna())
        if not len(vals):
            continue
        ax.scatter(np.full(len(vals), i) + np.random.default_rng(0).normal(0, 0.05, len(vals)),
                   vals, color=colours[k], s=26, alpha=0.8,
                   edgecolor=SURFACE, linewidth=0.7)
        ax.hlines(vals.mean(), i - 0.22, i + 0.22, color=INK, lw=2.4)
        ax.text(i, vals.mean() + 0.02, f"mean {vals.mean():.2f}", ha="center",
                fontsize=8, color=INK2)
    ax.axhline(0.25, color=STATUS["critical"], ls="--", lw=1.4)
    ax.text(1.42, 0.26, "preregistered\nthreshold 0.25 SD", fontsize=7.5,
            color=STATUS["critical"], va="bottom", ha="right")
    ax.set_xticks(range(len(kinds)))
    ax.set_xticklabels(["top-synergy tuples", "control tuples"], fontsize=9)
    ax.set_ylabel(r"|non-additivity $\eta$| (ensemble SD)")
    ax.set_title("A  Do statistical modules interact in the equations?")

    g = pert.dropna(subset=["o_information", "mean_nonadditivity_sd"])
    models = _order_models(g["model_id"].unique())
    for i, m in enumerate(models):
        gg = g[g["model_id"] == m]
        ax2.scatter(gg["o_information"], np.abs(gg["mean_nonadditivity_sd"]),
                    color=SERIES[i % 8], s=26, label=_lab(m),
                    edgecolor=SURFACE, linewidth=0.7)
    ax2.set_xlabel(r"tuple O-information $\Omega$ (more negative = more synergistic)")
    ax2.set_ylabel(r"|non-additivity $\eta$| (SD)")
    ax2.set_title("B  Statistical synergy vs functional non-additivity")
    ax2.legend(fontsize=7, ncol=2)
    fig.suptitle("Figure 9  Direct perturbation test of candidate synergy modules",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Figure 10 - geometry drives the endpoints
# --------------------------------------------------------------------------- #
def fig_geometry_vs_endpoint(endpoints: pd.DataFrame) -> plt.Figure:
    ep = endpoints[(endpoints["estimator"] == "gaussian") &
                   (endpoints["protocol_family"] == "absolute")].copy()
    ep["intrinsic_norm"] = ep["intrinsic_dimension"] / ep["n_params"]
    ep["effrank_norm"] = ep["effective_rank"] / ep["n_params"]
    panels = [("mean_abs_pearson", "mean |Pearson r|"),
              ("effrank_norm", "effective rank / n params"),
              ("intrinsic_norm", "intrinsic dimension / n params")]
    models = _order_models(ep["model_id"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.9))
    for k, (col, lab) in enumerate(panels):
        ax = axes[k]
        for i, m in enumerate(models):
            g = ep[ep["model_id"] == m]
            ax.scatter(g[col], g["net_o_information"], s=22, color=SERIES[i % 8],
                       label=_lab(m), edgecolor=SURFACE, linewidth=0.6, alpha=0.85)
        d = ep[[col, "net_o_information"]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(d) > 5:
            r = float(np.corrcoef(d[col], d["net_o_information"])[0, 1])
            sl, ic = np.polyfit(d[col], d["net_o_information"], 1)
            xs = np.linspace(d[col].min(), d[col].max(), 20)
            ax.plot(xs, sl * xs + ic, color=INK, lw=1.6, ls="--")
            ax.text(0.03, 0.95, f"r = {r:+.2f}  (n={len(d)})", transform=ax.transAxes,
                    fontsize=8.5, va="top", color=INK)
        ax.axhline(0, color=INK3, lw=1.0)
        ax.set_xlabel(lab)
        if k == 0:
            ax.set_ylabel("net O-information")
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Figure 10  Statistical geometry predicts the information endpoints across all models",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Figure 11 - module stability
# --------------------------------------------------------------------------- #
def fig_module_stability(stab: pd.DataFrame, tuple_scores: pd.DataFrame) -> plt.Figure:
    if stab is None or len(stab) == 0:
        return _empty_figure("module stability unavailable")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.0))
    models = _order_models(stab["model_id"].unique()) if len(stab) else []
    vals = [float(stab[stab["model_id"] == m]["mean_jaccard"].iloc[0]) for m in models]
    bars = ax.bar(range(len(models)), vals, color=SERIES[0],
                  edgecolor=SURFACE, linewidth=1.2)
    for b, v in zip(bars, vals, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}",
                ha="center", fontsize=8, color=INK2)
    ax.axhline(0.5, color=STATUS["critical"], ls="--", lw=1.4)
    ax.text(len(models) - 0.5, 0.52, "prereg. threshold 0.50", fontsize=7.5,
            color=STATUS["critical"], ha="right")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([_lab(m) for m in models], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("mean Jaccard of top-10 tuples")
    ax.set_ylim(0, 1.05)
    ax.set_title("A  Module stability across estimators")

    ts = tuple_scores[(tuple_scores["regime"] == "E3")]
    piv = ts.pivot_table(index=["model_id", "tuple"], columns="estimator",
                         values="o_information").dropna()
    if {"gaussian", "copula"}.issubset(piv.columns):
        mods = _order_models(piv.index.get_level_values(0).unique())
        for i, m in enumerate(mods):
            g = piv.loc[m]
            ax2.scatter(g["gaussian"], g["copula"], s=9, color=SERIES[i % 8],
                        alpha=0.6, label=_lab(m), edgecolor="none")
        lo = float(min(piv["gaussian"].min(), piv["copula"].min()))
        hi = float(max(piv["gaussian"].max(), piv["copula"].max()))
        ax2.plot([lo, hi], [lo, hi], color=INK3, ls="--", lw=1.2)
        r = float(np.corrcoef(piv["gaussian"], piv["copula"])[0, 1])
        ax2.text(0.03, 0.95, f"r = {r:+.3f}", transform=ax2.transAxes, fontsize=9, va="top")
        ax2.set_xlabel(r"$\Omega$ (Gaussian estimator)")
        ax2.set_ylabel(r"$\Omega$ (copula estimator)")
        ax2.set_title("B  Per-tuple agreement between estimators")
        ax2.legend(fontsize=7, ncol=2)
    fig.suptitle("Figure 11  Stability of candidate modules",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Figure 12 - decision table
# --------------------------------------------------------------------------- #
def fig_decision(decision: dict) -> plt.Figure:
    crit = decision["criteria"]
    n = len(crit)
    fig, ax = plt.subplots(figsize=(10.4, 0.44 * n + 1.9))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n + 1.6)
    for i, c in enumerate(reversed(crit)):
        y = i + 0.5
        col = STATUS["good"] if c["passed"] else INK3
        ax.add_patch(mpl.patches.Rectangle((0.005, y - 0.34), 0.028, 0.62,
                                           color=col, transform=ax.transData))
        ax.text(0.045, y, c["criterion"], fontsize=8.5, va="center", color=INK)
        ax.text(0.735, y, c["supports"], fontsize=8.5, va="center",
                color=SYNERGY if c["supports"] == "H-G" else SERIES[1],
                fontweight="bold")
        ax.text(0.80, y, f"{c['value']:.3f}", fontsize=8.5, va="center", color=INK2)
        ax.text(0.875, y, "PASS" if c["passed"] else "not met", fontsize=8.5,
                va="center", color=col, fontweight="bold")
    ax.text(0.045, n + 0.75, "criterion", fontsize=8.5, fontweight="bold", color=INK2)
    ax.text(0.735, n + 0.75, "supports", fontsize=8.5, fontweight="bold", color=INK2)
    ax.text(0.80, n + 0.75, "value", fontsize=8.5, fontweight="bold", color=INK2)
    ax.text(0.875, n + 0.75, "outcome", fontsize=8.5, fontweight="bold", color=INK2)
    concl = decision["conclusion"]
    ax.text(0.045, n + 1.35,
            f"H-G criteria passed: {decision['n_hg_passed']}/{decision['n_hg_total']}      "
            f"H-B criteria passed: {decision['n_hb_passed']}/{decision['n_hb_total']}      "
            f"conclusion: {concl}",
            fontsize=10, fontweight="bold", color=INK)
    fig.suptitle("Figure 12  Preregistered decision table",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
def fig_rsi(rsi: pd.DataFrame) -> plt.Figure:
    """Target-specific (RSI) synergy per electrical feature and regime."""
    if rsi is None or len(rsi) == 0:
        return _empty_figure("RSI results unavailable")
    d = rsi[rsi["estimator"] == "gaussian"]
    feats = list(dict.fromkeys(d["target"]))
    regimes = [r for r in ["E1", "E3", "E4"] if r in set(d["regime"])]
    fig, ax = plt.subplots(figsize=(10.6, 4.2))
    x = np.arange(len(feats))
    w = 0.8 / max(1, len(regimes))
    for i, reg in enumerate(regimes):
        vals, errs = [], []
        for f in feats:
            g = d[(d["target"] == f) & (d["regime"] == reg)]["frac_synergistic"]
            vals.append(float(g.mean()) if len(g) else np.nan)
            errs.append(float(g.std()) if len(g) > 1 else 0.0)
        ax.bar(x + (i - (len(regimes) - 1) / 2) * w, vals, width=w * 0.88,
               yerr=errs, capsize=2, color=SERIES[i], label=REGIME_LABEL.get(reg, reg),
               edgecolor=SURFACE, linewidth=1.1,
               error_kw={"ecolor": INK3, "elinewidth": 1})
    ax.axhline(0.5, color=INK3, ls="--", lw=1.2)
    ax.text(len(feats) - 0.4, 0.51, "no net bias", fontsize=7.5, color=INK2, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([f.split(".")[-1].replace("_", "\n") for f in feats], fontsize=7.5)
    ax.set_ylabel("fraction of order-3 tuples with RSI > 0\n(synergistic about the feature)")
    ax.legend(fontsize=8)
    ax.set_title("Figure 13  Target-specific information (RSI) per electrical feature,\n"
                 "mean over models",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig
