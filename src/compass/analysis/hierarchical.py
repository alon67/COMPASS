"""Cross-model variance partitioning and generalisation tests (blueprint 18).

The decision between H-G and H-B turns on one quantity: how much of the
variation in the information endpoints is explained by *sampling and geometry*
versus by *model identity*, once both are given a fair chance.  Three
complementary analyses are provided because no single one is decisive.

1. :func:`variance_partition` -- commonality analysis over predictor blocks.
   Reports each block's unique (semi-partial) contribution and the variance it
   shares with the other blocks, so that a large raw R^2 for model identity that
   is merely collinear with geometry cannot be mistaken for a biological effect.
2. :func:`leave_one_model_out` -- can an endpoint be predicted for a model family
   never seen in training?  Only sampling/geometry predictors can do this by
   construction, which is why blueprint 19.1 lists transfer to unseen families
   as evidence for H-G.
3. :func:`classification_test` -- can sampling regime, and separately model
   identity, be recovered from the information summaries alone?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "PREDICTOR_BLOCKS",
    "classification_test",
    "leave_one_model_out",
    "mixed_effects_components",
    "variance_partition",
]

PREDICTOR_BLOCKS = {
    "sampling": ["regime", "tolerance", "log_n_samples", "acceptance_rate"],
    "geometry": ["mean_abs_pearson", "effective_rank_norm", "intrinsic_dimension_norm",
                 "participation_ratio_norm", "pc1_variance_fraction"],
    "model_identity": ["model_id"],
    "architecture": ["n_params", "log_n_compartments", "is_point_model"],
}


def _design(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Numeric design matrix; categorical columns become dummies."""
    parts = []
    for c in cols:
        if c not in df.columns:
            continue
        s = df[c]
        if s.dtype == object or isinstance(s.dtype, pd.CategoricalDtype):
            d = pd.get_dummies(s, prefix=c, drop_first=True).to_numpy(dtype=float)
            if d.size:
                parts.append(d)
        else:
            v = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
            v = np.where(np.isfinite(v), v, np.nanmedian(v[np.isfinite(v)]) if np.isfinite(v).any() else 0.0)
            if np.nanstd(v) > 0:
                parts.append(((v - v.mean()) / v.std())[:, None])
    return np.hstack(parts) if parts else np.zeros((len(df), 0))


def _r2(y: np.ndarray, X: np.ndarray) -> float:
    """OLS R^2 with an intercept; 0 if there are no usable predictors."""
    if X.shape[1] == 0:
        return 0.0
    A = np.hstack([np.ones((len(y), 1)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return float(1.0 - np.sum(resid**2) / ss_tot) if ss_tot > 0 else 0.0


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Add the derived, scale-free covariates used by the hierarchical models."""
    out = df.copy()
    out["log_n_samples"] = np.log10(out["n_samples"].clip(lower=1))
    out["log_n_compartments"] = np.log10(out.get("n_compartments", pd.Series(1, index=out.index)).clip(lower=1))
    # normalise geometry measures by parameter count so that models with
    # different dimensionality are comparable
    for src, dst in (("effective_rank", "effective_rank_norm"),
                     ("participation_ratio", "participation_ratio_norm"),
                     ("intrinsic_dimension", "intrinsic_dimension_norm")):
        if src in out.columns:
            out[dst] = out[src] / out["n_params"]
    return out


def variance_partition(df: pd.DataFrame, endpoint: str,
                       blocks: dict | None = None) -> pd.DataFrame:
    """Commonality analysis: unique and shared R^2 for each predictor block."""
    blocks = blocks or PREDICTOR_BLOCKS
    d = prepare(df).dropna(subset=[endpoint])
    if len(d) < 20:
        return pd.DataFrame()
    y = d[endpoint].to_numpy(dtype=float)
    y = (y - y.mean()) / (y.std() if y.std() > 0 else 1.0)

    names = list(blocks)
    full = _r2(y, np.hstack([_design(d, blocks[b]) for b in names]))
    rows = []
    for b in names:
        alone = _r2(y, _design(d, blocks[b]))
        others = [x for x in names if x != b]
        without = _r2(y, np.hstack([_design(d, blocks[o]) for o in others])) if others else 0.0
        rows.append({
            "endpoint": endpoint,
            "block": b,
            "r2_alone": alone,
            "r2_unique": max(0.0, full - without),
            "r2_shared": max(0.0, alone - max(0.0, full - without)),
            "r2_full_model": full,
            "n": int(len(d)),
        })
    return pd.DataFrame(rows)


def mixed_effects_components(df: pd.DataFrame, endpoint: str) -> dict:
    """Mixed model with model family as a random effect (blueprint 18.2)."""
    import statsmodels.formula.api as smf

    d = prepare(df).dropna(subset=[endpoint]).copy()
    if d["model_id"].nunique() < 3 or len(d) < 30:
        return {"endpoint": endpoint, "error": "insufficient data"}
    d["y"] = (d[endpoint] - d[endpoint].mean()) / (d[endpoint].std() or 1.0)
    fixed = ("C(regime) + mean_abs_pearson + effective_rank_norm "
             "+ intrinsic_dimension_norm + log_n_samples")
    try:
        m = smf.mixedlm(f"y ~ {fixed}", d, groups=d["model_id"]).fit(reml=True)
    except Exception as exc:  # noqa: BLE001
        return {"endpoint": endpoint, "error": str(exc)[:120]}
    # Variance decomposition (Nakagawa & Schielzeth 2013).
    #
    # The fixed-effect variance MUST be computed from the fixed part alone.  An
    # earlier version used `np.var(m.fittedvalues)`, but statsmodels' MixedLM
    # `fittedvalues` already contain the random-effect BLUPs, so the group
    # variance was counted once inside `fitted_var` and again as `grp`.  That
    # inflates the denominator and the reported "ICC" of 0.048 was neither the
    # adjusted ICC nor the variance-partition coefficient -- it was not a
    # standard quantity at all.  Caught by the 2026-08-06 audit (M7).
    var_grp = float(m.cov_re.iloc[0, 0]) if m.cov_re.size else 0.0
    var_res = float(m.scale)
    var_fix = float(np.var(np.asarray(m.model.exog) @ np.asarray(m.fe_params)))
    total = var_fix + var_grp + var_res
    # Adjusted ICC: between-family share of the variance that remains AFTER the
    # fixed covariates have done their work.  This is the standard ICC and the
    # only one comparable with values quoted elsewhere in the literature.
    icc_adj = var_grp / (var_grp + var_res) if (var_grp + var_res) > 0 else np.nan
    return {
        "endpoint": endpoint,
        "var_fixed_effects": var_fix,
        "var_model_identity_random": var_grp,
        "var_residual": var_res,
        "icc_adjusted": icc_adj,
        "vpc_model_identity": var_grp / total if total > 0 else np.nan,
        "r2_marginal": var_fix / total if total > 0 else np.nan,
        "r2_conditional": (var_fix + var_grp) / total if total > 0 else np.nan,
        "n": int(len(d)),
        "n_models": int(d["model_id"].nunique()),
    }


def leave_one_model_out(df: pd.DataFrame, endpoint: str,
                        predictor_sets: dict | None = None) -> pd.DataFrame:
    """Predict a held-out model family's endpoints (blueprint 18.3).

    Model-identity dummies are unusable for an unseen family by construction, so
    a predictor set built only from sampling and geometry that still transfers
    is direct evidence that those factors, not neuron identity, carry the signal.
    """
    predictor_sets = predictor_sets or {
        "sampling_only": PREDICTOR_BLOCKS["sampling"],
        "geometry_only": PREDICTOR_BLOCKS["geometry"],
        "sampling+geometry": PREDICTOR_BLOCKS["sampling"] + PREDICTOR_BLOCKS["geometry"],
        "architecture_only": PREDICTOR_BLOCKS["architecture"],
        "all_but_identity": (PREDICTOR_BLOCKS["sampling"] + PREDICTOR_BLOCKS["geometry"]
                             + PREDICTOR_BLOCKS["architecture"]),
    }
    d = prepare(df).dropna(subset=[endpoint])
    rows = []
    models = sorted(d["model_id"].unique())
    for name, cols in predictor_sets.items():
        # build the design once on the pooled frame so dummy columns align
        X = _design(d, cols)
        y = d[endpoint].to_numpy(dtype=float)
        y = (y - y.mean()) / (y.std() or 1.0)
        for held in models:
            te = (d["model_id"] == held).to_numpy()
            tr = ~te
            if tr.sum() < 20 or te.sum() < 2 or X.shape[1] == 0:
                continue
            A = np.hstack([np.ones((tr.sum(), 1)), X[tr]])
            beta, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
            pred = np.hstack([np.ones((te.sum(), 1)), X[te]]) @ beta
            ss_res = float(np.sum((y[te] - pred) ** 2))
            ss_tot = float(np.sum((y[te] - y[tr].mean()) ** 2))
            rows.append({
                "endpoint": endpoint, "predictors": name, "held_out_model": held,
                "n_train": int(tr.sum()), "n_test": int(te.sum()),
                "rmse": float(np.sqrt(ss_res / te.sum())),
                "r2_out_of_sample": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
            })
    return pd.DataFrame(rows)


def classification_test(df: pd.DataFrame, feature_cols: list[str],
                        seed: int = 0) -> pd.DataFrame:
    """Predict sampling regime and model identity from information summaries.

    Regime is evaluated with *leave-one-model-family-out* CV so that success
    means the regime signature generalises to unseen model families; identity is
    evaluated with stratified CV within families.  Both are compared against the
    majority-class baseline (blueprint 18.4 -- secondary, nested CV).
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold

    d = prepare(df).copy()
    cols = [c for c in feature_cols if c in d.columns]
    X = d[cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median()).to_numpy(dtype=float)
    rows = []

    # (a) regime, leave-one-model-family-out
    y = d["regime"].to_numpy()
    groups = d["model_id"].to_numpy()
    accs, base = [], []
    for held in sorted(set(groups)):
        te = groups == held
        tr = ~te
        if tr.sum() < 30 or te.sum() < 3 or len(set(y[tr])) < 2:
            continue
        clf = RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=-1)
        clf.fit(X[tr], y[tr])
        accs.append(float((clf.predict(X[te]) == y[te]).mean()))
        vals, cnt = np.unique(y[tr], return_counts=True)
        base.append(float((y[te] == vals[cnt.argmax()]).mean()))
    if accs:
        rows.append({"task": "sampling_regime", "cv": "leave_one_model_family_out",
                     "accuracy": float(np.mean(accs)), "accuracy_sd": float(np.std(accs)),
                     "baseline": float(np.mean(base)), "n_folds": len(accs),
                     "n_classes": int(len(set(y)))})

    # (b) model identity, stratified CV
    y2 = d["model_id"].to_numpy()
    if len(set(y2)) > 1:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        accs2, base2 = [], []
        for tr, te in skf.split(X, y2):
            clf = RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=-1)
            clf.fit(X[tr], y2[tr])
            accs2.append(float((clf.predict(X[te]) == y2[te]).mean()))
            vals, cnt = np.unique(y2[tr], return_counts=True)
            base2.append(float((y2[te] == vals[cnt.argmax()]).mean()))
        rows.append({"task": "model_identity", "cv": "stratified_5fold",
                     "accuracy": float(np.mean(accs2)), "accuracy_sd": float(np.std(accs2)),
                     "baseline": float(np.mean(base2)), "n_folds": 5,
                     "n_classes": int(len(set(y2)))})
    return pd.DataFrame(rows)
