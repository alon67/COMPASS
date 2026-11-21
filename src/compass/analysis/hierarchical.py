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

import json

import numpy as np
import pandas as pd

__all__ = [
    "PREDICTOR_BLOCKS",
    "classification_test",
    "FoldPipeline",
    "cluster_bootstrap",
    "fold_distribution",
    "leave_one_lineage_out",
    "leave_one_model_out",
    "mixed_effects_components",
    "nested_variance_components",
    "variance_partition",
]

PREDICTOR_BLOCKS = {
    "sampling": ["regime", "tolerance", "log_n_samples", "acceptance_rate"],
    # Intrinsic dimension enters twice, from two estimators with different bias
    # behaviour, so the block cannot rest on one algorithm's quirks (audit 3.4
    # point 5).
    "geometry": ["mean_abs_pearson", "effective_rank_norm", "intrinsic_dimension_norm",
                 "intrinsic_dimension_mle_norm", "participation_ratio_norm",
                 "pc1_variance_fraction"],
    # The unit of identity is the SOURCE LINEAGE, not the model id.  Eight model
    # ids over six lineages made "model identity" partly a within-lineage
    # contrast, which is not what the H-B hypothesis is about (audit finding 4).
    "model_identity": ["source_lineage_id"],
    "implementation": ["model_id"],
    "architecture": ["n_params", "log_n_compartments", "is_point_model"],
}


def _design(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Numeric design matrix; categorical columns become dummies.

    For IN-SAMPLE fits only (variance partitioning).  Anything that scores a
    held-out fold must use :class:`FoldPipeline`, which learns the same
    transformations from the training rows alone.
    """
    p = FoldPipeline(cols).fit(df)
    return p.transform(df)


class FoldPipeline:
    """Learned preprocessing fitted on ONE training fold and applied elsewhere.

    Every step below is *estimated from data* and therefore has to be estimated
    inside the fold (2026-08-09 audit, 4.2): the pooled version of this code
    chose dummy levels, imputation medians, standardisation constants and the
    zero-variance drop list from the full dataset, including the rows it was
    about to predict.

    Two of those leaks are inert for an OLS R2 and two are not, and the
    distinction is worth stating precisely rather than claiming a blanket fix:

    * centring and scaling a predictor is an affine change absorbed by the OLS
      coefficients, so it leaves fitted values -- and R2 -- unchanged.  It does
      change the reported RMSE, which is quoted in standardised units.
    * median imputation is **not** affine.  A held-out lineage with missing
      geometry received a median computed partly from itself.
    * a predictor that is constant within the training fold but varies in the
      held-out fold used to survive the pooled variance filter, entering the fit
      perfectly collinear with the intercept and leaving an arbitrary
      coefficient to multiply a nonzero test value.
    * ``pd.get_dummies(..., drop_first=True)`` on the pooled frame creates a
      column for a level that occurs *only* in the held-out fold.  It is
      identically zero in training and one in test, so the prediction is shifted
      by a coefficient no training row could constrain.

    Levels unseen in training map to the all-zero (reference) row, which is the
    only defensible encoding for a category the model has never observed.
    """

    def __init__(self, cols: list[str]):
        self.cols = list(cols)
        self.plan: list[tuple] = []

    def fit(self, train: pd.DataFrame) -> FoldPipeline:
        self.plan = []
        for c in self.cols:
            if c not in train.columns:
                continue
            s = train[c]
            # pandas >= 3 stores text in a dedicated string dtype, not object; testing
            # for object alone sent text columns down the numeric branch, where they
            # coerced to NaN, had zero variance and were silently dropped.
            if (s.dtype == object or isinstance(s.dtype, pd.CategoricalDtype)
                    or pd.api.types.is_string_dtype(s.dtype)):
                levels = sorted(pd.Series(s).dropna().unique().tolist())
                if len(levels) > 1:                     # drop_first: [0] is the reference
                    self.plan.append(("cat", c, levels[1:]))
            else:
                v = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
                ok = np.isfinite(v)
                med = float(np.median(v[ok])) if ok.any() else 0.0
                filled = np.where(ok, v, med)
                sd = float(filled.std())
                if sd > 0:
                    self.plan.append(("num", c, (med, float(filled.mean()), sd)))
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        parts = []
        for kind, c, spec in self.plan:
            if kind == "cat":
                s = df[c].astype(object)
                parts.append(np.column_stack([(s == lv).to_numpy(dtype=float)
                                              for lv in spec]))
            else:
                med, mu, sd = spec
                v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
                v = np.where(np.isfinite(v), v, med)
                parts.append(((v - mu) / sd)[:, None])
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
                     ("intrinsic_dimension", "intrinsic_dimension_norm"),
                     ("intrinsic_dimension_mle", "intrinsic_dimension_mle_norm")):
        if src in out.columns:
            out[dst] = out[src] / out["n_params"]
    if "source_lineage_id" not in out.columns:
        from .cohorts import lineage_column

        out["source_lineage_id"] = lineage_column(out)
    return out


def cluster_bootstrap(values: np.ndarray, clusters: np.ndarray,
                      statistic=np.mean, n_boot: int = 5000,
                      seed: int = 20260806,
                      equal_cluster_weight: bool = True) -> dict:
    """Resample whole CLUSTERS, then rows within them, weighting clusters equally.

    The previous bootstrap resampled eight model ids as if they were eight
    independent units.  Two pairs of those ids share a source lineage, so the
    resampling distribution was narrower than the data support (audit finding
    4, section 7.6).  Resampling lineages fixes the unit; resampling rows within
    a drawn lineage keeps the within-lineage variability.

    Fixing the resampling unit was not, on its own, enough.  The statistic was
    still computed by pooling every row from every drawn cluster, so a lineage
    contributing two implementations put twice as many rows into the mean as a
    lineage contributing one -- the HH and Pospischil lineages carried double
    weight in both the point estimate and every replicate, which is the same
    over-counting the cluster resampling was introduced to remove
    (post-response audit, section 4).

    With ``equal_cluster_weight`` the statistic is applied within each cluster
    first and then averaged across clusters, so every lineage counts once
    however many implementations it happens to contain.  Set it False only to
    reproduce a pooled-row figure deliberately.
    """
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    idx_by = {c: np.flatnonzero(clusters == c) for c in uniq}

    def _combine(per_cluster: list[np.ndarray]) -> float:
        """One number from the drawn clusters' row sets."""
        if equal_cluster_weight:
            means = [float(statistic(v)) for v in per_cluster if v.size]
            return float(np.mean(means)) if means else float("nan")
        v = np.concatenate(per_cluster) if per_cluster else np.array([])
        return float(statistic(v)) if v.size else float("nan")

    stats = []
    for _ in range(n_boot):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        per = []
        for c in drawn:
            take = rng.choice(idx_by[c], size=len(idx_by[c]), replace=True)
            v = values[take]
            per.append(v[np.isfinite(v)])
        s = _combine(per)
        if np.isfinite(s):
            stats.append(s)
    if not stats:
        return {"point": float("nan"), "ci95": [float("nan")] * 2,
                "n_clusters": len(uniq)}
    obs = [values[idx_by[c]][np.isfinite(values[idx_by[c]])] for c in uniq]
    return {
        "point": _combine(obs),
        "ci95": [float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))],
        "sd": float(np.std(stats, ddof=1)),
        "n_clusters": int(len(uniq)),
        "n_rows": int(len(values)),
        "equal_cluster_weight": bool(equal_cluster_weight),
        "rows_per_cluster": {str(c): int(len(idx_by[c])) for c in uniq},
        "clusters": sorted(map(str, uniq)),
    }


def nested_variance_components(df: pd.DataFrame, endpoint: str) -> dict:
    """Variance at class / lineage / implementation level (audit 9.6 item 9).

    A flat "model identity" random effect cannot distinguish "different neuron
    classes behave differently" from "two implementations of the same source
    disagree".  Those are different scientific claims with different evidential
    requirements, so they get different variance components.
    """
    import statsmodels.formula.api as smf

    d = prepare(df).dropna(subset=[endpoint]).copy()
    need = {"source_lineage_id", "model_id"}
    if not need <= set(d.columns) or d["source_lineage_id"].nunique() < 3:
        return {"endpoint": endpoint, "error": "insufficient lineage structure"}
    d["y"] = (d[endpoint] - d[endpoint].mean()) / (d[endpoint].std() or 1.0)
    fixed = ("C(regime) + mean_abs_pearson + effective_rank_norm "
             "+ intrinsic_dimension_norm + log_n_samples")
    out: dict = {"endpoint": endpoint,
                 "n": int(len(d)),
                 "n_lineages": int(d["source_lineage_id"].nunique()),
                 "n_implementations": int(d["model_id"].nunique())}
    for level, col in (("lineage", "source_lineage_id"),
                       ("implementation", "model_id")):
        try:
            m = smf.mixedlm(f"y ~ {fixed}", d, groups=d[col]).fit(reml=True)
        except Exception as exc:  # noqa: BLE001
            out[f"{level}_error"] = str(exc)[:120]
            continue
        var_grp = float(m.cov_re.iloc[0, 0]) if m.cov_re.size else 0.0
        var_res = float(m.scale)
        var_fix = float(np.var(np.asarray(m.model.exog) @ np.asarray(m.fe_params)))
        total = var_fix + var_grp + var_res
        converged = bool(getattr(m, "converged", False))
        out[level] = {
            "var_group": var_grp, "var_residual": var_res, "var_fixed": var_fix,
            "icc_adjusted": var_grp / (var_grp + var_res) if (var_grp + var_res) > 0 else np.nan,
            "vpc": var_grp / total if total > 0 else np.nan,
            "n_groups": int(d[col].nunique()),
            # Production diagnostics, not just point estimates (audit section 1).
            "converged": converged,
            "method": str(getattr(m, "method", "reml")),
            "loglike": float(m.llf) if np.isfinite(m.llf) else None,
            # The convergence flag was recorded and then ignored: the lineage
            # level of this very model does NOT converge on the primary cohort,
            # and its ICC was being quoted in docs/RESULTS.md as a number, next
            # to a sentence asserting that both levels had converged. A variance
            # component from a fit that did not converge is not an estimate of
            # anything, so the record now says so in a field a reader cannot
            # miss rather than in a boolean three keys away.
            "quotable": converged,
            "quotability_note": (
                "" if converged else
                f"the {level}-level mixed model did NOT converge on "
                f"{d[col].nunique()} groups; icc_adjusted and vpc are reported "
                f"for completeness and must not be quoted as estimates"),
        }
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


def leave_one_lineage_out(df: pd.DataFrame, endpoint: str,
                          predictor_sets: dict | None = None) -> pd.DataFrame:
    """Hold out a whole SOURCE LINEAGE, not one model id.

    The audit's finding 4 in its most consequential form: "'leave-one-model-
    family-out' holds out IDs, so a related lineage may remain in training".
    Holding out ``hh_point`` while ``hh_cable`` -- same channels, same lineage,
    one extra compartment -- stays in the training set does not test transfer to
    an unseen family at all.
    """
    return leave_one_model_out(df, endpoint, predictor_sets, group_col="source_lineage_id")


def leave_one_model_out(df: pd.DataFrame, endpoint: str,
                        predictor_sets: dict | None = None,
                        group_col: str = "model_id") -> pd.DataFrame:
    """Predict a held-out group's endpoints (blueprint 18.3).

    Model-identity dummies are unusable for an unseen family by construction, so
    a predictor set built only from sampling and geometry that still transfers
    is direct evidence that those factors, not neuron identity, carry the signal.

    ``group_col`` selects the held-out unit.  Use ``source_lineage_id`` for the
    scientifically meaningful fold (see :func:`leave_one_lineage_out`);
    ``model_id`` is retained for backward comparison and is reported as an
    implementation-level fold, not a family-level one.
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
    if group_col not in d.columns:
        raise KeyError(
            f"leave_one_model_out: grouping column {group_col!r} is not in the "
            f"frame. Refusing to fall back to 'model_id': a fold silently built "
            f"on implementations while labelled as lineages is exactly the "
            f"defect this argument exists to prevent.")
    rows = []
    groups = sorted(d[group_col].unique())
    raw = d[endpoint].to_numpy(dtype=float)
    for name, cols in predictor_sets.items():
        for held in groups:
            te = (d[group_col] == held).to_numpy()
            tr = ~te
            if tr.sum() < 20 or te.sum() < 2:
                continue
            # EVERY learned transformation is fitted on the training rows only:
            # dummy levels, imputation medians, standardisation constants, the
            # zero-variance drop list, and the endpoint's own centring and
            # scaling (2026-08-09 audit, 4.2).
            pipe = FoldPipeline(cols).fit(d[tr])
            Xtr, Xte = pipe.transform(d[tr]), pipe.transform(d[te])
            if Xtr.shape[1] == 0:
                continue
            mu, sd = float(raw[tr].mean()), float(raw[tr].std() or 1.0)
            ytr, yte = (raw[tr] - mu) / sd, (raw[te] - mu) / sd
            A = np.hstack([np.ones((tr.sum(), 1)), Xtr])
            beta, *_ = np.linalg.lstsq(A, ytr, rcond=None)
            pred = np.hstack([np.ones((te.sum(), 1)), Xte]) @ beta
            ss_res = float(np.sum((yte - pred) ** 2))
            # Baseline: predicting the TRAINING mean for every held-out row.  A
            # positive score means "beats that prediction for a new group"; it
            # is not variance explained within the held-out group, which
            # r2_vs_heldout_mean (held-out mean as the baseline) describes.
            ss_tot = float(np.sum((yte - ytr.mean()) ** 2))
            ss_own = float(np.sum((yte - yte.mean()) ** 2))
            rows.append({
                "endpoint": endpoint, "predictors": name,
                # the unit actually held out, and what kind of unit it is, so a
                # stored fold can never again be described as something it isn't
                "held_out_group": held, "group_col": group_col,
                "n_train": int(tr.sum()), "n_test": int(te.sum()),
                "n_predictor_columns": int(Xtr.shape[1]),
                "preprocessing": "fold_local",
                "rmse": float(np.sqrt(ss_res / te.sum())),
                "r2_out_of_sample": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
                "r2_vs_heldout_mean": float(1.0 - ss_res / ss_own) if ss_own > 0 else np.nan,
            })
    return pd.DataFrame(rows)


def fold_distribution(lomo: pd.DataFrame, predictors: str,
                      metric: str = "r2_out_of_sample") -> dict:
    """Summarise a cross-validation as its whole fold distribution.

    A median over five folds is not evidence of reliable transfer, and quoting
    one hides the fold that matters: the ``migliore_1999`` lineage scores about
    -3.8 while the median is about 0.56 (2026-08-09 audit, 4.2).  Every fold, the
    worst fold, and the fraction of folds that beat the intercept-only null are
    reported together so the spread cannot be dropped in transcription.
    """
    sub = lomo[lomo["predictors"] == predictors] if "predictors" in lomo else lomo
    v = pd.to_numeric(sub[metric], errors="coerce").to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n_folds": 0}
    folds = {str(g): (None if not np.isfinite(x) else float(x))
             for g, x in zip(sub["held_out_group"], sub[metric], strict=True)}
    # `worst_fold` used to be `min(folds, key=folds.get)` over a dict that keeps
    # non-finite folds as None, so a single non-converged fold raised
    # TypeError comparing None with float -- in the one function whose purpose is
    # to report unstable folds (2026-08-09 post-response-3 audit, P1-5).  Rank on
    # the finite folds and name the non-finite ones separately, because "this
    # fold did not produce a number" is a different statement from "this fold
    # scored badly" and the reader needs both.
    finite = {k: x for k, x in folds.items() if x is not None}
    nonfinite = sorted(k for k, x in folds.items() if x is None)
    return {
        "n_folds": int(v.size),
        "n_folds_nonfinite": len(nonfinite),
        "nonfinite_folds": nonfinite,
        "median": float(np.median(v)),
        "mean": float(np.mean(v)),
        "min": float(v.min()),
        "max": float(v.max()),
        "iqr": [float(np.percentile(v, 25)), float(np.percentile(v, 75))],
        "range": [float(v.min()), float(v.max())],
        "n_folds_beating_null": int((v > 0).sum()),
        "worst_fold": (min(finite, key=lambda k: finite[k]) if finite else None),
        "folds": folds,
    }


def classification_test(df: pd.DataFrame, feature_cols: list[str],
                        seed: int = 0,
                        group_col: str = "source_lineage_id") -> pd.DataFrame:
    """Predict sampling regime and model identity from information summaries.

    Regime is evaluated with *leave-one-LINEAGE-out* CV so that success means the
    regime signature generalises to an unseen source lineage; identity is
    evaluated with stratified CV within families.  Both are compared against the
    majority-class baseline (blueprint 18.4 -- secondary, nested CV).

    ``group_col`` defaults to ``source_lineage_id``.  It used to be hard-coded to
    ``model_id`` while the row was labelled a family fold, which left sibling
    implementations (``hh_point``/``hh_cable``, ``pospischil_rs``/``_fs``) in the
    training set of a fold advertised as holding out their family.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold

    d = prepare(df).copy()
    if group_col not in d.columns:
        raise KeyError(
            f"classification_test: grouping column {group_col!r} is not in the "
            f"frame; refusing to fall back to 'model_id'.")
    cols = [c for c in feature_cols if c in d.columns]
    raw = d[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    def fold_local(tr: np.ndarray, te: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Impute from the TRAINING rows only (2026-08-09 audit, 4.2)."""
        med = np.nanmedian(np.where(np.isfinite(raw[tr]), raw[tr], np.nan), axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        fill = lambda M: np.where(np.isfinite(M), M, med)      # noqa: E731
        return fill(raw[tr]), fill(raw[te])

    rows = []

    # (a) regime, leave-one-lineage-out (or whatever group_col names)
    y = d["regime"].to_numpy()
    groups = d[group_col].to_numpy()
    accs, base, per_fold = [], [], {}
    for held in sorted(set(groups)):
        te = groups == held
        tr = ~te
        if tr.sum() < 30 or te.sum() < 3 or len(set(y[tr])) < 2:
            continue
        Xtr, Xte = fold_local(tr, te)
        clf = RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=-1)
        clf.fit(Xtr, y[tr])
        acc = float((clf.predict(Xte) == y[te]).mean())
        accs.append(acc)
        vals, cnt = np.unique(y[tr], return_counts=True)
        base.append(float((y[te] == vals[cnt.argmax()]).mean()))
        per_fold[str(held)] = round(acc, 4)
    if accs:
        rows.append({"task": "sampling_regime",
                     "cv": f"leave_one_{group_col}_out", "group_col": group_col,
                     "accuracy": float(np.mean(accs)), "accuracy_sd": float(np.std(accs)),
                     "baseline": float(np.mean(base)), "n_folds": len(accs),
                     "accuracy_min": float(np.min(accs)), "accuracy_max": float(np.max(accs)),
                     "per_fold_accuracy": json.dumps(per_fold),
                     "preprocessing": "fold_local",
                     "n_classes": int(len(set(y)))})

    # (b) model identity, stratified CV
    y2 = d["model_id"].to_numpy()
    if len(set(y2)) > 1:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        accs2, base2 = [], []
        for tr_i, te_i in skf.split(raw, y2):
            tr = np.zeros(len(d), dtype=bool)
            te = np.zeros(len(d), dtype=bool)
            tr[tr_i], te[te_i] = True, True
            Xtr, Xte = fold_local(tr, te)
            clf = RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=-1)
            clf.fit(Xtr, y2[tr_i])
            accs2.append(float((clf.predict(Xte) == y2[te_i]).mean()))
            vals, cnt = np.unique(y2[tr_i], return_counts=True)
            base2.append(float((y2[te_i] == vals[cnt.argmax()]).mean()))
        rows.append({"task": "model_identity", "cv": "stratified_5fold",
                     "accuracy": float(np.mean(accs2)), "accuracy_sd": float(np.std(accs2)),
                     "baseline": float(np.mean(base2)), "n_folds": 5,
                     "accuracy_min": float(np.min(accs2)), "accuracy_max": float(np.max(accs2)),
                     "preprocessing": "fold_local",
                     "n_classes": int(len(set(y2)))})
    return pd.DataFrame(rows)
