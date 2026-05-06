"""
Ranked Probability Score utilities for ordinal classification, and
sklearn-compatible scorers for use with cross_val_score / GridSearchCV.

The metric functions follow the sklearn metric signature
    metric(y_true, y_proba, *, classes=None, ...)
and can be wrapped by sklearn.metrics.make_scorer. Pre-built scorer
objects are also exported for direct use.

IMPORTANT for longitudinal panel data: cross-validation MUST split by
SUBJECT, not by row. Use sklearn.model_selection.GroupKFold (or
LeaveOneGroupOut, StratifiedGroupKFold, etc.) with `groups=` set to the
subject id column. Standard KFold leaks within-subject correlation.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from sklearn.metrics import make_scorer


# ---------------- core metric ----------------

def _prepare(y_true, y_proba, classes):
    """Validate inputs and return (y_rank, P, K)."""
    y_true = np.asarray(y_true)
    P = np.asarray(y_proba, dtype=float)
    if P.ndim != 2:
        raise ValueError(f"y_proba must be 2-D; got shape {P.shape}.")
    if P.shape[0] != len(y_true):
        raise ValueError(
            f"y_proba rows ({P.shape[0]}) != len(y_true) ({len(y_true)})."
        )

    if classes is None:
        if not np.issubdtype(y_true.dtype, np.number):
            raise ValueError(
                "Cannot infer class order from non-numeric y_true; "
                "pass `classes=` explicitly."
            )
        classes = np.unique(y_true)
        # If predict_proba has more columns than y_true has unique classes,
        # we cannot infer column ordering — require explicit classes.
        if len(classes) != P.shape[1]:
            raise ValueError(
                f"Cannot infer class order: y_true has {len(classes)} unique "
                f"values but y_proba has {P.shape[1]} columns. Pass `classes=` "
                "explicitly (must be the ordered list matching y_proba columns)."
            )
    else:
        classes = np.asarray(classes)
        if len(classes) != P.shape[1]:
            raise ValueError(
                f"len(classes) {len(classes)} != y_proba columns {P.shape[1]}."
            )

    rank_map = {c: i for i, c in enumerate(classes.tolist())}
    try:
        y_rank = np.array([rank_map[v] for v in y_true.tolist()], dtype=int)
    except KeyError as e:
        raise ValueError(
            f"y_true contains a value not in classes: {e}. classes={list(classes)}"
        )
    return y_rank, P, len(classes)


def _per_obs_rps(y_rank: np.ndarray, P: np.ndarray, K: int) -> np.ndarray:
    """Per-observation ranked probability score, shape (n,)."""
    cum_pred = np.cumsum(P, axis=1)
    cum_obs = (np.arange(K)[None, :] >= y_rank[:, None]).astype(float)
    return np.sum((cum_pred - cum_obs) ** 2, axis=1)


def ranked_probability_score(
    y_true,
    y_proba,
    *,
    classes: Optional[Sequence] = None,
    sample_weight: Optional[Sequence[float]] = None,
) -> float:
    """
    Plain Ranked Probability Score (lower is better; 0 is perfect).

    For each observation with rank-encoded true class c in {0, ..., K-1}
    and predicted class probabilities p_0, ..., p_{K-1},

        RPS_i = sum_{m=0}^{K-1} ( sum_{j<=m} p_j  -  1[c <= m] )^2

    The returned value is the (optionally weighted) mean over observations.

    Parameters
    ----------
    y_true : array-like, shape (n,)
        Observed class labels.
    y_proba : array-like, shape (n, K)
        Predicted class probabilities; columns must be in `classes` order.
    classes : array-like, optional
        Ordered class labels matching y_proba columns. If None, inferred
        from sorted np.unique(y_true) — only safe when y_true is numeric
        AND all K classes are represented in y_true. Pass explicitly otherwise.
    sample_weight : array-like, optional
        Per-observation weights. Useful for cost-weighting (e.g., upweight
        observations of rare classes for diagnostics). Note: weighting
        breaks the proper-scoring-rule guarantee — use plain RPS for honest
        model selection, weighted RPS only for diagnostics or when the
        weights reflect a real cost structure.

    Returns
    -------
    float
        Mean (weighted) RPS over observations.
    """
    y_rank, P, K = _prepare(y_true, y_proba, classes)
    rps = _per_obs_rps(y_rank, P, K)
    if sample_weight is None:
        return float(rps.mean())
    w = np.asarray(sample_weight, dtype=float)
    if len(w) != len(rps):
        raise ValueError(f"sample_weight length {len(w)} != n {len(rps)}.")
    return float(np.average(rps, weights=w))


def macro_ranked_probability_score(
    y_true,
    y_proba,
    *,
    classes: Optional[Sequence] = None,
) -> float:
    """
    Macro-averaged Ranked Probability Score: per-class mean RPS averaged
    over classes with equal weight (lower is better).

    Computes RPS_i for each observation i, groups by the true class c,
    averages within each class, then averages those K class-means with
    equal weight. Compared to plain RPS, this stops populated classes
    from dominating the score and surfaces rare-class miscalibration.

    Classes that do not appear in y_true are skipped (not given zero
    weight) — the macro-average is over OBSERVED classes only.

    NOTE: macro-RPS is not a proper scoring rule globally. It's intended
    for diagnostics and class-balanced model selection, not for honest
    probability estimation. For probability calibration, use plain RPS.

    Returns
    -------
    float
        Equal-weighted average of per-class mean RPS over observed classes.
    """
    y_rank, P, K = _prepare(y_true, y_proba, classes)
    rps = _per_obs_rps(y_rank, P, K)
    per_class = []
    for k in range(K):
        mask = y_rank == k
        if mask.any():
            per_class.append(rps[mask].mean())
    if not per_class:
        raise ValueError("y_true is empty; cannot compute macro RPS.")
    return float(np.mean(per_class))


def per_class_ranked_probability_score(
    y_true,
    y_proba,
    *,
    classes: Optional[Sequence] = None,
) -> dict:
    """
    Diagnostic helper: per-class mean RPS as a dict {class_label: rps}.
    Returns NaN for classes absent from y_true. Not intended for
    cross_val_score (which expects a scalar) — use as a post-hoc breakdown.
    """
    y_rank, P, K = _prepare(y_true, y_proba, classes)
    rps = _per_obs_rps(y_rank, P, K)
    cls_arr = np.asarray(classes) if classes is not None else np.unique(y_true)
    out = {}
    for k, c in enumerate(cls_arr.tolist()):
        mask = y_rank == k
        out[c] = float(rps[mask].mean()) if mask.any() else float("nan")
    return out


# ---------------- sklearn scorer wrappers ----------------

def make_rps_scorer(classes: Optional[Sequence] = None):
    """
    Build an sklearn scorer for plain RPS (lower is better).

    If `classes` is None, the scorer infers classes from y_true at
    evaluation time — only safe when y_true contains all K classes
    AND is numeric. Pass `classes` explicitly otherwise (recommended).

    Use with cross_val_score, GridSearchCV, etc. Cross-validation MUST
    split by subject — pass GroupKFold(...).split(X, y, groups=subject_ids)
    or equivalent.

    Examples
    --------
    >>> from sklearn.model_selection import cross_val_score, GroupKFold
    >>> scorer = make_rps_scorer(classes=[0, 1, 2, 3])
    >>> cv = GroupKFold(n_splits=5)
    >>> scores = cross_val_score(
    ...     model, X_train, y_train,
    ...     scoring=scorer, cv=cv, groups=X_train['subj_id']
    ... )
    >>> print(-scores.mean())  # negate because greater_is_better=False
    """
    return make_scorer(
        ranked_probability_score,
        response_method="predict_proba",
        greater_is_better=False,
        classes=classes,
    )


def make_macro_rps_scorer(classes: Optional[Sequence] = None):
    """
    Build an sklearn scorer for macro-RPS (per-class equal-weight, lower is better).
    See `make_rps_scorer` for usage.
    """
    return make_scorer(
        macro_ranked_probability_score,
        response_method="predict_proba",
        greater_is_better=False,
        classes=classes,
    )


# ---------------- pre-built default scorers (infer classes from y_true) ----------------
# These work out-of-the-box for numeric ordinal targets where all classes
# are represented in y_true; for non-numeric labels or sparse classes,
# build a custom scorer with make_rps_scorer(classes=...).

rps_scorer = make_rps_scorer()
macro_rps_scorer = make_macro_rps_scorer()