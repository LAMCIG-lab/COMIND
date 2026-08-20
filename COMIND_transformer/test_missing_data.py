"""Verification checks for masked-likelihood missing-biomarker support."""
from __future__ import annotations

import warnings

import numpy as np

from COMIND_transformer.assignments import sse_matrix
from COMIND_transformer.model_selection import compute_sse_per_biomarker
from COMIND_transformer.optimizer_beta import reconstruction_sse
from COMIND_transformer.subtyping_em_transformer import SubtypingEM
from COMIND_transformer.utils import split_observed, solve_system


def _assert_close(a, b, msg, rtol=1e-12, atol=1e-12):
    if not np.allclose(a, b, rtol=rtol, atol=atol):
        raise AssertionError(f"{msg}: {a!r} vs {b!r}")


def test_split_observed():
    X = np.array([[1.0, np.nan], [0.5, 2.0], [np.inf, 3.0]])
    filled, w = split_observed(X)
    expected_w = np.isfinite(X).astype(float)
    _assert_close(w, expected_w, "obs_weight mismatch")
    if np.any(~np.isfinite(filled)):
        raise AssertionError("X_filled still has non-finite values")
    observed = expected_w > 0
    _assert_close(filled[observed], X[observed], "observed values not preserved")
    _assert_close(filled[~observed], 0.0, "missing entries not filled with 0")
    print("PASS 1: split_observed")


def test_reconstruction_sse_masking():
    rng = np.random.default_rng(0)
    n_b, n_t, n_span = 3, 4, 50
    t_span = np.linspace(0.0, 5.0, n_span)
    dt = np.array([0.0, 0.5, 1.0, 1.5])
    s = np.ones(n_b)
    X_pred = np.clip(np.cumsum(rng.uniform(0.01, 0.05, size=(n_b, n_span)), axis=1), 0, 0.99)
    X_obs = rng.uniform(0.1, 0.8, size=(n_t, n_b))
    beta = 1.2

    miss_t, miss_b = 2, 1
    X_nan = X_obs.copy()
    X_nan[miss_t, miss_b] = np.nan
    X_filled, w = split_observed(X_nan)

    sse_masked = reconstruction_sse(beta, X_filled, dt, X_pred, t_span, s, obs_weight_i=w)

    t_pred = dt + beta
    X_interp = np.array([np.interp(t_pred, t_span, s[b] * X_pred[b]) for b in range(n_b)])
    residuals = X_filled.T - X_interp
    residuals[miss_b, miss_t] = 0.0
    sse_hand = float(np.sum(residuals ** 2))
    _assert_close(sse_masked, sse_hand, "masked SSE != hand-zeroed residual")

    X_imputed = X_filled.copy()
    X_imputed[miss_t, miss_b] = X_interp[miss_b, miss_t]
    sse_imputed = reconstruction_sse(beta, X_imputed, dt, X_pred, t_span, s)
    _assert_close(sse_masked, sse_imputed, "masking != self-consistent imputation")
    print("PASS 2+3: reconstruction_sse masking / EM-equivalence")


def test_fully_missing_row():
    rng = np.random.default_rng(1)
    n_b, n_t, n_span = 3, 4, 50
    t_span = np.linspace(0.0, 5.0, n_span)
    dt = np.array([0.0, 0.5, 1.0, 1.5])
    s = np.ones(n_b)
    X_pred = np.clip(np.cumsum(rng.uniform(0.01, 0.05, size=(n_b, n_span)), axis=1), 0, 0.99)
    X_obs = rng.uniform(0.1, 0.8, size=(n_t, n_b))
    beta = 0.8

    drop = 1
    X_nan = X_obs.copy()
    X_nan[drop, :] = np.nan
    X_filled, w = split_observed(X_nan)
    sse_row = reconstruction_sse(beta, X_filled, dt, X_pred, t_span, s, obs_weight_i=w)

    keep = np.ones(n_t, dtype=bool)
    keep[drop] = False
    sse_drop = reconstruction_sse(
        beta, X_obs[keep], dt[keep], X_pred, t_span, s
    )
    _assert_close(sse_row, sse_drop, "fully-missing row != dropped visit")
    print("PASS 4: fully-missing row == drop visit")


def test_sse_helpers_with_nans():
    rng = np.random.default_rng(2)
    n_patients, n_visits, n_b, n_sub = 4, 3, 3, 2
    n_span = 40
    t_span = np.linspace(0.0, 5.0, n_span)
    s = np.ones(n_b)
    K = 0.1 * np.eye(n_b) + 0.02 * rng.random((n_b, n_b))
    K = 0.5 * (K + K.T)
    cluster_f = [rng.uniform(0.01, 0.08, size=n_b) for _ in range(n_sub)]
    X_preds = [
        solve_system(np.zeros(n_b), cluster_f[z], K, t_span, 0.25, np.zeros(n_b))
        for z in range(n_sub)
    ]
    X_raw = rng.uniform(0.1, 0.7, size=(n_patients * n_visits, n_b))
    nan_idx = rng.choice(X_raw.size, size=max(1, int(0.15 * X_raw.size)), replace=False)
    X_raw.ravel()[nan_idx] = np.nan
    X_obs, w = split_observed(X_raw)
    dt = np.tile(np.array([0.0, 0.8, 1.6]), n_patients)
    ids = np.repeat(np.arange(n_patients), n_visits)
    beta = rng.uniform(0.5, 2.0, size=n_patients)
    assignments = rng.integers(0, n_sub, size=n_patients)

    sse = sse_matrix(X_obs, dt, ids, beta, X_preds, s, t_span, obs_weight=w)
    if sse.shape != (n_patients, n_sub) or not np.all(np.isfinite(sse)):
        raise AssertionError(f"sse_matrix bad: shape={sse.shape}, finite={np.isfinite(sse).all()}")

    sse_b = compute_sse_per_biomarker(
        X_obs, dt, ids, beta, assignments, cluster_f, s, 0.25,
        np.zeros(n_b), K, t_span, obs_weight=w,
    )
    if sse_b.shape != (n_b,) or not np.all(np.isfinite(sse_b)):
        raise AssertionError(f"compute_sse_per_biomarker bad: {sse_b}")
    print("PASS 5: sse_matrix / compute_sse_per_biomarker with NaNs")


def _make_patients(rng, n_patients=6, n_visits=3, n_b=4, nan_frac=0.15, fully_missing_patient=None):
    patients = []
    X_raw_blocks = []
    for i in range(n_patients):
        X = rng.uniform(0.05, 0.6, size=(n_visits, n_b))
        if fully_missing_patient is not None and i == fully_missing_patient:
            X[:] = np.nan
        else:
            n_nan = max(1, int(nan_frac * X.size))
            idx = rng.choice(X.size, size=n_nan, replace=False)
            X.ravel()[idx] = np.nan
            if not np.any(np.isfinite(X)):
                X[0, 0] = 0.2
        patients.append({
            "id": i,
            "X_obs": X,
            "dt": np.array([0.0, 0.7, 1.4][:n_visits], dtype=float),
            "cog": np.zeros((n_visits, 1)),
        })
        X_raw_blocks.append(X)
    return patients, np.vstack(X_raw_blocks)


def test_end_to_end_and_transform():
    rng = np.random.default_rng(3)
    n_b = 4
    patients, X_raw = _make_patients(rng, n_b=n_b)
    n_true_obs = int(np.isfinite(X_raw).sum())
    K = 0.15 * np.eye(n_b) + 0.03 * rng.random((n_b, n_b))
    K = 0.5 * (K + K.T)

    model = SubtypingEM(
        max_iter=2,
        t_max=5.0,
        step=0.1,
        K=K,
        n_subtypes=2,
        rng=np.random.default_rng(4),
        verbose=0,
        lambda_f=0.01,
        lambda_cog=0.0,
    )
    model.fit(patients)
    flat = model._prepare_data(patients)
    if flat["n_obs"] != n_true_obs:
        raise AssertionError(f"n_obs={flat['n_obs']} != true observed count {n_true_obs}")
    if not np.isfinite(model.bic_):
        raise AssertionError(f"bic_ not finite: {model.bic_}")

    held_out, _ = _make_patients(rng, n_patients=3, n_b=n_b)
    out = model.transform(held_out)
    if out.shape != (3,) or not np.all(np.isfinite(out["beta"])):
        raise AssertionError(f"transform failed: {out}")
    out_s = model.transform_soft(held_out)
    if out_s.shape != (3,) or not np.all(np.isfinite(out_s["beta"])):
        raise AssertionError(f"transform_soft failed: {out_s}")
    n_visits = sum(len(p["dt"]) for p in held_out)
    out_tp = model.transform_soft(held_out, per_timepoint=True)
    if out_tp.shape != (n_visits,):
        raise AssertionError(f"transform_soft per_timepoint shape {out_tp.shape} != {(n_visits,)}")
    for i, p in enumerate(held_out):
        mask = out_tp["patient_idx"] == i
        if not np.allclose(out_tp["beta"][mask], out_s["beta"][i]):
            raise AssertionError("per_timepoint beta is not the subject-level beta")
        t_obs_expected = p["dt"] + out_s["beta"][i]
        if not np.allclose(out_tp["t_obs"][mask], t_obs_expected):
            raise AssertionError("t_obs != beta + dt")
        proba = np.column_stack([out_tp[f"proba_{z + 1}"][mask] for z in range(model.n_subtypes)])
        if not np.allclose(proba.sum(axis=1), 1.0):
            raise AssertionError("per-timepoint proba rows do not sum to 1")
    print("PASS 6+7: end-to-end fit / transform / transform_soft")


def test_guardrails():
    rng = np.random.default_rng(5)
    n_b = 3
    K = 0.2 * np.eye(n_b)

    patients, _ = _make_patients(rng, n_patients=4, n_b=n_b, fully_missing_patient=0)
    # Ensure other patients keep every biomarker observed at least twice.
    for p in patients[1:]:
        p["X_obs"] = np.clip(p["X_obs"], 0.05, None)
        p["X_obs"][np.isnan(p["X_obs"])] = 0.2

    model = SubtypingEM(max_iter=1, t_max=5.0, step=0.2, K=K, n_subtypes=2, verbose=0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        model._prepare_data(patients)
    runtime = [w for w in caught if issubclass(w.category, RuntimeWarning)
               and "zero observed" in str(w.message)]
    if not runtime:
        raise AssertionError("expected RuntimeWarning for fully-missing patient")

    patients2, _ = _make_patients(rng, n_patients=3, n_b=n_b, nan_frac=0.0)
    for p in patients2:
        p["X_obs"][:, 0] = np.nan
        p["X_obs"][0, 0] = 0.3  # only 1 observed value for biomarker 0
    # 3 patients * 1 visit = 3... wait we set ALL rows of col 0 to nan then restore only [0,0]
    # of each patient, so 3 observed values. Need fewer than 2 total.
    for p in patients2:
        p["X_obs"][:, 0] = np.nan
    patients2[0]["X_obs"][0, 0] = 0.3  # exactly one observed value

    try:
        model._prepare_data(patients2)
    except ValueError as exc:
        if "fewer than 2 observed" not in str(exc):
            raise AssertionError(f"wrong ValueError: {exc}")
    else:
        raise AssertionError("expected ValueError for biomarker with <2 observations")
    print("PASS 8: guardrails (RuntimeWarning + ValueError)")


if __name__ == "__main__":
    test_split_observed()
    test_reconstruction_sse_masking()
    test_fully_missing_row()
    test_sse_helpers_with_nans()
    test_end_to_end_and_transform()
    test_guardrails()
    print("\nAll missing-data verification checks passed.")
