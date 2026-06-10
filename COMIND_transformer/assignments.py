"""
Cluster assignment (E-step) helpers for SubtypingEM.

Vectorized SSE over patients × subtypes via ``sse_matrix``; hard and jittered
assignment updaters share precomputed subtype trajectories.
"""

import numpy as np

from .optimizer_beta import reconstruction_sse
from .utils import solve_system


def precompute_trajectories(cluster_f, K, t_span, scalar_K, kappa, ode_method="LSODA"):
    """
    Solve the ODE once per subtype.

    Returns
    -------
    list[np.ndarray]
        Each element has shape (n_biomarkers, len(t_span)).
    """
    n_biomarkers = np.ravel(cluster_f[0]).shape[0]
    x0 = np.zeros(n_biomarkers)
    return [
        solve_system(
            x0, np.ravel(cluster_f[z]), K, t_span, scalar_K, kappa,
            ode_method=ode_method,
        )
        for z in range(len(cluster_f))
    ]


def sse_matrix(
    X_obs,
    dt,
    ids,
    beta,
    X_preds,
    s,
    t_span,
    cog=None,
    cluster_cog_a=None,
    cluster_cog_b=None,
    lambda_cog=0.0,
):
    """
    Per-patient, per-subtype reconstruction SSE (plus cognitive penalty if enabled).

    Parameters
    ----------
    X_obs : (n_rows, n_biomarkers)
    dt, ids : (n_rows,)
    beta : (n_patients,)
    X_preds : list of (n_biomarkers, len(t_span))
    s : (n_biomarkers,)

    Returns
    -------
    sse : (n_patients, n_subtypes)
    """
    unique_ids = np.unique(ids)
    n_patients = len(unique_ids)
    n_subtypes = len(X_preds)
    n_biomarkers = X_obs.shape[1]
    sse = np.zeros((n_patients, n_subtypes))

    use_cog = (
        lambda_cog > 0
        and cog is not None
        and cluster_cog_a is not None
        and cluster_cog_b is not None
    )

    for p_idx, pid in enumerate(unique_ids):
        mask = ids == pid
        X_i = X_obs[mask]
        dt_i = dt[mask]
        beta_i = beta[p_idx]
        t_pred = dt_i + beta_i

        for k in range(n_subtypes):
            X_pred_k = X_preds[k]
            X_interp = np.empty_like(X_i)
            for b in range(n_biomarkers):
                X_interp[:, b] = np.interp(t_pred, t_span, s[b] * X_pred_k[b])
            total = float(np.sum((X_i - X_interp) ** 2))
            if use_cog:
                cog_i = cog[mask]
                cog_pred = cog_i @ cluster_cog_a[k] + cluster_cog_b[k]
                total += lambda_cog * np.sum((t_pred - cog_pred) ** 2)
            sse[p_idx, k] = total

    return sse


def update_assignments_hard(
    X_obs,
    dt,
    ids,
    cog,
    beta,
    cluster_f,
    scalar_K,
    kappa,
    s,
    K,
    t_span,
    cluster_cog_a,
    cluster_cog_b,
    lambda_cog,
    ode_method="LSODA",
):
    """Hard E-step: assign each patient to the subtype with minimum SSE."""
    X_preds = precompute_trajectories(
        cluster_f, K, t_span, scalar_K, kappa, ode_method=ode_method
    )
    sse = sse_matrix(
        X_obs, dt, ids, beta, X_preds, s, t_span,
        cog=cog, cluster_cog_a=cluster_cog_a, cluster_cog_b=cluster_cog_b,
        lambda_cog=lambda_cog,
    )
    return np.argmin(sse, axis=1).astype(int)


def update_assignments_jitter(
    X_obs,
    dt,
    ids,
    cog,
    beta,
    cluster_f,
    scalar_K,
    kappa,
    s,
    K,
    t_span,
    cluster_cog_a,
    cluster_cog_b,
    lambda_cog,
    temperature=1.0,
    rng=None,
    ode_method="LSODA",
):
    """
    Sample assignments from softmax(-SSE / temperature).

    Uses reconstruction-only SSE (no cognitive term), matching the legacy jitter step.
    """
    if rng is None:
        rng = np.random.default_rng()

    unique_ids = np.unique(ids)
    n_patients = len(unique_ids)
    n_subtypes = len(cluster_f)
    assignments = np.zeros(n_patients, dtype=int)
    probabilities = np.zeros((n_patients, n_subtypes))

    X_preds = precompute_trajectories(
        cluster_f, K, t_span, scalar_K, kappa, ode_method=ode_method
    )

    for p_idx, pid in enumerate(unique_ids):
        mask = ids == pid
        X_obs_i = X_obs[mask]
        dt_i = dt[mask]
        beta_i = beta[p_idx]

        sse_vec = np.zeros(n_subtypes)
        for subtype in range(n_subtypes):
            sse_vec[subtype] = reconstruction_sse(
                beta_i, X_obs_i, dt_i, X_preds[subtype], t_span, s
            )

        log_p = -sse_vec / temperature
        log_p -= np.max(log_p)
        p = np.exp(log_p)
        p /= p.sum()

        probabilities[p_idx, :] = p
        assignments[p_idx] = rng.choice(n_subtypes, p=p)

    return assignments, probabilities
