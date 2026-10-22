"""
Cluster assignment (E-step) helpers for SubtypingEM.

Vectorized SSE over patients × subtypes via ``sse_matrix``; hard and jittered
assignment updaters share precomputed subtype trajectories.
"""

import numpy as np

from .optimizer_beta import reconstruction_sse
from .utils import interp_uniform, solve_system


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
    obs_weight=None,
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
    obs_weight : (n_rows, n_biomarkers) or None
        Elementwise 1.0 = observed, 0.0 = missing. ``None`` treats every
        entry as observed.

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
            residuals = X_i - X_interp
            w_i = obs_weight[mask] if obs_weight is not None else None
            if w_i is not None:
                residuals = residuals * w_i
            total = float(np.sum(residuals ** 2))
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
    obs_weight=None,
    X_preds=None,
):
    """Hard E-step: assign each patient to the subtype with minimum SSE.

    X_preds : list[np.ndarray], optional
        Precomputed per-subtype trajectories (e.g. returned by
        fit_theta_globals at its converged parameters) to avoid re-solving
        the ODE. Only valid if cluster_f/scalar_K/kappa here match what
        produced it exactly — if the caller's step order ever changes so that
        cluster_f is updated between fit_theta_globals and this call, pass
        None instead to force a fresh solve.
    """
    if X_preds is None:
        X_preds = precompute_trajectories(
            cluster_f, K, t_span, scalar_K, kappa, ode_method=ode_method
        )
    sse = sse_matrix(
        X_obs, dt, ids, beta, X_preds, s, t_span,
        cog=cog, cluster_cog_a=cluster_cog_a, cluster_cog_b=cluster_cog_b,
        lambda_cog=lambda_cog,
        obs_weight=obs_weight,
    )
    return np.argmin(sse, axis=1).astype(int)


def joint_grid_losses(
    X_obs,
    dt,
    ids,
    candidates,
    X_preds,
    s,
    t_span,
    cog=None,
    cluster_cog_a=None,
    cluster_cog_b=None,
    lambda_cog=0.0,
    obs_weight=None,
):
    """
    ``sse_matrix`` evaluated at every row of a candidate-β table.

    Parameters
    ----------
    X_obs : (n_rows, n_biomarkers)
        Observations with missing entries filled (see ``utils.split_observed``).
    dt, ids : (n_rows,)
    candidates : (n_candidates, n_patients)
        Candidate β per patient; row c is one β vector.
    X_preds : list of (n_biomarkers, len(t_span))
    s : (n_biomarkers,)
    t_span : (T,)
    cog, cluster_cog_a, cluster_cog_b, lambda_cog
        Cognitive term, included under the same condition as ``sse_matrix``.
    obs_weight : (n_rows, n_biomarkers) or None

    Returns
    -------
    np.ndarray
        Shape (n_candidates, n_patients, n_subtypes). Entry ``[c, i, z]`` is
        ``sse_matrix`` at ``beta = candidates[c]``, patient i, subtype z (up to
        summation order).
    """
    candidates = np.atleast_2d(np.asarray(candidates, dtype=float))
    unique_ids, row_pidx = np.unique(ids, return_inverse=True)
    n_patients = len(unique_ids)
    n_subtypes = len(X_preds)
    use_cog = (
        lambda_cog > 0
        and cog is not None
        and cluster_cog_a is not None
        and cluster_cog_b is not None
    )
    s = np.asarray(s, dtype=float)
    scaled = [s[:, None] * np.asarray(X_preds[z]) for z in range(n_subtypes)]
    dts = np.diff(t_span)
    slopes = [np.diff(y, axis=1) / dts for y in scaled]
    cog_pred = (
        [cog @ cluster_cog_a[z] + cluster_cog_b[z] for z in range(n_subtypes)]
        if use_cog else None
    )

    out = np.zeros((candidates.shape[0], n_patients, n_subtypes))
    for c, beta_c in enumerate(candidates):
        t = dt + beta_c[row_pidx]
        for z in range(n_subtypes):
            pred = interp_uniform(scaled[z], t, t_span, slopes[z]).T
            r = X_obs - pred
            if obs_weight is not None:
                r = r * obs_weight
            per_row = np.sum(r ** 2, axis=1)
            if use_cog:
                per_row = per_row + lambda_cog * (t - cog_pred[z]) ** 2
            out[c, :, z] = np.bincount(row_pidx, weights=per_row, minlength=n_patients)
    return out


def select_joint_grid(losses, candidates, beta_upper=None):
    """
    Joint (candidate β, subtype) argmin per patient.

    Parameters
    ----------
    losses : (n_candidates, n_patients, n_subtypes)
    candidates : (n_candidates, n_patients)
    beta_upper : (n_patients,) or None
        Candidates above a patient's bound get loss ``+inf``.

    Returns
    -------
    assignments : (n_patients,) int
    beta_start : (n_patients,) float

    Notes
    -----
    The argmin is over the candidate-major flattening, so on ties the lowest
    candidate index wins and, within one candidate, the lowest subtype index
    wins (as ``np.argmin`` does in the hard E-step).
    """
    losses = np.array(losses, dtype=float, copy=True)
    n_cand, n_patients, n_subtypes = losses.shape
    if beta_upper is not None:
        losses[candidates > np.asarray(beta_upper)[None, :]] = np.inf
    flat = losses.transpose(1, 0, 2).reshape(n_patients, n_cand * n_subtypes)
    idx = np.argmin(flat, axis=1)
    best_c = idx // n_subtypes
    best_z = idx % n_subtypes
    return best_z.astype(int), candidates[best_c, np.arange(n_patients)].astype(float)


def update_assignments_joint_grid(
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
    beta_grid,
    beta_upper=None,
    ode_method="LSODA",
    obs_weight=None,
    X_preds=None,
):
    """
    Joint-grid E-step: choose each patient's (β, subtype) pair jointly.

    For patient i the candidates are their current β (index 0) followed by
    ``beta_grid``. Candidates above ``beta_upper[i]`` are skipped. Every
    (candidate, subtype) pair is scored with the ``sse_matrix`` patient loss
    (reconstruction SSE plus the ``lambda_cog`` term), and the joint argmin is
    taken with ties going to the current β, then to the lower subtype index.

    Guarantee: each patient's chosen loss is <= their loss at the current
    (β, z), because that pair is always a candidate. The current β is never
    masked: the effective bound is ``max(beta_upper[i], beta[i])``.

    Not guaranteed: the full EM objective can still rise, because this search
    ignores the JSD term and the β prior. The β step that follows optimizes
    the full objective from ``beta_start``.

    Parameters
    ----------
    X_obs, dt, ids, cog, beta, cluster_f, scalar_K, kappa, s, K, t_span,
    cluster_cog_a, cluster_cog_b, lambda_cog, ode_method, obs_weight, X_preds
        As in ``update_assignments_hard``.
    beta_grid : (G,)
        Grid candidates (see ``utils.make_beta_grid``).
    beta_upper : (n_patients,) or None
        Per-patient β upper bound. ``None`` means no bound.

    Returns
    -------
    assignments : (n_patients,) int
    beta_start : (n_patients,) float
        The chosen candidate β per patient; the β step warm-starts from it.
    """
    if X_preds is None:
        X_preds = precompute_trajectories(
            cluster_f, K, t_span, scalar_K, kappa, ode_method=ode_method
        )
    beta = np.asarray(beta, dtype=float)
    beta_grid = np.asarray(beta_grid, dtype=float).ravel()
    candidates = np.vstack(
        [beta[None, :], np.repeat(beta_grid[:, None], len(beta), axis=1)]
    )
    losses = joint_grid_losses(
        X_obs, dt, ids, candidates, X_preds, s, t_span,
        cog=cog, cluster_cog_a=cluster_cog_a, cluster_cog_b=cluster_cog_b,
        lambda_cog=lambda_cog, obs_weight=obs_weight,
    )
    upper = None if beta_upper is None else np.maximum(np.asarray(beta_upper, float), beta)
    return select_joint_grid(losses, candidates, upper)


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
    obs_weight=None,
    X_preds=None,
):
    """
    Sample assignments from softmax(-SSE / temperature).

    Uses reconstruction-only SSE (no cognitive term), matching the legacy jitter step.

    X_preds : list[np.ndarray], optional
        Precomputed per-subtype trajectories (e.g. returned by
        fit_theta_globals at its converged parameters) to avoid re-solving
        the ODE. Only valid if cluster_f/scalar_K/kappa here match what
        produced it exactly — if the caller's step order ever changes so that
        cluster_f is updated between fit_theta_globals and this call, pass
        None instead to force a fresh solve.
    """
    if rng is None:
        rng = np.random.default_rng()

    unique_ids = np.unique(ids)
    n_patients = len(unique_ids)
    n_subtypes = len(cluster_f)
    assignments = np.zeros(n_patients, dtype=int)
    probabilities = np.zeros((n_patients, n_subtypes))

    if X_preds is None:
        X_preds = precompute_trajectories(
            cluster_f, K, t_span, scalar_K, kappa, ode_method=ode_method
        )

    for p_idx, pid in enumerate(unique_ids):
        mask = ids == pid
        X_obs_i = X_obs[mask]
        dt_i = dt[mask]
        beta_i = beta[p_idx]

        sse_vec = np.zeros(n_subtypes)
        w_i = obs_weight[mask] if obs_weight is not None else None
        for subtype in range(n_subtypes):
            sse_vec[subtype] = reconstruction_sse(
                beta_i, X_obs_i, dt_i, X_preds[subtype], t_span, s,
                obs_weight_i=w_i,
            )

        log_p = -sse_vec / temperature
        log_p -= np.max(log_p)
        p = np.exp(log_p)
        p /= p.sum()

        probabilities[p_idx, :] = p
        assignments[p_idx] = rng.choice(n_subtypes, p=p)

    return assignments, probabilities
