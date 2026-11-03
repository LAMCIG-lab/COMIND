"""
f calibration and KKT checks with the JAX core (exact gradients, network included).

Model (per subtype z, per biomarker b):
    dx/dt = (1 - x) * (K_eff x + f_z),   x(0) = 0,   K_eff = s_t K + diag(kappa)
    y_ijb = s_b * x_{z_i, b}(beta_i + dt_ij) + eps_ijb,   eps ~ N(0, sigma_b^2)

pack_patients   Stack a patient list into jax_core.PackedData.
f_pull          -dSSE/dF at any F: how hard the data pull each f_zb upward.
lambda_f_max    Smallest lambda_f that zeros every f (lasso lambda_max).
kkt_report      Check the L1 + nonnegativity optimality conditions at a fitted F.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def pack_patients(
    patients: Sequence[dict],
    pad_rows: Optional[int] = None,
    pad_patients: Optional[int] = None,
):
    """
    Stack patient dicts (``X_obs``, ``dt``) into ``(PackedData, meta)``.

    Pass the estimator's ``jax_pad_rows`` / ``jax_pad_patients`` so ``f_pull``
    reuses the compiled ``_cluster_vg`` from the fit instead of compiling again.
    """
    from COMIND_transformer import jax_core as jc

    X = np.vstack([np.asarray(p["X_obs"], dtype=float) for p in patients])
    dt = np.concatenate([np.asarray(p["dt"], dtype=float).ravel() for p in patients])
    ids = np.concatenate([np.full(len(p["dt"]), i) for i, p in enumerate(patients)])
    return jc.pack_data(X, None, dt, ids, np.zeros((len(dt), 1)), len(patients),
                        pad_rows=pad_rows, pad_patients=pad_patients)


def f_pull(
    packed,
    F: np.ndarray,
    s: np.ndarray,
    kappa: np.ndarray,
    scalar_K: float,
    beta: np.ndarray,
    assignments: np.ndarray,
    K: np.ndarray,
    t_span: np.ndarray,
) -> np.ndarray:
    """
    -dSSE/dF at ``F``, shape (Z, B).

    For the f-step loss SSE(F) + lambda_f * sum(F) with F >= 0, the
    optimality (KKT) conditions are, per entry:
        f_zb > 0  ->  pull_zb == lambda_f
        f_zb = 0  ->  pull_zb <= lambda_f
    """
    from COMIND_transformer import jax_core as jc

    data, meta = packed
    P = meta["P"]
    _, h, n_steps = jc.grid_from_t_span(t_span)
    _, g = jc._cluster_vg(
        jc.f64(np.asarray(F, dtype=float)),
        jc.f64(np.ravel(s)),
        jc.f64(np.ravel(kappa)),
        jc.f64(float(scalar_K)),
        jc.pad_beta(beta, P),
        jc.pad_assignments(assignments, P),
        data,
        jc.f64(K),
        jc.make_hyp(lambda_f=0.0),
        jc.f64(h),
        n_steps=n_steps,
    )
    return -np.asarray(g, dtype=np.float64)


def lambda_f_max(packed, s, kappa, scalar_K, beta, assignments, K, t_span, n_subtypes):
    """
    Smallest lambda_f for which F = 0 is optimal (the lasso lambda_max).

    At F = 0 every trajectory is identically 0, so the pull does not depend on
    the current f. Given s, kappa, scalar_K, beta and z, it is the exact upper
    end of the useful lambda_f range.

    Caution: at x = 0 the sensitivity dx/df_b is integral_0^t exp(K_eff u) du e_b,
    which grows exponentially when the network is strong. lambda_max is then
    inflated by hub nodes, and near lambda_max the L1 puts forcing on hubs
    (one unit of f at a hub buys atrophy everywhere). Search a log grid that
    reaches far below it, e.g. 1e-7 ... 1e-1 times lambda_max.

    Returns
    -------
    lam_max : float
    pull0 : (Z, B) per-entry threshold; f_zb leaves 0 once lambda_f < pull0[z, b].
    """
    B = np.asarray(K).shape[0]
    pull0 = f_pull(packed, np.zeros((n_subtypes, B)), s, kappa, scalar_K,
                   beta, assignments, K, t_span)
    return float(pull0.max()), pull0


def kkt_report(F_fit: np.ndarray, lambda_f: float, pull: np.ndarray, rel_tol: float = 0.05) -> dict:
    """
    Did the f-step reach its optimum? ``pull`` = ``f_pull`` evaluated at ``F_fit``.

    If the active entries have pull close to lambda_f, the optimizer did its
    job and a dense F is what the objective wants. Large violations mean the
    optimizer stopped early.
    """
    F_fit = np.asarray(F_fit, dtype=float)
    active = F_fit > 0
    viol_active = np.abs(pull[active] - lambda_f) / max(lambda_f, 1e-12)
    viol_zero = np.maximum(pull[~active] - lambda_f, 0.0) / max(lambda_f, 1e-12)
    return dict(
        n_active=int(active.sum()),
        n_zero=int((~active).sum()),
        active_median_rel_violation=float(np.median(viol_active)) if active.any() else 0.0,
        active_frac_ok=float(np.mean(viol_active < rel_tol)) if active.any() else 1.0,
        zero_max_rel_violation=float(viol_zero.max()) if (~active).any() else 0.0,
    )
