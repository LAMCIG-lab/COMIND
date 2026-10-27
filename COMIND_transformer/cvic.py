"""
Held-out integral CVIC scores for SubtypingEM.

Predictive log density of a test subject, with time-shift marginalized by
Gauss–Hermite quadrature in log space and subtype mixed with the training
empirical prior. Residuals, KDE prior, and log-sum-exp are float64 throughout
(never form a product of densities). CVIC = -2 * sum of those log scores.

The quadrature, KDE prior, curvature floor, and missing-data mask follow
``implement-integral-cvic.md``. Ordered Nadeau–Bengio Tukey selection of k
follows ``cvic-subtype-selection.md`` (no Bonferroni).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from .utils import (
    beta_upper_from_followup,
    interp_uniform,
    make_beta_grid,
    solve_system,
    split_observed,
)

_LOG_2PI = np.log(2.0 * np.pi)
_GH_ORDER_DEFAULT = 15
_SIGMA_FLOOR = 1e-12
_VAR_FLOOR = 1e-8


def silverman_bandwidth(x, bandwidth=None):
    """Silverman's rule ``0.9 * min(sd, IQR/1.34) * n^(-1/5)``, or ``bandwidth``."""
    if bandwidth is not None:
        h = float(bandwidth)
        if not np.isfinite(h) or h <= 0:
            raise ValueError(f"kde bandwidth must be positive and finite, got {bandwidth!r}")
        return h
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n < 2:
        return 1.0
    sd = float(np.std(x, ddof=1))
    q75, q25 = np.percentile(x, [75.0, 25.0])
    iqr = float(q75 - q25)
    scale = min(sd, iqr / 1.34) if iqr > 0.0 else sd
    scale = max(scale, _VAR_FLOOR)
    return float(0.9 * scale * n ** (-0.2))


def log_kde_prior(beta, beta_train, bandwidth):
    """
    ``log p̂(β) = logsumexp_m log N(β | β_m, b²) - log M``.

    Parameters
    ----------
    beta : array-like
        Query locations, any shape.
    beta_train : (M,)
    bandwidth : float

    Returns
    -------
    ndarray
        Same shape as ``beta``.
    """
    beta = np.asarray(beta, dtype=np.float64)
    beta_train = np.asarray(beta_train, dtype=np.float64).ravel()
    b2 = float(bandwidth) ** 2
    log_norm = -0.5 * np.log(2.0 * np.pi * b2)
    u = (beta[..., None] - beta_train) / float(bandwidth)
    log_comp = log_norm - 0.5 * u ** 2
    return logsumexp(log_comp, axis=-1) - np.log(beta_train.size)


def log_kde_grad(beta, beta_train, bandwidth):
    """Gradient of ``log_kde_prior`` wrt ``beta`` (same shape as ``beta``)."""
    beta = np.asarray(beta, dtype=np.float64)
    beta_train = np.asarray(beta_train, dtype=np.float64).ravel()
    b2 = float(bandwidth) ** 2
    u = (beta[..., None] - beta_train) / float(bandwidth)
    log_comp = -0.5 * u ** 2
    w = np.exp(log_comp - logsumexp(log_comp, axis=-1, keepdims=True))
    return np.sum(w * (beta_train - beta[..., None]) / b2, axis=-1)


def _hermite_nodes(n_gh):
    u, w = np.polynomial.hermite.hermgauss(int(n_gh))
    u = np.asarray(u, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    return u, np.log(w) + u ** 2


def _dxdt_from_traj(x, f, K_eff):
    """ODE right-hand side on the trajectory grid: ``(1-x)(K_eff x + f)``."""
    f = np.asarray(f, dtype=np.float64).ravel()
    return (1.0 - x) * (K_eff @ x + f[:, None])


def subtype_trajs_and_dxdt(cluster_f, K, t_span, scalar_K, kappa, ode_method="LSODA"):
    """Per-subtype (trajectory, dx/dt) on ``t_span``, both (B, T)."""
    n_biomarkers = np.ravel(cluster_f[0]).shape[0]
    kappa_arr = np.zeros(n_biomarkers) if kappa is None else np.asarray(kappa, dtype=np.float64)
    K_eff = float(scalar_K) * np.asarray(K, dtype=np.float64) + np.diag(kappa_arr)
    trajs, dxdts = [], []
    x0 = np.zeros(n_biomarkers)
    for z in range(len(cluster_f)):
        f = np.ravel(cluster_f[z]).astype(np.float64)
        x = np.asarray(
            solve_system(x0, f, K, t_span, scalar_K, kappa, ode_method=ode_method),
            dtype=np.float64,
        )
        trajs.append(x)
        dxdts.append(_dxdt_from_traj(x, f, K_eff))
    return trajs, dxdts


def _yhat_at_times(traj, s, c, times, t_span, slopes=None):
    """Predictions ``c + s * x(t)`` at ``times`` (N,), shape (N, B)."""
    x = interp_uniform(traj, times, t_span, slopes=slopes)
    return (np.asarray(c, dtype=np.float64)[:, None]
            + np.asarray(s, dtype=np.float64)[:, None] * x).T


def g_at_betas(y, dt, mask, betas, traj, s, c, sigma, t_span, beta_train, bandwidth,
               slopes=None):
    """Log-integrand ``g_iz(β)`` at each β, shape (Q,)."""
    betas = np.asarray(betas, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    dt = np.asarray(dt, dtype=np.float64).ravel()
    j = dt.size
    q = betas.size
    times = (dt[None, :] + betas[:, None]).ravel()
    yhat = _yhat_at_times(traj, s, c, times, t_span, slopes=slopes).reshape(q, j, -1)
    resid = np.where(mask[None, :, :], (y[None, :, :] - yhat) / sigma, 0.0)
    n_obs = float(mask.sum())
    log_sigma_sum = float((mask * np.log(sigma)).sum())
    g = (
        -0.5 * np.square(resid).reshape(q, -1).sum(axis=1)
        - log_sigma_sum
        - 0.5 * n_obs * _LOG_2PI
        + log_kde_prior(betas, beta_train, bandwidth)
    )
    return g


def _g_and_grad_scalar(beta, y, dt, mask, traj, dxdt, s, c, sigma, t_span,
                       beta_train, bandwidth, traj_slopes, dx_slopes):
    """``g(β)`` and ``dg/dβ`` at a scalar β (for MAP)."""
    beta = float(np.asarray(beta, dtype=np.float64).ravel()[0])
    dt = np.asarray(dt, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    t = dt + beta
    x = interp_uniform(traj, t, t_span, slopes=traj_slopes)
    dx = interp_uniform(dxdt, t, t_span, slopes=dx_slopes)
    yhat_bj = c[:, None] + s[:, None] * x
    mask_bj = mask.T
    resid = (y.T - yhat_bj) * mask_bj
    n_obs = float(mask.sum())
    log_sigma_sum = float((mask * np.log(sigma)).sum())
    log_p = float(log_kde_prior(beta, beta_train, bandwidth))
    g = (
        -0.5 * np.sum((resid / sigma[:, None]) ** 2)
        - log_sigma_sum
        - 0.5 * n_obs * _LOG_2PI
        + log_p
    )
    dy_dbeta = s[:, None] * dx
    dg_like = float(np.sum(resid / (sigma[:, None] ** 2) * dy_dbeta))
    dg_prior = float(log_kde_grad(beta, beta_train, bandwidth))
    return g, dg_like + dg_prior


def gauss_newton_curvature(y, dt, mask, beta, traj, dxdt, s, c, sigma, t_span,
                           traj_slopes, dx_slopes, beta_var):
    """Gauss–Newton curvature of -g at ``beta``, floored at ``1 / Var(β_train)``."""
    dt = np.asarray(dt, dtype=np.float64).ravel()
    mask = np.asarray(mask, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    t = dt + float(beta)
    dx = interp_uniform(dxdt, t, t_span, slopes=dx_slopes)
    dy_dbeta = s[:, None] * dx
    h_gn = float(np.sum(mask.T * (dy_dbeta / sigma[:, None]) ** 2))
    h_floor = 1.0 / max(float(beta_var), _VAR_FLOOR)
    return max(h_gn, h_floor)


def gauss_hermite_log_integral(g_nodes, tau, log_w):
    """``log ∫ e^g dβ ≈ log(√2 τ) + logsumexp(g_q + log w_q + u_q²)``."""
    tau = float(tau)
    if not np.isfinite(tau) or tau <= 0:
        return -np.inf
    return float(np.log(np.sqrt(2.0) * tau) + logsumexp(g_nodes + log_w))


def map_beta(y, dt, mask, traj, dxdt, s, c, sigma, t_span, beta_train, bandwidth,
             t_max, n_beta_grid, cap=None, traj_slopes=None, dx_slopes=None):
    """Grid-then-L-BFGS-B MAP of ``g_iz(β)``. Returns ``(beta_star, g_star)``."""
    grid = make_beta_grid(t_max, n_beta_grid).astype(np.float64)
    if cap is not None:
        ok = grid <= float(cap) + 1e-15
        if not np.any(ok):
            grid = np.array([max(0.0, float(cap))], dtype=np.float64)
        else:
            grid = grid[ok]
    g_grid = g_at_betas(
        y, dt, mask, grid, traj, s, c, sigma, t_span, beta_train, bandwidth,
        slopes=traj_slopes,
    )
    beta0 = float(grid[int(np.argmax(g_grid))])
    hi = float(t_max) if cap is None else float(cap)

    def objective(v):
        g, dg = _g_and_grad_scalar(
            v[0], y, dt, mask, traj, dxdt, s, c, sigma, t_span,
            beta_train, bandwidth, traj_slopes, dx_slopes,
        )
        return -g, np.array([-dg], dtype=np.float64)

    result = minimize(
        objective,
        x0=np.array([beta0], dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        bounds=[(0.0, hi)],
        options={"maxiter": 50, "ftol": 1e-12, "gtol": 1e-8},
    )
    beta_star = float(result.x[0])
    g_star, _ = _g_and_grad_scalar(
        beta_star, y, dt, mask, traj, dxdt, s, c, sigma, t_span,
        beta_train, bandwidth, traj_slopes, dx_slopes,
    )
    return beta_star, g_star


def log_integrate_beta(
    y, dt, mask, traj, dxdt, s, c, sigma, t_span, beta_train, bandwidth,
    beta_var, t_max, n_beta_grid, n_gh=_GH_ORDER_DEFAULT, cap=None,
    traj_slopes=None, dx_slopes=None, u=None, log_w=None,
):
    """``log ∫ e^{g_iz(β)} dβ`` by adaptive Gauss–Hermite around the MAP."""
    if u is None or log_w is None:
        u, log_w = _hermite_nodes(n_gh)
    beta_star, _ = map_beta(
        y, dt, mask, traj, dxdt, s, c, sigma, t_span, beta_train, bandwidth,
        t_max, n_beta_grid, cap=cap, traj_slopes=traj_slopes, dx_slopes=dx_slopes,
    )
    h = gauss_newton_curvature(
        y, dt, mask, beta_star, traj, dxdt, s, c, sigma, t_span,
        traj_slopes, dx_slopes, beta_var,
    )
    tau = 1.0 / np.sqrt(h)
    betas = beta_star + np.sqrt(2.0) * tau * u
    g_nodes = g_at_betas(
        y, dt, mask, betas, traj, s, c, sigma, t_span, beta_train, bandwidth,
        slopes=traj_slopes,
    )
    return gauss_hermite_log_integral(g_nodes, tau, log_w), beta_star, h


def unpack_patient(patient):
    """Finite observations, mask, and dt for one patient dict. Does not mutate ``patient``."""
    y_raw = np.asarray(patient["X_obs"], dtype=np.float64)
    y, mask = split_observed(y_raw)
    dt = np.asarray(patient["dt"], dtype=np.float64).ravel()
    if y.ndim == 1:
        y = y.reshape(1, -1)
        mask = mask.reshape(1, -1)
    if dt.shape[0] != y.shape[0]:
        raise ValueError(
            f"dt length {dt.shape[0]} != n_visits {y.shape[0]} for patient "
            f"{patient.get('id', '?')}"
        )
    return y, mask, dt


def subject_logliks_cvic(
    patients,
    cluster_f,
    s,
    c,
    scalar_K,
    kappa,
    K,
    t_span,
    sigma,
    log_pi,
    beta_train,
    kde_bandwidth,
    beta_var,
    t_max,
    n_beta_grid,
    ode_method="LSODA",
    n_gh=_GH_ORDER_DEFAULT,
    cap_beta_by_followup=False,
    verbose=0,
    trajs=None,
    dxdts=None,
):
    """
    Predictive log density of each held-out patient, subtype and β marginalized.

    Parameters
    ----------
    patients : list of dict
        Each has ``X_obs`` (J, B) and ``dt`` (J,). NaNs are missing.
    sigma : (B,)
        Training residual standard deviations.
    log_pi : (Z,)
        Log empirical subtype prior (``-inf`` for empty subtypes).
    beta_train : (N_train,)
        Fitted training time-shifts (KDE atoms).

    Returns
    -------
    ndarray, shape (n_patients,)
        ``ℓ_i = logsumexp_z (log π_z + log ∫ e^{g_iz} dβ)``. A patient with
        no observed entries scores 0.
    """
    n = len(patients)
    log_pi = np.asarray(log_pi, dtype=np.float64).ravel()
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64).ravel(), np.sqrt(_SIGMA_FLOOR))
    s = np.asarray(s, dtype=np.float64).ravel()
    c = np.zeros_like(s) if c is None else np.asarray(c, dtype=np.float64).ravel()
    beta_train = np.asarray(beta_train, dtype=np.float64).ravel()
    n_subtypes = len(cluster_f)
    if log_pi.shape != (n_subtypes,):
        raise ValueError(f"log_pi shape {log_pi.shape} != ({n_subtypes},)")

    if trajs is None or dxdts is None:
        trajs, dxdts = subtype_trajs_and_dxdt(
            cluster_f, K, t_span, scalar_K, kappa, ode_method=ode_method
        )
    traj_slopes = [np.diff(tr, axis=1) / np.diff(t_span) for tr in trajs]
    dx_slopes = [np.diff(dx, axis=1) / np.diff(t_span) for dx in dxdts]
    u, log_w = _hermite_nodes(n_gh)
    caps = None
    if cap_beta_by_followup:
        caps = beta_upper_from_followup([p["dt"] for p in patients], t_max)

    out = np.zeros(n, dtype=np.float64)
    iterable = patients
    if verbose:
        try:
            from tqdm import tqdm
            iterable = tqdm(patients, desc="CVIC subject log-likelihood")
        except ImportError:
            iterable = patients

    for i, p in enumerate(iterable):
        y, mask, dt = unpack_patient(p)
        if mask.sum() == 0:
            out[i] = 0.0
            continue
        L = np.full(n_subtypes, -np.inf, dtype=np.float64)
        cap_i = None if caps is None else float(caps[i])
        for z in range(n_subtypes):
            if not np.isfinite(log_pi[z]):
                continue
            L[z], _, _ = log_integrate_beta(
                y, dt, mask, trajs[z], dxdts[z], s, c, sigma, t_span,
                beta_train, kde_bandwidth, beta_var, t_max, n_beta_grid,
                n_gh=n_gh, cap=cap_i,
                traj_slopes=traj_slopes[z], dx_slopes=dx_slopes[z],
                u=u, log_w=log_w,
            )
        out[i] = float(logsumexp(log_pi + L))
    return out


def two_way_tau(X):
    """
    Residual scale of an n × m log-score matrix after removing subject and
    model means. ``tau = sqrt(sum(R^2) / ((n-1)(m-1)))``.
    """
    X = np.asarray(X, dtype=np.float64)
    n, m = X.shape
    if n < 2 or m < 2:
        raise ValueError(f"two-way tau needs n>=2 and m>=2, got {(n, m)}")
    row = X.mean(axis=1, keepdims=True)
    col = X.mean(axis=0, keepdims=True)
    R = X - row - col + X.mean()
    df = (n - 1) * (m - 1)
    tau = float(np.sqrt(np.sum(R ** 2) / df))
    return tau, int(df)


def nadeau_bengio_se(tau, n, n_folds):
    """SE of a CVIC level: ``2 * sqrt(n) * tau * sqrt(1 + F/(F-1))``."""
    if n_folds < 2:
        raise ValueError(f"Nadeau–Bengio factor needs n_folds>=2, got {n_folds}")
    factor = np.sqrt(1.0 + n_folds / (n_folds - 1.0))
    return float(2.0 * np.sqrt(n) * tau * factor)


def ordered_tukey_nb(
    log_scores,
    n_folds,
    alpha=0.05,
    n_draw=200_000,
    seed=0,
    bonferroni=True,
):
    """
    Ordered one-sided Tukey tests for the smallest credible k.

    ``log_scores`` is n × K_max held-out log densities (column 0 is k=1).
    CVIC(k) = -2 * sum of column k-1. For each k = 1, ..., K_max-1 test
    equal expected score among {k, ..., K_max} against some m>k being
    better (lower CVIC). Stop at the first non-rejection.

    ``bonferroni=True`` (default) rejects at ``alpha / (K_max - 1)``, the
    same divisor DPMoSt ``_cvic_tukey_k`` uses. The studentized-range group
    size inside each null is still ``K_max - k + 1``.

    Returns
    -------
    selected_k : int
        1-based n_subtypes.
    info : dict
        ``cvic``, ``pvalues``, ``stats``, ``gaps``, ``se``, ``df``, ``rejected``,
        ``alpha`` (the threshold actually used).
    """
    X_all = np.asarray(log_scores, dtype=np.float64)
    if X_all.ndim != 2:
        raise ValueError(f"log_scores must be 2-D, got shape {X_all.shape}")
    n, k_max = X_all.shape
    if k_max < 1:
        raise ValueError("log_scores must have at least one column")
    if n < 2:
        raise ValueError(f"need at least 2 subjects for Tukey, got n={n}")
    cvic = -2.0 * X_all.sum(axis=0)
    pvalues = np.full(max(k_max - 1, 0), np.nan)
    stats = np.full(max(k_max - 1, 0), np.nan)
    gaps = np.full(max(k_max - 1, 0), np.nan)
    ses = np.full(max(k_max - 1, 0), np.nan)
    dfs = np.full(max(k_max - 1, 0), np.nan)
    rejected = np.zeros(max(k_max - 1, 0), dtype=bool)

    threshold = float(alpha)
    if bonferroni and k_max > 1:
        threshold = float(alpha) / (k_max - 1)
    if k_max == 1:
        return 1, dict(
            cvic=cvic, pvalues=pvalues, stats=stats, gaps=gaps, se=ses, df=dfs,
            rejected=rejected, selected_k=1, alpha=threshold, bonferroni=bool(bonferroni),
        )

    rng = np.random.default_rng(seed)
    selected = k_max
    for k in range(1, k_max):
        idx = k - 1
        X = X_all[:, idx:]
        m = X.shape[1]
        tau, df = two_way_tau(X)
        se = nadeau_bengio_se(tau, n, n_folds)
        gap = float(cvic[idx] - np.min(cvic[idx + 1:]))
        stat = gap / se if se > 0 else np.inf
        r = k_max - k
        Z = rng.standard_normal((int(n_draw), r + 1))
        V = rng.chisquare(df, size=int(n_draw)) / df
        null = (Z[:, 0] - Z[:, 1:].min(axis=1)) / np.sqrt(V)
        p = (float(np.count_nonzero(null >= stat) + 1) / (int(n_draw) + 1))
        pvalues[idx] = p
        stats[idx] = stat
        gaps[idx] = gap
        ses[idx] = se
        dfs[idx] = df
        reject = bool(p < threshold)
        rejected[idx] = reject
        if not reject:
            selected = k
            break
    info = dict(
        cvic=cvic,
        pvalues=pvalues,
        stats=stats,
        gaps=gaps,
        se=ses,
        df=dfs,
        rejected=rejected,
        selected_k=int(selected),
        alpha=threshold,
        bonferroni=bool(bonferroni),
    )
    return int(selected), info
