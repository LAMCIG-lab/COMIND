import numpy as np
from scipy.optimize import minimize
from scipy.integrate import cumulative_simpson
from scipy.interpolate import CubicSpline
from .utils import solve_system
from .sensitivity_lsoda import (
    integrate_f_sensitivities_lsoda,
    interp_sensitivity_at_obs,
)

SPARSE_PSEUDO_HUBER_DELTA = 0.01


def _pseudo_huber_penalty(values: np.ndarray, delta: float = SPARSE_PSEUDO_HUBER_DELTA) -> float:
    """Elementwise pseudo-Huber penalty summed over values."""
    values = np.asarray(values, dtype=float)
    scaled = values / delta
    return float(np.sum((delta ** 2) * (np.sqrt(1.0 + scaled ** 2) - 1.0)))


def _pseudo_huber_grad(values: np.ndarray, delta: float = SPARSE_PSEUDO_HUBER_DELTA) -> np.ndarray:
    """Gradient of summed pseudo-Huber penalty w.r.t. values."""
    values = np.asarray(values, dtype=float)
    return values / np.sqrt(1.0 + (values / delta) ** 2)


def theta_cluster_loss(
    params: np.ndarray,
    t_obs: np.ndarray,
    x_obs: np.ndarray,
    K: np.ndarray,
    t_span: np.ndarray,
    s: np.ndarray,
    scalar_K: float,
    lambda_f: float,
    kappa: np.ndarray = None,
) -> float:
    """Loss for cluster-level f with fixed global s and scalar_K."""
    n_biomarkers = x_obs.shape[1]
    f = params
    x0 = np.zeros(n_biomarkers)

    x = solve_system(x0, f, K, t_span, scalar_K, kappa)
    x_scaled = s[:, None] * x

    t_obs_clamped = np.clip(t_obs, t_span[0], t_span[-1])
    x_pred = np.zeros_like(x_obs)
    for j in range(n_biomarkers):
        x_pred[:, j] = np.interp(t_obs_clamped, t_span, x_scaled[j])

    residuals = x_obs - x_pred
    return np.sum(residuals ** 2) + lambda_f * np.sum(f)


def theta_cluster_loss_jac(
    params: np.ndarray,
    t_obs: np.ndarray,
    x_obs: np.ndarray,
    K: np.ndarray,
    t_span: np.ndarray,
    s: np.ndarray,
    scalar_K: float,
    lambda_f: float,
    kappa: np.ndarray = None,
) -> tuple:
    """Loss and approximate gradient for cluster-level f (cumulative_simpson)."""
    n_biomarkers = x_obs.shape[1]
    f = params
    x0 = np.zeros(n_biomarkers)

    x = solve_system(x0, f, K, t_span, scalar_K, kappa)
    x_scaled = s[:, None] * x

    t_obs_clamped = np.clip(t_obs, t_span[0], t_span[-1])

    x_pred = np.zeros_like(x_obs)
    for j in range(n_biomarkers):
        x_pred[:, j] = np.interp(t_obs_clamped, t_span, x_scaled[j])

    residuals = x_obs - x_pred
    loss = np.sum(residuals ** 2) + lambda_f * np.sum(f)

    cum_int = np.array([
        cumulative_simpson(1 - x[i], x=t_span, initial=0)
        for i in range(n_biomarkers)
    ])

    df_obs = np.zeros_like(x_obs)
    for i in range(n_biomarkers):
        cs_integ = CubicSpline(t_span, cum_int[i], extrapolate=False)
        df_obs[:, i] = cs_integ(t_obs_clamped)

    grad_f = (
        -2.0 * np.sum(residuals * (df_obs * s[None, :]), axis=0)
        + lambda_f * np.ones(n_biomarkers)
    )

    return loss, grad_f


def theta_cluster_loss_jac_exact(
    params: np.ndarray,
    t_obs: np.ndarray,
    x_obs: np.ndarray,
    K: np.ndarray,
    t_span: np.ndarray,
    s: np.ndarray,
    scalar_K: float,
    lambda_f: float,
    kappa: np.ndarray = None,
) -> tuple:
    """Loss and gradient w.r.t. f using exact sensitivities solved with LSODA."""
    n_biomarkers = x_obs.shape[1]
    f = params
    x0 = np.zeros(n_biomarkers)
    if kappa is None:
        kappa = np.zeros(n_biomarkers)
    K_eff = scalar_K * K + np.diag(kappa)

    x = solve_system(x0, f, K, t_span, scalar_K, kappa)
    x_scaled = s[:, None] * x
    t_obs_clamped = np.clip(t_obs, t_span[0], t_span[-1])
    x_pred = np.zeros_like(x_obs)
    for j in range(n_biomarkers):
        x_pred[:, j] = np.interp(t_obs_clamped, t_span, x_scaled[j])

    residuals = x_obs - x_pred
    loss = np.sum(residuals ** 2) + lambda_f * np.sum(f)

    U = integrate_f_sensitivities_lsoda(t_span, x, K_eff, K, f)
    grad_f = lambda_f * np.ones(n_biomarkers)
    for b in range(n_biomarkers):
        w_at = interp_sensitivity_at_obs(U[:, b, :], t_span, t_obs_clamped)
        grad_f[b] -= 2.0 * np.sum(residuals * s[None, :] * w_at)

    return loss, grad_f


def fit_theta_cluster(
    X_obs: np.ndarray,
    dt_obs: np.ndarray,
    ids: np.ndarray,
    K: np.ndarray,
    t_span: np.ndarray,
    *,
    s: np.ndarray,
    scalar_K: float,
    lambda_f: float,
    beta_pred: np.ndarray = None,
    f_guess: np.ndarray = None,
    rng: np.random.Generator = None,
    kappa: np.ndarray = None,
    method: str = "lbfgs_approx",
) -> np.ndarray:
    """
    Optimizes cluster-level f for patients in a specific cluster.

    method:
        'lbfgs_approx' — L-BFGS-B with cumulative_simpson approximate Jacobian.
        'lbfgs_exact'  — L-BFGS-B with exact LSODA sensitivities.
        'nelder_mead'  — derivative-free, loss only.
    """
    if rng is None:
        rng = np.random.default_rng(75)

    t_pred = dt_obs + beta_pred[ids]
    n_biomarkers = X_obs.shape[1]

    if f_guess is None:
        f_guess = rng.uniform(0, 0.2, size=n_biomarkers)

    bounds = [(0.0, np.inf)] * n_biomarkers

    if method == "lbfgs_exact":
        loss_fn, scipy_method, use_jac = theta_cluster_loss_jac_exact, "L-BFGS-B", True
    elif method == "lbfgs_approx":
        loss_fn, scipy_method, use_jac = theta_cluster_loss_jac, "L-BFGS-B", True
    elif method == "nelder_mead":
        loss_fn, scipy_method, use_jac = theta_cluster_loss, "Nelder-Mead", False
    else:
        raise ValueError(
            f"Unknown method={method!r}; use "
            "'lbfgs_approx', 'lbfgs_exact', or 'nelder_mead'."
        )

    args = (t_pred, X_obs, K, t_span, s, scalar_K, lambda_f, kappa)

    result = minimize(
        loss_fn,
        f_guess,
        args=args,
        method=scipy_method,
        jac=use_jac,
        bounds=bounds,
    )

    return result.x
