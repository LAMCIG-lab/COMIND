import numpy as np
from scipy.optimize import minimize
from scipy.integrate import cumulative_simpson
from scipy.interpolate import CubicSpline
from .utils import solve_system
from .sensitivity_lsoda import (
    integrate_linear_sensitivity_lsoda,
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


def theta_cluster_loss(params: np.ndarray, t_obs: np.ndarray, x_obs: np.ndarray,
                       K: np.ndarray, t_span: np.ndarray, s: np.ndarray, scalar_K: float,
                       lambda_f: float, kappa: np.ndarray = None) -> float:
    """
    Computes loss for optimizing cluster-level f (with fixed global s and scalar_K).
    
    Parameters
    ----------
    params : np.ndarray
        Current guess for f (shape: n_biomarkers).
    t_obs : np.ndarray
        Observation times.
    x_obs : np.ndarray
        Observed biomarker values (shape: n_obs x n_biomarkers).
    K : np.ndarray
        Connectivity matrix.
    t_span : np.ndarray
        Time points for trajectory simulation.
    s : np.ndarray
        Fixed global scaling parameter (shape: n_biomarkers).
    scalar_K : float
        Fixed global scalar_K parameter.
    lambda_f : float
        Regularization strength for f.
        
    Returns
    -------
    float
        Loss value.
    """
    n_biomarkers = x_obs.shape[1]
    f = params
    x0 = np.zeros(n_biomarkers)

    x = solve_system(x0, f, K, t_span, scalar_K, kappa)
    x_scaled = s[:, None] * x

    t_obs_clamped = np.clip(t_obs, t_span[0], t_span[-1])
    x_pred = np.zeros_like(x_obs)  # (n_obs, n_biomarkers)
    for j in range(n_biomarkers):
        x_pred[:, j] = np.interp(t_obs_clamped, t_span, x_scaled[j])
    
    residuals = x_obs - x_pred
    loss = np.sum(residuals**2) + lambda_f * _pseudo_huber_penalty(f)
    
    return loss

def theta_cluster_loss_jac(params: np.ndarray, t_obs: np.ndarray, x_obs: np.ndarray,
                           K: np.ndarray, t_span: np.ndarray, s: np.ndarray, scalar_K: float,
                           lambda_f: float, kappa: np.ndarray = None) -> tuple:
    """
    Computes loss and gradient for optimizing cluster-level f (with fixed global s and scalar_K).
    
    Parameters
    ----------
    params : np.ndarray
        Current guess for f (shape: n_biomarkers).
    t_obs : np.ndarray
        Observation times.
    x_obs : np.ndarray
        Observed biomarker values (shape: n_obs x n_biomarkers).
    K : np.ndarray
        Connectivity matrix.
    t_span : np.ndarray
        Time points for trajectory simulation.
    s : np.ndarray
        Fixed global scaling parameter (shape: n_biomarkers).
    scalar_K : float
        Fixed global scalar_K parameter.
    lambda_f : float
        Regularization strength for f.
        
    Returns
    -------
    tuple
        Loss scalar and gradient vector grad_f.
    """
    n_biomarkers = x_obs.shape[1]
    f = params
    x0 = np.zeros(n_biomarkers)

    x = solve_system(x0, f, K, t_span, scalar_K, kappa)
    x_scaled = s[:, None] * x

    t_obs_clamped = np.clip(t_obs, t_span[0], t_span[-1])

    x_pred = np.zeros_like(x_obs)  # (n_obs, n_biomarkers)
    for j in range(n_biomarkers):
        x_pred[:, j] = np.interp(t_obs_clamped, t_span, x_scaled[j])
    
    residuals = x_obs - x_pred
    loss = np.sum(residuals**2) + lambda_f * _pseudo_huber_penalty(f)
    
    ### Gradient computations
    
    ## Gradient with respect to f
    cum_int = np.array([
        cumulative_simpson(1 - x[i], x=t_span, initial=0)
        for i in range(n_biomarkers)
    ])

    df_obs = np.zeros_like(x_obs)
    for i in range(n_biomarkers):
        cs_integ = CubicSpline(t_span, cum_int[i], extrapolate=False)
        df_obs[:, i] = cs_integ(t_obs_clamped)
    
    # Need to account for s scaling: d/ds(s*x) with respect to f
    grad_f = -2 * np.sum(residuals * (df_obs * s[None, :]), axis=0) + lambda_f * _pseudo_huber_grad(f)
    
    return loss, grad_f


def theta_cluster_loss_jac_exact(params: np.ndarray, t_obs: np.ndarray, x_obs: np.ndarray,
                                 K: np.ndarray, t_span: np.ndarray, s: np.ndarray, scalar_K: float,
                                 lambda_f: float, kappa: np.ndarray = None) -> tuple:
    """
    Loss and gradient w.r.t. f using exact sensitivities solved with LSODA.
    """
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
    loss = np.sum(residuals**2) + lambda_f * _pseudo_huber_penalty(f)

    grad_f = lambda_f * _pseudo_huber_grad(f)
    for b in range(n_biomarkers):
        w_b = integrate_linear_sensitivity_lsoda(
            t_span, x, K_eff, K, f, "f_diag", b
        )
        w_at = interp_sensitivity_at_obs(w_b, t_span, t_obs_clamped)
        grad_f[b] -= 2.0 * np.sum(residuals * s[None, :] * w_at)

    return loss, grad_f


def fit_theta_cluster(X_obs: np.ndarray, dt_obs: np.ndarray, ids: np.ndarray, K: np.ndarray,
                      t_span: np.ndarray, *,
                      s: np.ndarray, scalar_K: float, lambda_f: float,
                      use_jacobian: bool = None,
                      beta_pred: np.ndarray = None,
                      f_guess: np.ndarray = None,
                      rng: np.random.Generator = None,
                      kappa: np.ndarray = None,
                      solver_stage: str = None) -> np.ndarray:
    """
    Optimizes cluster-level f for patients in a specific cluster (with fixed global s and scalar_K).

    solver_stage: 'lbfgs_approx' | 'lbfgs_exact' | 'nelder_mead'
    If None, use_jacobian maps True -> lbfgs_exact, False -> lbfgs_approx.
    """
    if rng is None:
        rng = np.random.default_rng(75)

    if solver_stage is None:
        if use_jacobian is None:
            use_jacobian = False
        solver_stage = "lbfgs_exact" if use_jacobian else "lbfgs_approx"
    unique_ids = np.unique(ids)
    id_to_index = {pid: i for i, pid in enumerate(unique_ids)}
    index_array = np.array([id_to_index[i] for i in ids])  # shape: (n_obs_cluster,)
    t_pred = dt_obs + beta_pred[index_array]
    
    n_biomarkers = X_obs.shape[1]
    
    # Initial guess if None
    if f_guess is None:
        f_guess = rng.uniform(0, 0.2, size=n_biomarkers)
    
    # Bounds: f must be non-negative
    bounds = [(0.0, np.inf)] * n_biomarkers
    
    if solver_stage == "lbfgs_exact":
        loss_function = theta_cluster_loss_jac_exact
        use_jac = True
    elif solver_stage == "lbfgs_approx":
        loss_function = theta_cluster_loss
        use_jac = False
    elif solver_stage == "nelder_mead":
        loss_function = theta_cluster_loss
        use_jac = False
    else:
        raise ValueError(
            f"Unknown solver_stage={solver_stage!r}; use "
            "'lbfgs_approx', 'lbfgs_exact', or 'nelder_mead'."
        )

    args = (t_pred, X_obs, K, t_span, s, scalar_K, lambda_f, kappa)

    if solver_stage == "nelder_mead":
        result = minimize(
            loss_function,
            f_guess,
            args=args,
            method="Nelder-Mead",
            bounds=bounds,
        )
    else:
        result = minimize(
            loss_function,
            f_guess,
            args=args,
            method="L-BFGS-B",
            jac=use_jac,
            bounds=bounds,
        )
    
    f_fit = result.x

    return f_fit

