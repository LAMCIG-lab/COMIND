import numpy as np
from scipy.optimize import minimize
from .utils import solve_system
from .sensitivity_lsoda import (
    integrate_all_sensitivities_lsoda,
    interp_sensitivity_at_obs,
)
from .sensitivity_rk4 import integrate_all_sensitivities_rk4

# Log-normal prior center for scalar_K (penalty pulls toward this value in real space)
SCALAR_K_CENTER = 0.25
# Lower bound for scalar_K in log-normal penalty/gradient to avoid log(0) and divide-by-zero
SCALAR_K_MIN = 1e-12
SPARSE_PSEUDO_HUBER_DELTA = 0.05


def _pseudo_huber_penalty(values: np.ndarray, delta: float = SPARSE_PSEUDO_HUBER_DELTA) -> float:
    """Elementwise pseudo-Huber penalty summed over values."""
    values = np.asarray(values, dtype=float)
    scaled = values / delta
    return float(np.sum((delta ** 2) * (np.sqrt(1.0 + scaled ** 2) - 1.0)))


def _pseudo_huber_grad(values: np.ndarray, delta: float = SPARSE_PSEUDO_HUBER_DELTA) -> np.ndarray:
    """Gradient of summed pseudo-Huber penalty w.r.t. values."""
    values = np.asarray(values, dtype=float)
    return values / np.sqrt(1.0 + (values / delta) ** 2)


def _lognorm_scalar_K_penalty(
    scalar_K_safe: float, lambda_scalar: float, scalar_K_center: float
) -> float:
    return 0.5 * lambda_scalar * (
        np.log(scalar_K_safe) - np.log(scalar_K_center)
    ) ** 2


def _lognorm_scalar_K_grad(
    scalar_K_safe: float, lambda_scalar: float, scalar_K_center: float
) -> float:
    return lambda_scalar * (
        (np.log(scalar_K_safe) - np.log(scalar_K_center)) / scalar_K_safe
    )


def _solve_and_residuals(
    params: np.ndarray,
    t_obs: np.ndarray,
    x_obs: np.ndarray,
    K: np.ndarray,
    t_span: np.ndarray,
    cluster_f: list,
    observation_assignments: np.ndarray,
    ode_method: str = "LSODA",
):
    """Unpack globals, solve subtype trajectories, build x_pred and residuals."""
    n_biomarkers = x_obs.shape[1]
    s = params[:n_biomarkers]
    kappa = params[n_biomarkers:-1]
    scalar_K = params[-1]
    x0 = np.zeros(n_biomarkers)
    n_subtypes = len(cluster_f)

    x_list = []
    for k in range(n_subtypes):
        f_k = np.ravel(cluster_f[k])
        x_k = solve_system(
            x0, f_k, K, t_span, scalar_K, kappa, ode_method=ode_method
        )
        x_list.append(x_k)

    x_pred = np.zeros_like(x_obs)
    for k in range(n_subtypes):
        mask_k = observation_assignments == k
        if not np.any(mask_k):
            continue
        t_k = t_obs[mask_k]
        x_scaled_k = s[:, None] * x_list[k]
        for j in range(n_biomarkers):
            x_pred[mask_k, j] = np.interp(t_k, t_span, x_scaled_k[j])

    residuals = x_obs - x_pred
    return s, kappa, scalar_K, x_list, x_pred, residuals


def theta_s_loss_multi(
    params: np.ndarray,
    t_obs: np.ndarray,
    x_obs: np.ndarray,
    K: np.ndarray,
    t_span: np.ndarray,
    cluster_f: list,
    observation_assignments: np.ndarray,
    lambda_s: float = 0.0,
    lambda_scalar: float = 0.0,
    lambda_kappa: float = 0.0,
    scalar_K_center: float = SCALAR_K_CENTER,
    ode_method: str = "LSODA",
) -> float:
    """
    Loss for s and scalar_K when each observation uses its assigned subtype's f.
    observation_assignments[i] = subtype index for the i-th observation row.
    """
    s, kappa, scalar_K, _x_list, _x_pred, residuals = _solve_and_residuals(
        params, t_obs, x_obs, K, t_span, cluster_f, observation_assignments,
        ode_method=ode_method,
    )
    scalar_K_safe = max(scalar_K, SCALAR_K_MIN)
    lognorm_penalty = _lognorm_scalar_K_penalty(
        scalar_K_safe, lambda_scalar, scalar_K_center
    )
    return (
        np.sum(residuals ** 2)
        + lambda_s * np.sum(s ** 2)
        + lognorm_penalty
        + lambda_kappa * np.sum(kappa)
    )


def theta_s_loss_jac_exact_multi(
    params: np.ndarray,
    t_obs: np.ndarray,
    x_obs: np.ndarray,
    K: np.ndarray,
    t_span: np.ndarray,
    cluster_f: list,
    observation_assignments: np.ndarray,
    lambda_s: float = 0.0,
    lambda_scalar: float = 0.0,
    lambda_kappa: float = 0.0,
    scalar_K_center: float = SCALAR_K_CENTER,
    ode_method: str = "LSODA",
) -> tuple:
    """
    Same loss as theta_s_loss_multi; gradients via sensitivity IVPs integrated
    with LSODA on the stored forward trajectory.
    """
    s, kappa, scalar_K, x_list, _x_pred, residuals = _solve_and_residuals(
        params, t_obs, x_obs, K, t_span, cluster_f, observation_assignments,
        ode_method=ode_method,
    )
    n_biomarkers = x_obs.shape[1]
    scalar_K_safe = max(scalar_K, SCALAR_K_MIN)
    lognorm_penalty = _lognorm_scalar_K_penalty(
        scalar_K_safe, lambda_scalar, scalar_K_center
    )
    loss = (
        np.sum(residuals ** 2)
        + lambda_s * np.sum(s ** 2)
        + lognorm_penalty
        + lambda_kappa * np.sum(kappa)
    )

    K_eff = scalar_K * K + np.diag(kappa)
    sens_records = []
    for k, f_k in enumerate(cluster_f):
        f_k = np.ravel(f_k)
        x_k = x_list[k]
        U_k = integrate_all_sensitivities_lsoda(t_span, x_k, K_eff, K, f_k)
        u_sk = U_k[:, 0, :]
        u_kappa = [U_k[:, b + 1, :] for b in range(n_biomarkers)]
        sens_records.append((x_k, u_sk, u_kappa))

    grad_s = 2.0 * lambda_s * s.copy()
    grad_kappa = np.zeros(n_biomarkers)
    grad_scalar_K = 0.0

    for k, (x_k, u_sk, u_kappa) in enumerate(sens_records):
        mask_k = observation_assignments == k
        if not np.any(mask_k):
            continue
        t_k = np.clip(t_obs[mask_k], t_span[0], t_span[-1])
        res_k = residuals[mask_k]

        x_at = interp_sensitivity_at_obs(x_k, t_span, t_k)
        u_sk_at = interp_sensitivity_at_obs(u_sk, t_span, t_k)
        grad_s -= 2.0 * np.sum(res_k * x_at, axis=0)
        grad_scalar_K -= 2.0 * np.sum(res_k * s[None, :] * u_sk_at)

        for b in range(n_biomarkers):
            u_b_at = interp_sensitivity_at_obs(u_kappa[b], t_span, t_k)
            grad_kappa[b] -= 2.0 * np.sum(res_k * s[None, :] * u_b_at)

    grad_kappa += lambda_kappa * np.sign(kappa)

    grad_scalar_K += _lognorm_scalar_K_grad(
        scalar_K_safe, lambda_scalar, scalar_K_center
    )
    grad = np.concatenate([grad_s, grad_kappa, [grad_scalar_K]])
    return loss, grad


def theta_s_loss_jac_rk4_multi(
    params: np.ndarray,
    t_obs: np.ndarray,
    x_obs: np.ndarray,
    K: np.ndarray,
    t_span: np.ndarray,
    cluster_f: list,
    observation_assignments: np.ndarray,
    lambda_s: float = 0.0,
    lambda_scalar: float = 0.0,
    lambda_kappa: float = 0.0,
    scalar_K_center: float = SCALAR_K_CENTER,
    ode_method: str = "LSODA",
) -> tuple:
    """
    Same loss as theta_s_loss_multi; gradients via RK4 sensitivities on t_span.
    """
    s, kappa, scalar_K, x_list, _x_pred, residuals = _solve_and_residuals(
        params, t_obs, x_obs, K, t_span, cluster_f, observation_assignments,
        ode_method=ode_method,
    )
    n_biomarkers = x_obs.shape[1]
    scalar_K_safe = max(scalar_K, SCALAR_K_MIN)
    lognorm_penalty = _lognorm_scalar_K_penalty(
        scalar_K_safe, lambda_scalar, scalar_K_center
    )
    loss = (
        np.sum(residuals ** 2)
        + lambda_s * np.sum(s ** 2)
        + lognorm_penalty
        + lambda_kappa * np.sum(kappa)
    )

    K_eff = scalar_K * K + np.diag(kappa)
    sens_records = []
    for k, f_k in enumerate(cluster_f):
        f_k = np.ravel(f_k)
        x_k = x_list[k]
        U_k = integrate_all_sensitivities_rk4(
            t_span, x_k, K_eff, K, f_k, param_type="globals"
        )
        u_sk = U_k[:, 0, :]
        u_kappa = [U_k[:, b + 1, :] for b in range(n_biomarkers)]
        sens_records.append((x_k, u_sk, u_kappa))

    grad_s = 2.0 * lambda_s * s.copy()
    grad_kappa = np.zeros(n_biomarkers)
    grad_scalar_K = 0.0

    for k, (x_k, u_sk, u_kappa) in enumerate(sens_records):
        mask_k = observation_assignments == k
        if not np.any(mask_k):
            continue
        t_k = np.clip(t_obs[mask_k], t_span[0], t_span[-1])
        res_k = residuals[mask_k]

        x_at = interp_sensitivity_at_obs(x_k, t_span, t_k)
        u_sk_at = interp_sensitivity_at_obs(u_sk, t_span, t_k)
        grad_s -= 2.0 * np.sum(res_k * x_at, axis=0)
        grad_scalar_K -= 2.0 * np.sum(res_k * s[None, :] * u_sk_at)

        for b in range(n_biomarkers):
            u_b_at = interp_sensitivity_at_obs(u_kappa[b], t_span, t_k)
            grad_kappa[b] -= 2.0 * np.sum(res_k * s[None, :] * u_b_at)

    grad_kappa += lambda_kappa * np.sign(kappa)

    grad_scalar_K += _lognorm_scalar_K_grad(
        scalar_K_safe, lambda_scalar, scalar_K_center
    )
    grad = np.concatenate([grad_s, grad_kappa, [grad_scalar_K]])
    return loss, grad


def fit_theta_globals(
    X_obs: np.ndarray,
    dt_obs: np.ndarray,
    ids: np.ndarray,
    K: np.ndarray,
    t_span: np.ndarray,
    cluster_f: list,
    assignments: np.ndarray,
    beta_pred: np.ndarray = None,
    s_guess: np.ndarray = None,
    scalar_K_guess: float = None,
    kappa_guess: np.ndarray = None,
    lambda_s: float = 0.0,
    lambda_scalar: float = 0.0,
    lambda_kappa: float = 0.0,
    scalar_K_center: float = SCALAR_K_CENTER,
    method: str = "lbfgs_approx",
    strict_tol: bool = False,
    ode_method: str = "LSODA",
) -> tuple:
    """
    Optimizes global (s, kappa, scalar_K) using assignment-aware subtype trajectories.

    method:
        'lbfgs_approx' — L-BFGS-B, jac=False (SciPy finite-difference gradient).
        'lbfgs_exact'  — L-BFGS-B, jac=True, exact sensitivities solved with LSODA.
        'lbfgs_rk4'    — L-BFGS-B, jac=True, sensitivities via RK4 on t_span.
        'nelder_mead'  — derivative-free, loss only.
    """
    t_pred = dt_obs + beta_pred[ids]
    n_biomarkers = X_obs.shape[1]

    if s_guess is None:
        s_guess = np.ones(n_biomarkers)
    if scalar_K_guess is None:
        scalar_K_guess = 1.0
    if kappa_guess is None:
        kappa_guess = np.zeros(K.shape[0])
    initial_params = np.concatenate([s_guess, kappa_guess, [scalar_K_guess]])
    bounds = (
        [(0.0, np.inf)] * n_biomarkers
        + [(0.0, np.inf)] * n_biomarkers
        + [(0.0, np.inf)]
    )

    observation_assignments = assignments[ids]

    if method == "lbfgs_exact":
        loss_fn, scipy_method, use_jac = theta_s_loss_jac_exact_multi, "L-BFGS-B", True
    elif method == "lbfgs_rk4":
        loss_fn, scipy_method, use_jac = theta_s_loss_jac_rk4_multi, "L-BFGS-B", True
    elif method == "lbfgs_approx":
        loss_fn, scipy_method, use_jac = theta_s_loss_multi, "L-BFGS-B", False
    elif method == "nelder_mead":
        loss_fn, scipy_method, use_jac = theta_s_loss_multi, "Nelder-Mead", False
    else:
        raise ValueError(
            f"Unknown method={method!r}; use "
            "'lbfgs_approx', 'lbfgs_exact', 'lbfgs_rk4', or 'nelder_mead'."
        )

    args = (
        t_pred,
        X_obs,
        K,
        t_span,
        cluster_f,
        observation_assignments,
        lambda_s,
        lambda_scalar,
        lambda_kappa,
        scalar_K_center,
        ode_method,
    )

    minimize_kwargs = dict(
        fun=loss_fn,
        x0=initial_params,
        args=args,
        method=scipy_method,
        jac=use_jac,
        bounds=bounds,
    )
    if scipy_method == "L-BFGS-B" and strict_tol:
        minimize_kwargs["options"] = {
            "maxiter": 200,
            "ftol": 2.2e-10,
            "gtol": 1e-6,
        }

    result = minimize(**minimize_kwargs)

    fitted_params = result.x
    s_fit = fitted_params[:n_biomarkers]
    kappa_fit = fitted_params[n_biomarkers:-1]
    scalar_K_fit = fitted_params[-1]

    return s_fit, kappa_fit, scalar_K_fit
