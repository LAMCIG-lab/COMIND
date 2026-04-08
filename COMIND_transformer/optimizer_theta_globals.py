import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
from scipy.integrate import cumulative_simpson
from .utils import solve_system
from .sensitivity_lsoda import (
    integrate_linear_sensitivity_lsoda,
    interp_sensitivity_at_obs,
)

# Log-normal prior center for scalar_K (penalty pulls toward this value)
SCALAR_K_CENTER = 0.1
# Lower bound for scalar_K in log-normal penalty/gradient to avoid log(0) and divide-by-zero
SCALAR_K_MIN = 1e-12

def theta_s_loss(params: np.ndarray, t_obs: np.ndarray, x_obs: np.ndarray,
                 K: np.ndarray, t_span: np.ndarray, f: np.ndarray,
                 lambda_s: float = 0.0, lambda_scalar: float = 0.0) -> float:
    """
    Wrapper around `theta_s_loss_multi` for the single-f (no subtype) case.
    Uses a single subtype with all observations assigned to it.
    """
    n_obs = x_obs.shape[0]
    cluster_f = [f]
    observation_assignments = np.zeros(n_obs, dtype=int)
    return theta_s_loss_multi(
        params, t_obs, x_obs, K, t_span,
        cluster_f, observation_assignments,
        lambda_s, lambda_scalar
    )

def theta_s_loss_jac(params: np.ndarray, t_obs: np.ndarray, x_obs: np.ndarray,
                     K: np.ndarray, t_span: np.ndarray, f: np.ndarray,
                     lambda_s: float = 0.0, lambda_scalar: float = 0.0) -> tuple:
    """
    Wrapper around `theta_s_loss_jac_multi` for the single-f (no subtype) case.
    Uses a single subtype with all observations assigned to it.
    """
    n_obs = x_obs.shape[0]
    cluster_f = [f]
    observation_assignments = np.zeros(n_obs, dtype=int)
    return theta_s_loss_jac_multi(
        params, t_obs, x_obs, K, t_span,
        cluster_f, observation_assignments,
        lambda_s, lambda_scalar
    )


def theta_s_loss_multi(params: np.ndarray, t_obs: np.ndarray, x_obs: np.ndarray,
                       K: np.ndarray, t_span: np.ndarray, cluster_f: list,
                       observation_assignments: np.ndarray,
                       lambda_s: float = 0.0, lambda_scalar: float = 0.0,
                       kappa: np.ndarray = None) -> float:
    """
    Loss for s and scalar_K when each observation uses its assigned subtype's f.
    observation_assignments[i] = subtype index for the i-th observation row.
    """
    n_biomarkers = x_obs.shape[1]
    s = params[:n_biomarkers]
    kappa = params[n_biomarkers:-1]
    scalar_K = params[-1]
    x0 = np.zeros(n_biomarkers)

    # One trajectory per subtype (unscaled then scale by s)
    n_subtypes = len(cluster_f)
    x_scaled_list = []
    for k in range(n_subtypes):
        f_k = np.ravel(cluster_f[k])
        x_k = solve_system(x0, f_k, K, t_span, scalar_K, kappa)
        x_scaled_k = s[:, None] * x_k
        x_scaled_list.append(x_scaled_k)

    # Predicted value per observation: vectorize by subtype (one interp per subtype per biomarker)
    x_pred = np.zeros_like(x_obs)
    for k in range(n_subtypes):
        mask_k = (observation_assignments == k)
        if not np.any(mask_k):
            continue
        t_k = t_obs[mask_k]
        for j in range(n_biomarkers):
            x_pred[mask_k, j] = np.interp(t_k, t_span, x_scaled_list[k][j])

    residuals = x_obs - x_pred
    # loss = np.sum(residuals ** 2) + lambda_s * np.sum(s ** 2) + lambda_scalar * scalar_K ** 2  # old L2
    scalar_K_safe = max(scalar_K, SCALAR_K_MIN)
    lognorm_penalty = 0.5 * lambda_scalar * (np.log(scalar_K_safe) - np.log(SCALAR_K_CENTER)) ** 2
    loss = np.sum(residuals ** 2) + lambda_s * np.sum(s ** 2) + lognorm_penalty
    return loss


def theta_s_loss_jac_multi(params: np.ndarray, t_obs: np.ndarray, x_obs: np.ndarray,
                           K: np.ndarray, t_span: np.ndarray, cluster_f: list,
                           observation_assignments: np.ndarray,
                           lambda_s: float = 0.0, lambda_scalar: float = 0.0,
                           kappa: np.ndarray = None) -> tuple:
    """
    Loss and gradient for s, kappa, and scalar_K when each observation uses
    its assigned subtype's forcing term f.
    """
    n_biomarkers = x_obs.shape[1]
    s = params[:n_biomarkers]
    kappa = params[n_biomarkers:-1]
    scalar_K = params[-1]
    x0 = np.zeros(n_biomarkers)
    n_subtypes = len(cluster_f)

    # Trajectories per subtype (unscaled and scaled)
    x_list = []
    x_scaled_list = []
    for k in range(n_subtypes):
        f_k = np.ravel(cluster_f[k])
        x_k = solve_system(x0, f_k, K, t_span, scalar_K, kappa)
        x_list.append(x_k)
        x_scaled_k = s[:, None] * x_k
        x_scaled_list.append(x_scaled_k)

    # Precompute CubicSplines for scalar_K gradient: one per (subtype, biomarker)
    splines_scalar = []
    for k in range(n_subtypes):
        x_k = x_list[k]
        f_k = np.ravel(cluster_f[k])
        K_eff = scalar_K * K + np.diag(kappa)
        Kx_plus_f = (K_eff @ x_k) + f_k[:, None]
        scalar_expr = (1 - x_k) * Kx_plus_f
        splines_scalar.append([
            CubicSpline(t_span, scalar_expr[j], extrapolate=False)
            for j in range(n_biomarkers)
        ])

    # Precompute splines for kappa
    splines_kappa = []
    for k in range(n_subtypes):
        x_k = x_list[k]
        cum_int_kappa = np.array([
            cumulative_simpson((1 - x_k[b]) * x_k[b], x=t_span, initial=0)
            for b in range(n_biomarkers)
        ])
        splines_kappa.append([
            CubicSpline(t_span, cum_int_kappa[b], extrapolate=False)
            for b in range(n_biomarkers)
        ])

    # Predictions and residuals: vectorize by subtype
    x_pred = np.zeros_like(x_obs)
    for k in range(n_subtypes):
        mask_k = (observation_assignments == k)
        if not np.any(mask_k):
            continue
        t_k = t_obs[mask_k]
        for j in range(n_biomarkers):
            x_pred[mask_k, j] = np.interp(t_k, t_span, x_scaled_list[k][j])

    residuals = x_obs - x_pred
    scalar_K_safe = max(scalar_K, SCALAR_K_MIN)
    lognorm_penalty = 0.5 * lambda_scalar * (np.log(scalar_K_safe) - np.log(SCALAR_K_CENTER)) ** 2
    loss = np.sum(residuals ** 2) + lambda_s * np.sum(s ** 2) + lognorm_penalty

    # Gradient for s: vectorize by subtype (one interp per subtype per biomarker, then sum)
    grad_s = np.zeros(n_biomarkers)
    for k in range(n_subtypes):
        mask_k = (observation_assignments == k)
        if not np.any(mask_k):
            continue
        t_k = t_obs[mask_k]
        x_interp_k = np.zeros((np.sum(mask_k), n_biomarkers))
        for j in range(n_biomarkers):
            x_interp_k[:, j] = np.interp(t_k, t_span, x_list[k][j])
        grad_s -= 2 * np.sum(residuals[mask_k] * x_interp_k, axis=0)
    grad_s += 2 * lambda_s * s

    # Gradient for kappa
    grad_kappa = np.zeros(n_biomarkers)
    for k in range(n_subtypes):
        mask_k = (observation_assignments == k)
        if not np.any(mask_k):
            continue
        t_k = np.clip(t_obs[mask_k], t_span[0], t_span[-1])
        kappa_sens_k = np.zeros((np.sum(mask_k), n_biomarkers))
        for b in range(n_biomarkers):
            kappa_sens_k[:, b] = splines_kappa[k][b](t_k)
        grad_kappa -= 2 * np.sum(residuals[mask_k] * (kappa_sens_k * s[None, :]), axis=0)

    # Gradient for scalar_K: vectorized by subtype
    grad_scalar_K = 0.0
    n_obs = x_obs.shape[0]
    for k in range(n_subtypes):
        mask_k = (observation_assignments == k)
        if not np.any(mask_k):
            continue
        t_k = np.clip(t_obs[mask_k], t_span[0], t_span[-1])
        scalar_obs_k = np.zeros((np.sum(mask_k), n_biomarkers))
        for j in range(n_biomarkers):
            scalar_obs_k[:, j] = splines_scalar[k][j](t_k)
        grad_scalar_K -= 2 * np.sum(residuals[mask_k] * s[None, :] * scalar_obs_k)
    lognorm_grad = lambda_scalar * (np.log(scalar_K_safe) - np.log(SCALAR_K_CENTER)) / scalar_K_safe
    grad_scalar_K += lognorm_grad

    grad = np.concatenate([grad_s, grad_kappa, [grad_scalar_K]])
    return loss, grad


def theta_s_loss_jac_exact_multi(params: np.ndarray, t_obs: np.ndarray, x_obs: np.ndarray,
                                 K: np.ndarray, t_span: np.ndarray, cluster_f: list,
                                 observation_assignments: np.ndarray,
                                 lambda_s: float = 0.0, lambda_scalar: float = 0.0) -> tuple:
    """
    Same loss as theta_s_loss_multi; gradients via sensitivity IVPs integrated
    with LSODA on the stored forward trajectory.
    """
    n_biomarkers = x_obs.shape[1]
    s = params[:n_biomarkers]
    kappa = params[n_biomarkers:-1]
    scalar_K = params[-1]
    x0 = np.zeros(n_biomarkers)
    n_subtypes = len(cluster_f)
    K_eff = scalar_K * K + np.diag(kappa)

    x_list = []
    x_scaled_list = []
    for k in range(n_subtypes):
        f_k = np.ravel(cluster_f[k])
        x_k = solve_system(x0, f_k, K, t_span, scalar_K, kappa)
        x_list.append(x_k)
        x_scaled_k = s[:, None] * x_k
        x_scaled_list.append(x_scaled_k)

    x_pred = np.zeros_like(x_obs)
    for k in range(n_subtypes):
        mask_k = (observation_assignments == k)
        if not np.any(mask_k):
            continue
        t_k = t_obs[mask_k]
        for j in range(n_biomarkers):
            x_pred[mask_k, j] = np.interp(t_k, t_span, x_scaled_list[k][j])

    residuals = x_obs - x_pred
    scalar_K_safe = max(scalar_K, SCALAR_K_MIN)
    lognorm_penalty = 0.5 * lambda_scalar * (np.log(scalar_K_safe) - np.log(SCALAR_K_CENTER)) ** 2
    loss = np.sum(residuals ** 2) + lambda_s * np.sum(s ** 2) + lognorm_penalty

    sens_records = []
    for k, f_k in enumerate(cluster_f):
        f_k = np.ravel(f_k)
        x_k = x_list[k]
        u_sk = integrate_linear_sensitivity_lsoda(
            t_span, x_k, K_eff, K, f_k, "scalar_K"
        )
        u_kappa = [
            integrate_linear_sensitivity_lsoda(
                t_span, x_k, K_eff, K, f_k, "kappa_diag", b
            )
            for b in range(n_biomarkers)
        ]
        sens_records.append((x_k, u_sk, u_kappa))

    grad_s = 2.0 * lambda_s * s.copy()
    grad_kappa = np.zeros(n_biomarkers)
    grad_scalar_K = 0.0

    for k, (x_k, u_sk, u_kappa) in enumerate(sens_records):
        mask_k = (observation_assignments == k)
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

    grad_scalar_K += lambda_scalar * (
        (np.log(scalar_K_safe) - np.log(SCALAR_K_CENTER)) / scalar_K_safe
    )
    grad = np.concatenate([grad_s, grad_kappa, [grad_scalar_K]])
    return loss, grad


def theta_s_loss_jac_exact(params: np.ndarray, t_obs: np.ndarray, x_obs: np.ndarray,
                           K: np.ndarray, t_span: np.ndarray, f: np.ndarray,
                           lambda_s: float = 0.0, lambda_scalar: float = 0.0) -> tuple:
    n_obs = x_obs.shape[0]
    cluster_f = [f]
    observation_assignments = np.zeros(n_obs, dtype=int)
    return theta_s_loss_jac_exact_multi(
        params, t_obs, x_obs, K, t_span,
        cluster_f, observation_assignments,
        lambda_s, lambda_scalar,
    )


def fit_theta_globals(X_obs: np.ndarray, dt_obs: np.ndarray, ids: np.ndarray, K: np.ndarray,
                      t_span: np.ndarray, use_jacobian: bool = None,
                      f: np.ndarray = None,
                      beta_pred: np.ndarray = None,
                      s_guess: np.ndarray = None, scalar_K_guess: float = None, kappa_guess: np.ndarray = None,
                      lambda_s: float = 0.0, lambda_scalar: float = 0.0,
                      rng: np.random.Generator = None,
                      assignments: np.ndarray = None,
                      cluster_f: list = None,
                      solver_stage: str = None) -> tuple:
    """
    Optimizes global scaling parameter s (supremum) and scalar_K for all patients.
    If assignments and cluster_f are provided, uses assignment-aware loss (each observation
    predicted with its assigned subtype's f). Otherwise uses single f for all (legacy).

    solver_stage:
        'lbfgs_approx' — L-BFGS-B, jac=False (SciPy finite-difference gradient).
        'lbfgs_exact'  — L-BFGS-B, jac=True, exact sensitivities solved with LSODA.
        'nelder_mead'  — derivative-free, loss only.

    If solver_stage is None, use_jacobian maps: True -> lbfgs_exact, False -> lbfgs_approx.
    """
    if rng is None:
        rng = np.random.default_rng(75)

    if solver_stage is None:
        if use_jacobian is None:
            use_jacobian = False
        solver_stage = "lbfgs_exact" if use_jacobian else "lbfgs_approx"

    unique_ids = np.unique(ids)
    id_to_index = {pid: i for i, pid in enumerate(unique_ids)}
    index_array = np.array([id_to_index[i] for i in ids])  # shape: (n_obs,)
    t_pred = dt_obs + beta_pred[index_array]

    n_biomarkers = X_obs.shape[1]

    # Initial guesses if None
    if s_guess is None:
        s_guess = np.ones(n_biomarkers)
    if scalar_K_guess is None:
        scalar_K_guess = 1.0
    if kappa_guess is None:
        kappa_guess = np.zeros(K.shape[0])
    initial_params = np.concatenate([s_guess, kappa_guess, [scalar_K_guess]])
    bounds = ([(0.0, np.inf)] * n_biomarkers +             # s bounds
              [(0.0, np.inf)] * n_biomarkers +             # kappa bounds
              [(0.0, np.inf)])                             # scalar_K bound

    use_multi = (assignments is not None and cluster_f is not None)
    if use_multi:
        observation_assignments = assignments[index_array]  # assignment per observation row
        if solver_stage == "lbfgs_exact":
            loss_function = theta_s_loss_jac_exact_multi
            use_jac = True
        elif solver_stage == "lbfgs_approx":
            loss_function = theta_s_loss_multi
            use_jac = False
        elif solver_stage == "nelder_mead":
            loss_function = theta_s_loss_multi
            use_jac = False
        else:
            raise ValueError(
                f"Unknown solver_stage={solver_stage!r}; use "
                "'lbfgs_approx', 'lbfgs_exact', or 'nelder_mead'."
            )
        args = (t_pred, X_obs, K, t_span, cluster_f, observation_assignments, lambda_s, lambda_scalar)
    else:
        if f is None:
            raise ValueError("Must provide either f or (assignments and cluster_f).")
        if solver_stage == "lbfgs_exact":
            loss_function = theta_s_loss_jac_exact
            use_jac = True
        elif solver_stage == "lbfgs_approx":
            loss_function = theta_s_loss
            use_jac = False
        elif solver_stage == "nelder_mead":
            loss_function = theta_s_loss
            use_jac = False
        else:
            raise ValueError(
                f"Unknown solver_stage={solver_stage!r}; use "
                "'lbfgs_approx', 'lbfgs_exact', or 'nelder_mead'."
            )
        args = (t_pred, X_obs, K, t_span, f, lambda_s, lambda_scalar)

    if solver_stage == "nelder_mead":
        result = minimize(
            loss_function,
            initial_params,
            args=args,
            method="Nelder-Mead",
            bounds=bounds,
        )
    else:
        result = minimize(
            loss_function,
            initial_params,
            args=args,
            method="L-BFGS-B",
            jac=use_jac,
            bounds=bounds,
        )

    fitted_params = result.x
    s_fit = fitted_params[:n_biomarkers]
    kappa_fit = fitted_params[n_biomarkers:-1]
    scalar_K_fit = fitted_params[-1]

    return s_fit, kappa_fit,scalar_K_fit
