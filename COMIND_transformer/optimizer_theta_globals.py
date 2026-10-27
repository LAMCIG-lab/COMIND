import numpy as np
from scipy.optimize import minimize
from scipy.integrate import cumulative_simpson
from scipy.interpolate import CubicSpline
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


def resolve_s_mask(n_biomarkers: int, fit_s: bool = True, s_mask=None) -> np.ndarray:
    """Boolean mask of which ``s`` entries are free (True) vs held at 1 (False).

    ``fit_s=False`` fixes every entry. ``s_mask is None`` with ``fit_s=True``
    leaves every entry free (legacy behavior).
    """
    if not fit_s:
        return np.zeros(n_biomarkers, dtype=bool)
    if s_mask is None:
        return np.ones(n_biomarkers, dtype=bool)
    mask = np.asarray(s_mask, dtype=bool).ravel()
    if mask.shape != (n_biomarkers,):
        raise ValueError(
            f"s_mask must have shape ({n_biomarkers},), got {mask.shape}"
        )
    return mask


def _s_bounds(s_lower, n_biomarkers: int, s_upper=None) -> np.ndarray:
    """(n_biomarkers, 2) array of ``(lower, upper)`` s bounds.

    ``s_lower=None`` means 0. ``s_upper=None`` means inf. A scalar upper/lower
    is broadcast to every biomarker.
    """
    if s_lower is None:
        lower = np.zeros(n_biomarkers)
    else:
        lower = np.asarray(s_lower, dtype=float).ravel()
        if lower.size == 1:
            lower = np.full(n_biomarkers, float(lower[0]))
    if lower.shape != (n_biomarkers,):
        raise ValueError(f"s_lower must have shape ({n_biomarkers},), got {lower.shape}")
    if s_upper is None:
        upper = np.full(n_biomarkers, np.inf)
    else:
        upper = np.asarray(s_upper, dtype=float).ravel()
        if upper.size == 1:
            upper = np.full(n_biomarkers, float(upper[0]))
        if upper.shape != (n_biomarkers,):
            raise ValueError(f"s_upper must have shape ({n_biomarkers},), got {upper.shape}")
    return np.column_stack([lower, upper])


def _unpack_globals(params: np.ndarray, n_biomarkers: int, s_mask: np.ndarray):
    """Unpack (s, kappa, scalar_K); masked-off ``s`` entries are held at 1."""
    s = np.ones(n_biomarkers, dtype=float)
    n_free = int(np.count_nonzero(s_mask))
    if n_free:
        s[s_mask] = params[:n_free]
        kappa = params[n_free:-1]
    else:
        kappa = params[:-1]
    return s, kappa, float(params[-1])


def _pack_global_grad(grad_s, grad_kappa, grad_scalar_K, s_mask: np.ndarray) -> np.ndarray:
    if np.any(s_mask):
        return np.concatenate([grad_s[s_mask], grad_kappa, [grad_scalar_K]])
    return np.concatenate([grad_kappa, [grad_scalar_K]])


def _s_ridge_penalty(s: np.ndarray, lambda_s: float, s_mask: np.ndarray) -> float:
    if lambda_s == 0.0 or not np.any(s_mask):
        return 0.0
    return float(lambda_s * np.sum(s[s_mask] ** 2))


def _solve_and_residuals(
    params: np.ndarray,
    t_obs: np.ndarray,
    x_obs: np.ndarray,
    K: np.ndarray,
    t_span: np.ndarray,
    cluster_f: list,
    observation_assignments: np.ndarray,
    fit_s: bool = True,
    ode_method: str = "LSODA",
    obs_weight: np.ndarray = None,
    s_mask=None,
):
    """Unpack globals, solve subtype trajectories, build x_pred and residuals.

    ``obs_weight`` is an optional (n_rows, n_biomarkers) array (1.0 =
    observed, 0.0 = missing). Residuals are masked here so every caller
    gets pre-masked residuals. ``None`` treats every entry as observed.

    ``s_mask`` is an optional boolean vector (True = optimize that ``s``
    entry, False = hold it at 1). Packed ``params`` contain only free
    ``s`` entries, then kappa, then scalar_K.
    """
    n_biomarkers = x_obs.shape[1]
    s_mask_bool = resolve_s_mask(n_biomarkers, fit_s, s_mask)
    s, kappa, scalar_K = _unpack_globals(params, n_biomarkers, s_mask_bool)
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
    if obs_weight is not None:
        residuals = residuals * obs_weight
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
    fit_s: bool = True,
    ode_method: str = "LSODA",
    obs_weight: np.ndarray = None,
    s_mask=None,
) -> float:
    """
    Loss for s and scalar_K when each observation uses its assigned subtype's f.
    observation_assignments[i] = subtype index for the i-th observation row.
    """
    s, kappa, scalar_K, _x_list, _x_pred, residuals = _solve_and_residuals(
        params, t_obs, x_obs, K, t_span, cluster_f, observation_assignments,
        fit_s=fit_s,
        ode_method=ode_method,
        obs_weight=obs_weight,
        s_mask=s_mask,
    )
    scalar_K_safe = max(scalar_K, SCALAR_K_MIN)
    lognorm_penalty = _lognorm_scalar_K_penalty(
        scalar_K_safe, lambda_scalar, scalar_K_center
    )
    s_mask_bool = resolve_s_mask(x_obs.shape[1], fit_s, s_mask)
    s_penalty = _s_ridge_penalty(s, lambda_s, s_mask_bool)
    return (
        np.sum(residuals ** 2)
        + s_penalty
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
    fit_s: bool = True,
    ode_method: str = "LSODA",
    obs_weight: np.ndarray = None,
    s_mask=None,
) -> tuple:
    """
    Same loss as theta_s_loss_multi; gradients via sensitivity IVPs integrated
    with LSODA on the stored forward trajectory.
    """
    s, kappa, scalar_K, x_list, _x_pred, residuals = _solve_and_residuals(
        params, t_obs, x_obs, K, t_span, cluster_f, observation_assignments,
        fit_s=fit_s,
        ode_method=ode_method,
        obs_weight=obs_weight,
        s_mask=s_mask,
    )
    n_biomarkers = x_obs.shape[1]
    s_mask_bool = resolve_s_mask(n_biomarkers, fit_s, s_mask)
    any_s = bool(np.any(s_mask_bool))
    scalar_K_safe = max(scalar_K, SCALAR_K_MIN)
    lognorm_penalty = _lognorm_scalar_K_penalty(
        scalar_K_safe, lambda_scalar, scalar_K_center
    )
    s_penalty = _s_ridge_penalty(s, lambda_s, s_mask_bool)
    loss = (
        np.sum(residuals ** 2)
        + s_penalty
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

    grad_s = 2.0 * lambda_s * s.copy() if any_s else None
    grad_kappa = np.zeros(n_biomarkers)
    grad_scalar_K = 0.0
    s_factor = s[None, :]

    for k, (x_k, u_sk, u_kappa) in enumerate(sens_records):
        mask_k = observation_assignments == k
        if not np.any(mask_k):
            continue
        t_k = np.clip(t_obs[mask_k], t_span[0], t_span[-1])
        res_k = residuals[mask_k]

        x_at = interp_sensitivity_at_obs(x_k, t_span, t_k)
        u_sk_at = interp_sensitivity_at_obs(u_sk, t_span, t_k)
        if any_s:
            grad_s -= 2.0 * np.sum(res_k * x_at, axis=0)
        grad_scalar_K -= 2.0 * np.sum(res_k * s_factor * u_sk_at)

        for b in range(n_biomarkers):
            u_b_at = interp_sensitivity_at_obs(u_kappa[b], t_span, t_k)
            grad_kappa[b] -= 2.0 * np.sum(res_k * s_factor * u_b_at)

    grad_kappa += lambda_kappa  # constant: penalty is lambda_kappa * sum(kappa) (kappa >= 0
                                # via bounds, so this is already L1 — not sign-dependent).

    grad_scalar_K += _lognorm_scalar_K_grad(
        scalar_K_safe, lambda_scalar, scalar_K_center
    )
    return loss, _pack_global_grad(grad_s, grad_kappa, grad_scalar_K, s_mask_bool)


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
    fit_s: bool = True,
    ode_method: str = "LSODA",
    obs_weight: np.ndarray = None,
    s_mask=None,
) -> tuple:
    """
    Same loss as theta_s_loss_multi; gradients via RK4 sensitivities on t_span.
    """
    s, kappa, scalar_K, x_list, _x_pred, residuals = _solve_and_residuals(
        params, t_obs, x_obs, K, t_span, cluster_f, observation_assignments,
        fit_s=fit_s,
        ode_method=ode_method,
        obs_weight=obs_weight,
        s_mask=s_mask,
    )
    n_biomarkers = x_obs.shape[1]
    s_mask_bool = resolve_s_mask(n_biomarkers, fit_s, s_mask)
    any_s = bool(np.any(s_mask_bool))
    scalar_K_safe = max(scalar_K, SCALAR_K_MIN)
    lognorm_penalty = _lognorm_scalar_K_penalty(
        scalar_K_safe, lambda_scalar, scalar_K_center
    )
    s_penalty = _s_ridge_penalty(s, lambda_s, s_mask_bool)
    loss = (
        np.sum(residuals ** 2)
        + s_penalty
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

    grad_s = 2.0 * lambda_s * s.copy() if any_s else None
    grad_kappa = np.zeros(n_biomarkers)
    grad_scalar_K = 0.0
    s_factor = s[None, :]

    for k, (x_k, u_sk, u_kappa) in enumerate(sens_records):
        mask_k = observation_assignments == k
        if not np.any(mask_k):
            continue
        t_k = np.clip(t_obs[mask_k], t_span[0], t_span[-1])
        res_k = residuals[mask_k]

        x_at = interp_sensitivity_at_obs(x_k, t_span, t_k)
        u_sk_at = interp_sensitivity_at_obs(u_sk, t_span, t_k)
        if any_s:
            grad_s -= 2.0 * np.sum(res_k * x_at, axis=0)
        grad_scalar_K -= 2.0 * np.sum(res_k * s_factor * u_sk_at)

        for b in range(n_biomarkers):
            u_b_at = interp_sensitivity_at_obs(u_kappa[b], t_span, t_k)
            grad_kappa[b] -= 2.0 * np.sum(res_k * s_factor * u_b_at)

    grad_kappa += lambda_kappa  # constant: penalty is lambda_kappa * sum(kappa) (kappa >= 0
                                # via bounds, so this is already L1 — not sign-dependent).

    grad_scalar_K += _lognorm_scalar_K_grad(
        scalar_K_safe, lambda_scalar, scalar_K_center
    )
    return loss, _pack_global_grad(grad_s, grad_kappa, grad_scalar_K, s_mask_bool)


def theta_s_loss_jac_approx_multi(
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
    fit_s: bool = True,
    ode_method: str = "LSODA",
    obs_weight: np.ndarray = None,
    s_mask=None,
) -> tuple:
    """
    Cheap analytic-approx gradient for (s, kappa, scalar_K), mirroring the
    cumulative_simpson trick already used for cluster f in
    optimizer_theta_cluster.py.

    Approximation: freezes x(t) at its unperturbed trajectory and keeps only
    the *direct* RHS term for each parameter, dropping the feedback term
    -diag(dx/dparam)(K_eff x + f):
      - s:        exact (s doesn't enter the ODE, only scales output).
      - kappa_i:  diagonal/self-region only (kappa_i enters region i's own
                  equation as (1-x_i)*kappa_i*x_i; cross-region propagation
                  of the perturbation is dropped — same approximation
                  character as the f-cluster trick).
      - scalar_K: full direct term (Kx already couples regions, so this
                  naturally captures cross-region effects at t; only the
                  feedback/recursive term is dropped).

    Cheaper than finite differences: reuses the already-solved trajectory,
    no extra ODE solves, just matrix multiplies + 1-D cumulative_simpson
    quadratures.

    ``obs_weight`` is handled inside ``_solve_and_residuals``, so masked
    entries contribute zero to both loss and gradient.
    """
    s, kappa, scalar_K, x_list, _x_pred, residuals = _solve_and_residuals(
        params, t_obs, x_obs, K, t_span, cluster_f, observation_assignments,
        fit_s=fit_s,
        ode_method=ode_method,
        obs_weight=obs_weight,
        s_mask=s_mask,
    )
    n_biomarkers = x_obs.shape[1]
    s_mask_bool = resolve_s_mask(n_biomarkers, fit_s, s_mask)
    any_s = bool(np.any(s_mask_bool))
    scalar_K_safe = max(scalar_K, SCALAR_K_MIN)
    lognorm_penalty = _lognorm_scalar_K_penalty(
        scalar_K_safe, lambda_scalar, scalar_K_center
    )
    s_penalty = _s_ridge_penalty(s, lambda_s, s_mask_bool)
    loss = (
        np.sum(residuals ** 2)
        + s_penalty
        + lognorm_penalty
        + lambda_kappa * np.sum(kappa)
    )

    grad_s = 2.0 * lambda_s * s.copy() if any_s else None
    grad_kappa = np.zeros(n_biomarkers)
    grad_scalar_K = 0.0
    s_factor = s[None, :]

    for k, x_k in enumerate(x_list):
        mask_k = observation_assignments == k
        if not np.any(mask_k):
            continue
        t_k = np.clip(t_obs[mask_k], t_span[0], t_span[-1])
        res_k = residuals[mask_k]

        if any_s:
            x_at = np.vstack(
                [np.interp(t_k, t_span, x_k[i]) for i in range(n_biomarkers)]
            ).T  # (n_obs_k, p); exact: d(x_pred_i)/d(s_i) = x_i(t)
            grad_s -= 2.0 * np.sum(res_k * x_at, axis=0)

        Kx_k = K @ x_k  # (p, T)

        u_scalarK_at = np.zeros((len(t_k), n_biomarkers))
        u_kappa_at = np.zeros((len(t_k), n_biomarkers))
        for i in range(n_biomarkers):
            cum_scalarK = cumulative_simpson(
                (1 - x_k[i]) * Kx_k[i], x=t_span, initial=0
            )
            u_scalarK_at[:, i] = CubicSpline(
                t_span, cum_scalarK, extrapolate=False
            )(t_k)

            cum_kappa = cumulative_simpson(
                (1 - x_k[i]) * x_k[i], x=t_span, initial=0
            )
            u_kappa_at[:, i] = CubicSpline(
                t_span, cum_kappa, extrapolate=False
            )(t_k)

        grad_scalar_K -= 2.0 * np.sum(res_k * s_factor * u_scalarK_at)
        # diagonal only: column i of u_kappa_at pairs with residual column i
        grad_kappa -= 2.0 * np.sum(res_k * s_factor * u_kappa_at, axis=0)

    grad_kappa += lambda_kappa  # constant: penalty is lambda_kappa * sum(kappa) (kappa >= 0
                                # via bounds, so this is already L1 — not sign-dependent).
    grad_scalar_K += _lognorm_scalar_K_grad(
        scalar_K_safe, lambda_scalar, scalar_K_center
    )

    return loss, _pack_global_grad(grad_s, grad_kappa, grad_scalar_K, s_mask_bool)


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
    fit_s: bool = True,
    method: str = "lbfgs_approx",
    strict_tol: bool = False,
    ode_method: str = "LSODA",
    obs_weight: np.ndarray = None,
    s_mask=None,
    s_lower: np.ndarray = None,
    s_upper=None,
) -> tuple:
    """
    Optimizes global (s, kappa, scalar_K) using assignment-aware subtype trajectories.

    s_lower:
        Optional (n_biomarkers,) lower bound on each free s entry. ``None``
        (default) means 0 for every entry.

    fit_s:
        If True (default), optimize per-biomarker s (subject to ``s_mask``).
        If False, fix every s entry at 1 and optimize only (kappa, scalar_K).

    s_mask:
        Optional boolean vector of length n_biomarkers. True = optimize that
        s entry; False = hold it at 1. Ignored when ``fit_s`` is False.
        ``None`` (default) leaves every entry free when ``fit_s`` is True.

    method:
        'lbfgs_approx' — L-BFGS-B, jac=True, cheap analytic-approx gradient
                         (cumulative_simpson direct-term trick; drops the
                         sensitivity feedback term and, for kappa, cross-region
                         coupling — see theta_s_loss_jac_approx_multi).
        'lbfgs_exact'  — L-BFGS-B, jac=True, exact sensitivities solved with LSODA.
        'lbfgs_rk4'    — L-BFGS-B, jac=True, sensitivities via RK4 on t_span.
        'nelder_mead'  — derivative-free, loss only.

    Returns
    -------
    s_fit, kappa_fit, scalar_K_fit : fitted globals.
    trajectories : list of per-subtype (n_biomarkers, len(t_span)) arrays,
        solved at the converged parameters. Callers can pass these straight
        into the E-step to avoid re-solving the identical ODE system, but
        only while ``cluster_f`` is unchanged.
    """
    t_pred = dt_obs + beta_pred[ids]
    n_biomarkers = X_obs.shape[1]
    s_mask_bool = resolve_s_mask(n_biomarkers, fit_s, s_mask)
    n_free_s = int(np.count_nonzero(s_mask_bool))

    if scalar_K_guess is None:
        scalar_K_guess = 1.0
    if kappa_guess is None:
        kappa_guess = np.zeros(K.shape[0])
    if s_guess is None:
        s_guess = np.ones(n_biomarkers)
    else:
        s_guess = np.asarray(s_guess, dtype=float).ravel().copy()
        if s_guess.shape != (n_biomarkers,):
            raise ValueError(
                f"s_guess must have shape ({n_biomarkers},), got {s_guess.shape}"
            )
    s_guess[~s_mask_bool] = 1.0

    if n_free_s:
        initial_params = np.concatenate(
            [s_guess[s_mask_bool], kappa_guess, [scalar_K_guess]]
        )
        bounds = (
            [tuple(b) for b in _s_bounds(s_lower, n_biomarkers, s_upper)[s_mask_bool].tolist()]
            + [(0.0, np.inf)] * n_biomarkers
            + [(0.0, np.inf)]
        )
    else:
        initial_params = np.concatenate([kappa_guess, [scalar_K_guess]])
        bounds = [(0.0, np.inf)] * n_biomarkers + [(0.0, np.inf)]

    observation_assignments = assignments[ids]

    if method == "lbfgs_exact":
        loss_fn, scipy_method, use_jac = theta_s_loss_jac_exact_multi, "L-BFGS-B", True
    elif method == "lbfgs_rk4":
        loss_fn, scipy_method, use_jac = theta_s_loss_jac_rk4_multi, "L-BFGS-B", True
    elif method == "lbfgs_approx":
        loss_fn, scipy_method, use_jac = theta_s_loss_jac_approx_multi, "L-BFGS-B", True
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
        fit_s,
        ode_method,
        obs_weight,
        s_mask_bool,
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

    s_fit, kappa_fit, scalar_K_fit = _unpack_globals(
        result.x, n_biomarkers, s_mask_bool
    )

    # Solve once at the converged point and hand the trajectories back so
    # callers (the E-step) don't have to re-solve the identical ODE system.
    x0 = np.zeros(n_biomarkers)
    trajectories = [
        solve_system(
            x0, np.ravel(cluster_f[z]), K, t_span, scalar_K_fit, kappa_fit,
            ode_method=ode_method,
        )
        for z in range(len(cluster_f))
    ]

    return s_fit, kappa_fit, scalar_K_fit, trajectories
