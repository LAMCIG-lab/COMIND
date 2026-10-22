"""
NumPy-facing wrappers for the JAX backend.

Every wrapper takes and returns NumPy, converts at the boundary, and drives
``scipy.optimize.minimize(method="L-BFGS-B", jac=True)`` with loss/gradient
evaluations from the module-level jitted functions in ``jax_core``. Follow
the recompile rules in the ``jax_core`` module docstring when editing.

Each wrapper accepts an optional ``packed=(PackedData, meta)`` keyword. Pass
the pair built once by ``jax_core.pack_data`` (``SubtypingEM`` does) to avoid
re-packing on every call and to keep shapes fixed across calls. Without it,
the stacked NumPy arrays are packed with the default padding.

Non-finite guard: if a loss or gradient is non-finite (e.g. RK4 instability
at very stiff parameters), the objective returns ``(1e30, zeros)`` and one
``RuntimeWarning`` is emitted per wrapper call. The total count is available
from ``nonfinite_events()``.
"""

import warnings
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from . import jax_core as jc
from .optimizer_beta import _beta_bounds
from .optimizer_theta_globals import SCALAR_K_CENTER, _s_bounds, resolve_s_mask

_STRICT_OPTIONS = {"maxiter": 200, "ftol": 2.2e-10, "gtol": 1e-6}
_NONFINITE_LOSS = 1e30

_NONFINITE_EVENTS = 0


def nonfinite_events() -> int:
    """
    Total number of non-finite loss/gradient evaluations replaced by the guard.

    Returns
    -------
    int
    """
    return _NONFINITE_EVENTS


def reset_nonfinite_events() -> None:
    """Reset the module-level non-finite event counter to 0."""
    global _NONFINITE_EVENTS
    _NONFINITE_EVENTS = 0


class _Guard:
    """Per-wrapper-call bookkeeping for the non-finite guard."""

    def __init__(self, name: str):
        self.name = name
        self.count = 0

    def check(self, loss: float, grad: np.ndarray) -> Tuple[float, np.ndarray]:
        global _NONFINITE_EVENTS
        if np.isfinite(loss) and np.all(np.isfinite(grad)):
            return loss, grad
        _NONFINITE_EVENTS += 1
        self.count += 1
        if self.count == 1:
            warnings.warn(
                f"{self.name}: non-finite loss or gradient (loss={loss}); "
                f"returning {_NONFINITE_LOSS:g} with a zero gradient.",
                RuntimeWarning,
                stacklevel=3,
            )
        return _NONFINITE_LOSS, np.zeros_like(grad)


def _lbfgs_options(strict_tol: bool, base: Optional[dict] = None) -> Optional[dict]:
    opts = dict(base or {})
    if strict_tol:
        opts.update(_STRICT_OPTIONS)
    return opts or None


def _get_packed(packed, X_obs, obs_weight, dt, ids, cog, n_patients):
    if packed is not None:
        return packed
    if cog is None:
        cog = np.zeros((len(dt), 1))
    return jc.pack_data(X_obs, obs_weight, dt, ids, cog, n_patients)


def _n_patients_from_ids(ids: np.ndarray) -> int:
    return len(np.unique(ids))


def _stack_f(cluster_f: Sequence[np.ndarray]) -> np.ndarray:
    return np.stack([np.ravel(np.asarray(f, dtype=float)) for f in cluster_f])


def _stack_cog(cog_a, cog_b, n_subtypes: int, n_cog: int) -> Tuple[np.ndarray, np.ndarray]:
    """Per-subtype (Z, C) slopes and (Z,) intercepts from list or shared values."""
    if cog_a is None:
        A = np.zeros((n_subtypes, n_cog))
    elif isinstance(cog_a, list):
        A = np.stack([np.ravel(np.asarray(a, dtype=float)) for a in cog_a])
    else:
        A = np.tile(np.ravel(np.asarray(cog_a, dtype=float)), (n_subtypes, 1))
    if cog_b is None:
        b = np.zeros(n_subtypes)
    elif isinstance(cog_b, list):
        b = np.array([float(np.asarray(v).ravel()[0]) for v in cog_b])
    else:
        b = np.full(n_subtypes, float(np.asarray(cog_b).ravel()[0]))
    return A, b


def _grid(t_span: np.ndarray) -> Tuple[object, int]:
    _, h, n_steps = jc.grid_from_t_span(t_span)
    return jc.f64(h), n_steps


def trajectories_to_jax(trajectories) -> object:
    """
    Convert per-subtype trajectories to a (Z, T, B) float64 JAX array.

    Parameters
    ----------
    trajectories : list of np.ndarray or jax.Array
        Either a list of (B, T) arrays (NumPy convention) or a (Z, T, B)
        JAX/NumPy array (``_traj_jit`` output), which is passed through.

    Returns
    -------
    jax.Array
    """
    if isinstance(trajectories, (list, tuple)):
        return jc.f64(np.stack([np.asarray(x, dtype=float).T for x in trajectories]))
    return jc.f64(trajectories)


def trajectories_to_numpy(traj) -> List[np.ndarray]:
    """
    Convert a (Z, T, B) trajectory array to a list of (B, T) NumPy arrays.

    Parameters
    ----------
    traj : jax.Array
        Output of ``_traj_jit``.

    Returns
    -------
    list of np.ndarray
    """
    arr = np.asarray(traj, dtype=np.float64)
    return [np.ascontiguousarray(arr[z].T) for z in range(arr.shape[0])]


def solve_trajectories_jax(
    cluster_f: Sequence[np.ndarray],
    kappa: np.ndarray,
    scalar_K: float,
    K: np.ndarray,
    t_span: np.ndarray,
):
    """
    Solve every subtype trajectory with ``_traj_jit``.

    Parameters
    ----------
    cluster_f : sequence of np.ndarray
        Per-subtype forcing, each (B,).
    kappa : np.ndarray
        Shape (B,).
    scalar_K : float
    K : np.ndarray
        Shape (B, B).
    t_span : np.ndarray
        Uniform grid starting at 0.

    Returns
    -------
    jax.Array
        Trajectories of shape (Z, T, B).
    """
    h, n_steps = _grid(t_span)
    return jc._traj_jit(
        jc.f64(_stack_f(cluster_f)),
        jc.f64(np.ravel(kappa)),
        jc.f64(scalar_K),
        jc.f64(K),
        h,
        n_steps=n_steps,
    )


def fit_theta_globals_jax(
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
    method: str = "jax",
    strict_tol: bool = False,
    ode_method: str = "LSODA",
    obs_weight: np.ndarray = None,
    s_mask=None,
    s_lower: np.ndarray = None,
    *,
    packed: Optional[tuple] = None,
    info: Optional[dict] = None,
) -> tuple:
    """
    JAX version of ``fit_theta_globals``: fit (s, kappa, scalar_K) with exact gradients.

    Parameters
    ----------
    X_obs, dt_obs, ids, K, t_span, cluster_f, assignments, beta_pred,
    s_guess, scalar_K_guess, kappa_guess, lambda_s, lambda_scalar,
    lambda_kappa, scalar_K_center, fit_s, strict_tol, obs_weight, s_mask, s_lower
        As in ``fit_theta_globals``. ``s_mask`` False entries (and every
        entry when ``fit_s`` is False) are held at 1 with bounds (1, 1).
    method, ode_method
        Accepted for signature compatibility and ignored.
    packed : tuple, optional
        ``(PackedData, meta)`` from ``jax_core.pack_data``.
    info : dict, optional
        If given, filled with the scipy result fields ``nfev``, ``nit``,
        ``message`` and the number of non-finite evaluations.

    Returns
    -------
    s_fit, kappa_fit, scalar_K_fit
        Fitted globals (NumPy arrays and a float).
    trajectories : list of np.ndarray
        Per-subtype (B, T) trajectories solved with ``_traj_jit`` at the
        converged parameters. Valid for the E-step only while ``cluster_f``
        is unchanged.
    """
    del method, ode_method
    n_biomarkers = X_obs.shape[1]
    n_patients = len(assignments)
    data, meta = _get_packed(packed, X_obs, obs_weight, dt_obs, ids, None, n_patients)
    P = meta["P"]
    h, n_steps = _grid(t_span)

    s_mask_bool = resolve_s_mask(n_biomarkers, fit_s, s_mask)
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

    x0 = np.concatenate(
        [s_guess, np.asarray(kappa_guess, dtype=float).ravel(), [float(scalar_K_guess)]]
    )
    bounds = (
        [
            tuple(b) if free else (1.0, 1.0)
            for b, free in zip(_s_bounds(s_lower, n_biomarkers).tolist(), s_mask_bool)
        ]
        + [(0.0, np.inf)] * n_biomarkers
        + [(0.0, np.inf)]
    )

    F = jc.f64(_stack_f(cluster_f))
    beta_full = jc.pad_beta(beta_pred, P)
    assign_full = jc.pad_assignments(assignments, P)
    K_j = jc.f64(K)
    s_free = jc.f64(s_mask_bool.astype(float))
    hyp = jc.make_hyp(
        lambda_s=lambda_s,
        lambda_scalar=lambda_scalar,
        scalar_K_center=scalar_K_center,
        lambda_kappa=lambda_kappa,
    )
    guard = _Guard("fit_theta_globals_jax")
    B = n_biomarkers

    def objective(params):
        loss, (gs, gk, gK) = jc._globals_vg(
            jc.f64(params[:B]),
            jc.f64(params[B:2 * B]),
            jc.f64(params[-1]),
            F,
            beta_full,
            assign_full,
            data,
            K_j,
            hyp,
            h,
            n_steps=n_steps,
            s_free=s_free,
        )
        grad = np.concatenate(
            [np.asarray(gs), np.asarray(gk), np.atleast_1d(np.asarray(gK))]
        ).astype(np.float64)
        return guard.check(float(loss), grad)

    result = minimize(
        objective,
        x0=x0,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options=_lbfgs_options(strict_tol),
    )

    s_fit = np.asarray(result.x[:B], dtype=float).copy()
    s_fit[~s_mask_bool] = 1.0
    kappa_fit = np.asarray(result.x[B:2 * B], dtype=float).copy()
    scalar_K_fit = float(result.x[-1])

    traj = jc._traj_jit(
        F, jc.f64(kappa_fit), jc.f64(scalar_K_fit), K_j, h, n_steps=n_steps
    )
    if info is not None:
        info.update(nfev=result.nfev, nit=result.nit, message=str(result.message),
                    nonfinite=guard.count)
    return s_fit, kappa_fit, scalar_K_fit, trajectories_to_numpy(traj)


def fit_theta_globals_intercept_jax(
    X_obs: np.ndarray,
    dt_obs: np.ndarray,
    ids: np.ndarray,
    K: np.ndarray,
    t_span: np.ndarray,
    cluster_f: list,
    assignments: np.ndarray,
    beta_pred: np.ndarray,
    s_guess: np.ndarray,
    scalar_K_guess: float,
    kappa_guess: np.ndarray,
    c_guess: np.ndarray,
    c_bounds: np.ndarray,
    c_center: np.ndarray = None,
    lambda_c: float = 0.0,
    lambda_scalar: float = 0.0,
    lambda_kappa: float = 0.0,
    scalar_K_center: float = SCALAR_K_CENTER,
    fit_s: bool = True,
    strict_tol: bool = False,
    obs_weight: np.ndarray = None,
    s_mask=None,
    s_lower: np.ndarray = None,
    *,
    packed: Optional[tuple] = None,
    info: Optional[dict] = None,
) -> tuple:
    """
    ``fit_theta_globals_jax`` plus a per-biomarker intercept ``c``.

    The prediction is ``c + s * x``. ``packed`` must hold the UNSHIFTED data.

    Parameters
    ----------
    c_guess : (B,)
        Starting intercept.
    c_bounds : (B, 2)
        Lower and upper bound per biomarker. Equal bounds hold that entry fixed
        (use this for biomarkers whose zero is already meaningful).
    c_center, lambda_c
        Optional ridge ``lambda_c * sum((c - c_center)**2)``. ``c_center=None``
        means zeros.
    Other parameters as in ``fit_theta_globals_jax`` (``lambda_s`` is 0).

    Returns
    -------
    s_fit, kappa_fit, scalar_K_fit, c_fit, trajectories
        ``trajectories`` as in ``fit_theta_globals_jax``.
    """
    n_biomarkers = X_obs.shape[1]
    B = n_biomarkers
    n_patients = len(assignments)
    data, meta = _get_packed(packed, X_obs, obs_weight, dt_obs, ids, None, n_patients)
    P = meta["P"]
    h, n_steps = _grid(t_span)

    s_mask_bool = resolve_s_mask(n_biomarkers, fit_s, s_mask)
    s_guess = np.asarray(s_guess, dtype=float).ravel().copy()
    s_guess[~s_mask_bool] = 1.0
    c_guess = np.asarray(c_guess, dtype=float).ravel()
    c_bounds = np.asarray(c_bounds, dtype=float).reshape(B, 2)
    c_guess = np.clip(c_guess, c_bounds[:, 0], c_bounds[:, 1])
    c_center = np.zeros(B) if c_center is None else np.asarray(c_center, dtype=float).ravel()

    x0 = np.concatenate([
        s_guess, np.asarray(kappa_guess, dtype=float).ravel(), [float(scalar_K_guess)], c_guess,
    ])
    bounds = (
        [
            tuple(b) if free else (1.0, 1.0)
            for b, free in zip(_s_bounds(s_lower, n_biomarkers).tolist(), s_mask_bool)
        ]
        + [(0.0, np.inf)] * B
        + [(0.0, np.inf)]
        + [(float(lo), float(hi)) for lo, hi in c_bounds]
    )

    F = jc.f64(_stack_f(cluster_f))
    beta_full = jc.pad_beta(beta_pred, P)
    assign_full = jc.pad_assignments(assignments, P)
    K_j = jc.f64(K)
    s_free = jc.f64(s_mask_bool.astype(float))
    c_center_j = jc.f64(c_center)
    hyp = jc.make_hyp(
        lambda_scalar=lambda_scalar,
        scalar_K_center=scalar_K_center,
        lambda_kappa=lambda_kappa,
        lambda_c=lambda_c,
    )
    guard = _Guard("fit_theta_globals_intercept_jax")

    def objective(params):
        loss, (gs, gk, gK, gc) = jc._globals_c_vg(
            jc.f64(params[:B]),
            jc.f64(params[B:2 * B]),
            jc.f64(params[2 * B]),
            jc.f64(params[2 * B + 1:]),
            F,
            beta_full,
            assign_full,
            data,
            K_j,
            hyp,
            h,
            c_center_j,
            n_steps=n_steps,
            s_free=s_free,
        )
        grad = np.concatenate(
            [np.asarray(gs), np.asarray(gk), np.atleast_1d(np.asarray(gK)), np.asarray(gc)]
        ).astype(np.float64)
        return guard.check(float(loss), grad)

    result = minimize(
        objective, x0=x0, method="L-BFGS-B", jac=True, bounds=bounds,
        options=_lbfgs_options(strict_tol),
    )
    s_fit = np.asarray(result.x[:B], dtype=float).copy()
    s_fit[~s_mask_bool] = 1.0
    kappa_fit = np.asarray(result.x[B:2 * B], dtype=float).copy()
    scalar_K_fit = float(result.x[2 * B])
    c_fit = np.asarray(result.x[2 * B + 1:], dtype=float).copy()

    traj = jc._traj_jit(F, jc.f64(kappa_fit), jc.f64(scalar_K_fit), K_j, h, n_steps=n_steps)
    if info is not None:
        info.update(nfev=result.nfev, nit=result.nit, message=str(result.message),
                    nonfinite=guard.count)
    return s_fit, kappa_fit, scalar_K_fit, c_fit, trajectories_to_numpy(traj)


def fit_theta_cluster_all_jax(
    X_obs: np.ndarray,
    dt_obs: np.ndarray,
    ids: np.ndarray,
    K: np.ndarray,
    t_span: np.ndarray,
    *,
    cluster_f: list,
    assignments: np.ndarray,
    s: np.ndarray,
    scalar_K: float,
    lambda_f: float,
    beta_pred: np.ndarray,
    kappa: np.ndarray = None,
    strict_tol: bool = False,
    obs_weight: np.ndarray = None,
    packed: Optional[tuple] = None,
    info: Optional[dict] = None,
) -> List[np.ndarray]:
    """
    Fit every subtype's forcing ``f_z`` jointly (separable; see ``jax_core.cluster_loss``).

    Parameters
    ----------
    X_obs, dt_obs, ids : np.ndarray
        Stacked observations for all patients (not a per-subtype subset).
    K : np.ndarray
        Connectivity, shape (B, B).
    t_span : np.ndarray
        Uniform grid starting at 0.
    cluster_f : list of np.ndarray
        Current per-subtype forcing; used as the initial guess.
    assignments : np.ndarray
        Subtype per patient, shape (n_patients,).
    s, scalar_K, kappa
        Fixed globals. ``kappa=None`` means zeros.
    lambda_f : float
        L1 weight on f (f >= 0 via bounds).
    beta_pred : np.ndarray
        Betas per patient, shape (n_patients,).
    strict_tol : bool
        Use the strict L-BFGS-B options, as in ``fit_theta_cluster``.
    obs_weight : np.ndarray, optional
        (n_rows, B) observation weight.
    packed : tuple, optional
        ``(PackedData, meta)`` from ``jax_core.pack_data``.
    info : dict, optional
        Filled with ``nfev``, ``nit``, ``message`` and ``nonfinite``.

    Returns
    -------
    list of np.ndarray
        ``Z`` fitted forcing vectors. Subtypes with no assigned patients keep
        their input ``f_z`` exactly (bounds ``(f, f)``), as the NumPy path
        skips them.
    """
    n_biomarkers = X_obs.shape[1]
    n_patients = len(assignments)
    data, meta = _get_packed(packed, X_obs, obs_weight, dt_obs, ids, None, n_patients)
    P = meta["P"]
    h, n_steps = _grid(t_span)

    F0 = _stack_f(cluster_f)
    Z = F0.shape[0]
    assignments = np.asarray(assignments)
    empty = np.array([not np.any(assignments == z) for z in range(Z)])

    bounds = []
    for z in range(Z):
        if empty[z]:
            bounds.extend((float(v), float(v)) for v in F0[z])
        else:
            bounds.extend([(0.0, np.inf)] * n_biomarkers)

    if kappa is None:
        kappa = np.zeros(n_biomarkers)
    s_j = jc.f64(np.ravel(s))
    kappa_j = jc.f64(np.ravel(kappa))
    scalar_K_j = jc.f64(scalar_K)
    beta_full = jc.pad_beta(beta_pred, P)
    assign_full = jc.pad_assignments(assignments, P)
    K_j = jc.f64(K)
    hyp = jc.make_hyp(lambda_f=lambda_f)
    guard = _Guard("fit_theta_cluster_all_jax")

    def objective(params):
        loss, g = jc._cluster_vg(
            jc.f64(params.reshape(Z, n_biomarkers)),
            s_j,
            kappa_j,
            scalar_K_j,
            beta_full,
            assign_full,
            data,
            K_j,
            hyp,
            h,
            n_steps=n_steps,
        )
        return guard.check(float(loss), np.asarray(g, dtype=np.float64).ravel())

    result = minimize(
        objective,
        x0=F0.ravel(),
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options=_lbfgs_options(strict_tol),
    )

    F_fit = np.asarray(result.x, dtype=float).reshape(Z, n_biomarkers).copy()
    F_fit[empty] = F0[empty]
    if info is not None:
        info.update(nfev=result.nfev, nit=result.nit, message=str(result.message),
                    nonfinite=guard.count)
    return [F_fit[z].copy() for z in range(Z)]


def estimate_beta_jax(
    beta_all: np.ndarray,
    X_obs: np.ndarray,
    dt: np.ndarray,
    ids: np.ndarray,
    cog: np.ndarray,
    t_span: np.ndarray,
    trajectories,
    s: np.ndarray,
    assignments: np.ndarray,
    cog_a,
    cog_b,
    lambda_cog: float = 0.0,
    lambda_jsd: float = 0.0,
    lambda_beta: float = 0.0,
    beta_mean: float = None,
    beta_var: float = None,
    t_max: float = 12.0,
    jsd_n_bins: int = None,
    jsd_bandwidth: float = None,
    jsd_value_range: tuple = None,
    strict_tol: bool = False,
    obs_weight: np.ndarray = None,
    beta_upper: np.ndarray = None,
    *,
    packed: Optional[tuple] = None,
    info: Optional[dict] = None,
    maxiter: Optional[int] = None,
) -> tuple:
    """
    JAX version of ``estimate_beta``: optimize all real betas jointly.

    Parameters
    ----------
    beta_all : np.ndarray
        Initial betas, shape (n_patients,).
    X_obs, dt, ids, cog : np.ndarray
        Stacked observations.
    t_span : np.ndarray
        Uniform grid starting at 0.
    trajectories : list of np.ndarray or jax.Array
        Precomputed trajectories at the current (f, kappa, scalar_K): a list
        of (B, T) arrays or the (Z, T, B) ``_traj_jit`` output. The ODE is not
        solved here.
    s : np.ndarray
        Output scale, shape (B,).
    assignments : np.ndarray
        Subtype per patient, shape (n_patients,).
    cog_a, cog_b
        Per-subtype lists, or shared values, as in ``estimate_beta``.
    lambda_cog, lambda_jsd, lambda_beta, beta_mean, beta_var, t_max
        As in ``estimate_beta``. The beta prior is active only when
        ``lambda_beta > 0`` and both ``beta_mean`` and ``beta_var`` are given.
    jsd_n_bins, jsd_bandwidth, jsd_value_range
        JSD settings. Defaults: range ``(0, t_max)``,
        ``n_bins = max(20, int((hi - lo) * 2))``, automatic bandwidth.
    strict_tol : bool
        Add the strict L-BFGS-B options on top of ``maxiter = 100``.
    obs_weight : np.ndarray, optional
        (n_rows, B) observation weight.
    beta_upper : np.ndarray, optional
        Per-patient upper bound, shape (n_patients,). ``None`` means ``t_max``
        for everyone (the original bounds).
    packed : tuple, optional
        ``(PackedData, meta)`` from ``jax_core.pack_data``.
    info : dict, optional
        Filled with ``nfev``, ``nit``, ``message`` and ``nonfinite``.
    maxiter : int, optional
        Overrides the L-BFGS-B ``maxiter`` after ``strict_tol`` is applied.
        ``None`` keeps 100 (200 with ``strict_tol``).

    Returns
    -------
    beta : np.ndarray
        Optimized betas, shape (n_patients,).
    reconstruction_lse : float
        Plain weighted SSE at the optimum (no penalties).
    """
    n_patients = len(assignments)
    data, meta = _get_packed(packed, X_obs, obs_weight, dt, ids, cog, n_patients)
    P = meta["P"]
    h, _ = _grid(t_span)

    traj = trajectories_to_jax(trajectories)
    Z = traj.shape[0]
    n_cog = data.cog.shape[1]
    A, b = _stack_cog(cog_a, cog_b, Z, n_cog)

    use_prior = lambda_beta > 0 and beta_mean is not None and beta_var is not None
    hyp = jc.make_hyp(
        lambda_cog=lambda_cog,
        lambda_beta=lambda_beta if use_prior else 0.0,
        beta_mean=beta_mean if use_prior else 0.0,
        beta_var=beta_var if use_prior else 1.0,
        lambda_jsd=lambda_jsd,
    )
    lo, hi = jsd_value_range if jsd_value_range is not None else (0.0, t_max)
    n_bins = int(jsd_n_bins) if jsd_n_bins is not None else max(20, int((hi - lo) * 2))
    auto_bw = jsd_bandwidth is None
    bandwidth = jc.f64(1.0 if auto_bw else jsd_bandwidth)
    use_jsd = bool(lambda_jsd > 0)

    s_j = jc.f64(np.ravel(s))
    assign_full = jc.pad_assignments(assignments, P)
    A_j, b_j = jc.f64(A), jc.f64(b)
    lo_j, hi_j = jc.f64(lo), jc.f64(hi)
    guard = _Guard("estimate_beta_jax")
    pad = np.zeros(P - n_patients)

    def objective(beta_vec):
        loss, g = jc._beta_vg(
            jc.f64(np.concatenate([beta_vec, pad])),
            traj,
            s_j,
            assign_full,
            A_j,
            b_j,
            data,
            hyp,
            h,
            lo_j,
            hi_j,
            bandwidth,
            n_bins=n_bins,
            use_jsd=use_jsd,
            auto_bandwidth=auto_bw,
        )
        grad = np.asarray(g, dtype=np.float64)[:n_patients]
        return guard.check(float(loss), grad)

    options = _lbfgs_options(strict_tol, {"maxiter": 100})
    if maxiter is not None:
        options["maxiter"] = int(maxiter)
    result = minimize(
        objective,
        x0=np.asarray(beta_all, dtype=float).copy(),
        method="L-BFGS-B",
        jac=True,
        bounds=_beta_bounds(n_patients, t_max, beta_upper),
        options=options,
    )
    beta_opt = np.asarray(result.x, dtype=float)

    lse = float(
        jc._recon_sse_jit(traj, s_j, jc.pad_beta(beta_opt, P), assign_full, data, h)
    )
    if info is not None:
        info.update(nfev=result.nfev, nit=result.nit, message=str(result.message),
                    nonfinite=guard.count)
    return beta_opt, lse


def _sse_matrix_np(
    packed,
    beta: np.ndarray,
    X_preds,
    s: np.ndarray,
    t_span: np.ndarray,
    cluster_cog_a,
    cluster_cog_b,
    lambda_cog: float,
    n_patients: int,
) -> np.ndarray:
    data, meta = packed
    h, _ = _grid(t_span)
    traj = trajectories_to_jax(X_preds)
    A, b = _stack_cog(cluster_cog_a, cluster_cog_b, traj.shape[0], data.cog.shape[1])
    sse = jc._sse_matrix_jit(
        traj,
        jc.f64(np.ravel(s)),
        jc.pad_beta(beta, meta["P"]),
        data,
        jc.f64(A),
        jc.f64(b),
        jc.f64(lambda_cog),
        h,
    )
    return np.asarray(sse, dtype=np.float64)[:n_patients]


def update_assignments_hard_jax(
    X_obs: np.ndarray,
    dt: np.ndarray,
    ids: np.ndarray,
    cog: np.ndarray,
    beta: np.ndarray,
    cluster_f: list,
    scalar_K: float,
    kappa: np.ndarray,
    s: np.ndarray,
    K: np.ndarray,
    t_span: np.ndarray,
    cluster_cog_a,
    cluster_cog_b,
    lambda_cog: float,
    ode_method: str = "LSODA",
    obs_weight: np.ndarray = None,
    X_preds=None,
    *,
    packed: Optional[tuple] = None,
) -> np.ndarray:
    """
    Hard E-step: argmin of ``sse_matrix_jax`` over subtypes for each real patient.

    Parameters
    ----------
    X_obs, dt, ids, cog, beta, cluster_f, scalar_K, kappa, s, K, t_span,
    cluster_cog_a, cluster_cog_b, lambda_cog, obs_weight
        As in ``assignments.update_assignments_hard``.
    ode_method
        Ignored.
    X_preds : list of np.ndarray or jax.Array, optional
        Precomputed trajectories (only valid while ``cluster_f``, ``scalar_K``
        and ``kappa`` are unchanged). ``None`` solves with ``_traj_jit``.
    packed : tuple, optional
        ``(PackedData, meta)`` from ``jax_core.pack_data``.

    Returns
    -------
    np.ndarray
        Integer assignments, shape (n_patients,).
    """
    del ode_method
    n_patients = len(beta)
    packed = _get_packed(packed, X_obs, obs_weight, dt, ids, cog, n_patients)
    if X_preds is None:
        X_preds = solve_trajectories_jax(cluster_f, kappa, scalar_K, K, t_span)
    sse = _sse_matrix_np(
        packed, beta, X_preds, s, t_span, cluster_cog_a, cluster_cog_b,
        lambda_cog, n_patients,
    )
    return np.argmin(sse, axis=1).astype(int)


def _pad_upper(beta_upper: Optional[np.ndarray], n_patients: int, P: int, t_max: float):
    out = np.full(P, float(t_max))
    if beta_upper is None:
        out[:n_patients] = np.inf
    else:
        out[:n_patients] = np.asarray(beta_upper, dtype=float).ravel()
    return jc.f64(out)


def _grid_candidates(beta_grid: np.ndarray, P: int, beta: Optional[np.ndarray] = None):
    """(G, P) grid table, with the padded current β prepended when ``beta`` is given."""
    beta_grid = np.asarray(beta_grid, dtype=float).ravel()
    rows = np.repeat(beta_grid[:, None], P, axis=1)
    if beta is not None:
        rows = np.vstack([np.asarray(jc.pad_beta(beta, P))[None, :], rows])
    return jc.f64(rows)


def update_assignments_joint_grid_jax(
    X_obs: np.ndarray,
    dt: np.ndarray,
    ids: np.ndarray,
    cog: np.ndarray,
    beta: np.ndarray,
    cluster_f: list,
    scalar_K: float,
    kappa: np.ndarray,
    s: np.ndarray,
    K: np.ndarray,
    t_span: np.ndarray,
    cluster_cog_a,
    cluster_cog_b,
    lambda_cog: float,
    beta_grid: np.ndarray,
    beta_upper: np.ndarray = None,
    ode_method: str = "LSODA",
    obs_weight: np.ndarray = None,
    X_preds=None,
    *,
    packed: Optional[tuple] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    JAX version of ``assignments.update_assignments_joint_grid``.

    Candidates are the current β (index 0) followed by ``beta_grid``; the
    joint argmin over (candidate, subtype) comes from
    ``jax_core.joint_grid_select``. The same guarantee holds: each patient's
    chosen loss is <= their loss at the current (β, z), since the current β
    is never masked (effective bound ``max(beta_upper[i], beta[i])``). The
    full objective may still rise, because the search ignores the JSD term
    and the β prior.

    Parameters
    ----------
    X_obs, dt, ids, cog, beta, cluster_f, scalar_K, kappa, s, K, t_span,
    cluster_cog_a, cluster_cog_b, lambda_cog, beta_grid, beta_upper, obs_weight
        As in ``assignments.update_assignments_joint_grid``.
    ode_method
        Ignored.
    X_preds : list of np.ndarray or jax.Array, optional
        Precomputed trajectories. ``None`` solves with ``_traj_jit``.
    packed : tuple, optional
        ``(PackedData, meta)`` from ``jax_core.pack_data``.

    Returns
    -------
    assignments : np.ndarray
        Shape (n_patients,).
    beta_start : np.ndarray
        Chosen candidate β per patient, shape (n_patients,).
    """
    del ode_method
    n_patients = len(beta)
    data, meta = _get_packed(packed, X_obs, obs_weight, dt, ids, cog, n_patients)
    P = meta["P"]
    h, _ = _grid(t_span)
    if X_preds is None:
        X_preds = solve_trajectories_jax(cluster_f, kappa, scalar_K, K, t_span)
    traj = trajectories_to_jax(X_preds)
    A, b = _stack_cog(cluster_cog_a, cluster_cog_b, traj.shape[0], data.cog.shape[1])

    beta = np.asarray(beta, dtype=float)
    upper = None if beta_upper is None else np.maximum(np.asarray(beta_upper, float), beta)
    best_beta, best_z = jc._joint_grid_select_jit(
        traj,
        jc.f64(np.ravel(s)),
        _grid_candidates(beta_grid, P, beta),
        data,
        jc.f64(A),
        jc.f64(b),
        jc.f64(lambda_cog),
        _pad_upper(upper, n_patients, P, t_span[-1]),
        h,
    )
    return (
        np.asarray(best_z)[:n_patients].astype(int),
        np.asarray(best_beta, dtype=np.float64)[:n_patients].copy(),
    )


def update_assignments_jitter_jax(
    X_obs: np.ndarray,
    dt: np.ndarray,
    ids: np.ndarray,
    cog: np.ndarray,
    beta: np.ndarray,
    cluster_f: list,
    scalar_K: float,
    kappa: np.ndarray,
    s: np.ndarray,
    K: np.ndarray,
    t_span: np.ndarray,
    cluster_cog_a,
    cluster_cog_b,
    lambda_cog: float,
    temperature: float = 1.0,
    rng: np.random.Generator = None,
    ode_method: str = "LSODA",
    obs_weight: np.ndarray = None,
    X_preds=None,
    *,
    packed: Optional[tuple] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample assignments from ``softmax(-SSE / temperature)``.

    Uses the reconstruction-only SSE (``lambda_cog = 0``), matching
    ``assignments.update_assignments_jitter``, then the same per-patient
    NumPy softmax and ``rng.choice`` sequence.

    Parameters
    ----------
    X_obs, dt, ids, cog, beta, cluster_f, scalar_K, kappa, s, K, t_span,
    cluster_cog_a, cluster_cog_b, lambda_cog, temperature, rng, obs_weight
        As in ``assignments.update_assignments_jitter``. ``lambda_cog`` is
        ignored.
    ode_method
        Ignored.
    X_preds : list of np.ndarray or jax.Array, optional
        Precomputed trajectories. ``None`` solves with ``_traj_jit``.
    packed : tuple, optional
        ``(PackedData, meta)`` from ``jax_core.pack_data``.

    Returns
    -------
    assignments : np.ndarray
        Shape (n_patients,).
    probabilities : np.ndarray
        Shape (n_patients, n_subtypes).
    """
    del ode_method, lambda_cog
    if rng is None:
        rng = np.random.default_rng()
    n_patients = len(beta)
    n_subtypes = len(cluster_f)
    packed = _get_packed(packed, X_obs, obs_weight, dt, ids, cog, n_patients)
    if X_preds is None:
        X_preds = solve_trajectories_jax(cluster_f, kappa, scalar_K, K, t_span)
    sse = _sse_matrix_np(
        packed, beta, X_preds, s, t_span, cluster_cog_a, cluster_cog_b,
        0.0, n_patients,
    )

    assignments = np.zeros(n_patients, dtype=int)
    probabilities = np.zeros((n_patients, n_subtypes))
    for p_idx in range(n_patients):
        log_p = -sse[p_idx] / temperature
        log_p -= np.max(log_p)
        p = np.exp(log_p)
        p /= p.sum()
        probabilities[p_idx, :] = p
        assignments[p_idx] = rng.choice(n_subtypes, p=p)
    return assignments, probabilities
