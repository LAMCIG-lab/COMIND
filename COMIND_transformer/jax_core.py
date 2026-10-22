"""
JAX core for the SubtypingEM ``backend="jax"`` path.

Forward model (identical to the NumPy path)::

    dx/dt = (1 - x) * (K_eff @ x + f_z),   x(0) = 0,
    K_eff = scalar_K * K + diag(kappa),
    prediction for row r (patient i, visit j) = s * x_{z_i}(dt_ij + beta_i)

The ODE is integrated with fixed-step classical RK4 on the same ``t_span``
SubtypingEM uses (``np.linspace(0, t_max, int(t_max / step))``), and the
trajectory is linearly interpolated exactly like ``np.interp``. Gradients are
exact reverse-mode derivatives of the discretized loss. scipy's L-BFGS-B
(see ``optimizer_jax.py``) remains the optimizer and only sees NumPy vectors.

Recompile rules
---------------
Every edit to this module (and to ``optimizer_jax.py``) must keep these:

- Every ``jax.jit`` is defined at module level, once. Never create a jitted
  function inside ``fit``, ``_em_step``, a loop, or a closure rebuilt per call.
- Fixed shapes. Never pass a row subset like ``X_obs[mask]`` into a jitted
  function. Rows and patients are padded once (``pack_data``); subsets are
  expressed with masks or weights.
- Values that change between calls are traced array arguments: parameters,
  beta, assignments, every lambda, beta_mean, beta_var, cog_a, cog_b and the
  bandwidth override. Scalars are converted with
  ``jnp.asarray(x, dtype=jnp.float64)`` so a Python int vs float never
  changes the trace.
- Static arguments are only ``n_steps`` (int), ``n_bins`` (int), ``use_jsd``
  (bool) and ``auto_bandwidth`` (bool). Always pass them by keyword.
  ``n_subtypes`` enters through array shapes.
- No Python ``if`` on traced values inside jitted code. Use multiplication by
  the lambda, ``jnp.where``, or a static flag.
- Integer arrays are int32, float arrays are float64, always.
- Convert every NumPy input with ``jnp.asarray(..., dtype=...)`` before
  calling jitted code.
"""

import jax

# Must run before any JAX array is created.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from typing import NamedTuple, Optional, Tuple  # noqa: E402

from .utils import split_observed  # noqa: E402

CLIP_EPS = 1e-9
SCALAR_K_MIN = 1e-12
JSD_EPS = 1e-10
DEFAULT_ROW_MULTIPLE = 256
DEFAULT_PATIENT_MULTIPLE = 64


def f64(x) -> jnp.ndarray:
    """Convert ``x`` to a (non-weakly-typed) float64 JAX array."""
    return jnp.asarray(x, dtype=jnp.float64)


def i32(x) -> jnp.ndarray:
    """Convert ``x`` to an int32 JAX array."""
    return jnp.asarray(x, dtype=jnp.int32)


# ---------------------------------------------------------------------------
# Grid and solver
# ---------------------------------------------------------------------------


def make_grid(t_max: float, step: float) -> Tuple[np.ndarray, float, int]:
    """
    Build the SubtypingEM time grid and its RK4 step parameters.

    Parameters
    ----------
    t_max : float
        End of the time grid.
    step : float
        Nominal grid spacing. The actual spacing is ``t_span[1] - t_span[0]``
        because ``np.linspace`` is used, exactly as in ``SubtypingEM.fit``.

    Returns
    -------
    t_span : np.ndarray
        ``np.linspace(0, t_max, int(t_max / step))``.
    h : float
        Actual uniform spacing of ``t_span``.
    n_steps : int
        Number of RK4 steps, ``len(t_span) - 1``.
    """
    t_span = np.linspace(0.0, t_max, int(t_max / step))
    return grid_from_t_span(t_span)


def grid_from_t_span(t_span: np.ndarray) -> Tuple[np.ndarray, float, int]:
    """
    Derive ``(t_span, h, n_steps)`` from an existing uniform grid.

    Parameters
    ----------
    t_span : np.ndarray
        Uniform grid starting at 0.

    Returns
    -------
    t_span : np.ndarray
    h : float
    n_steps : int
    """
    t_span = np.asarray(t_span, dtype=float)
    if t_span.ndim != 1 or len(t_span) < 2:
        raise ValueError(f"t_span must be 1-D with at least 2 points, got {t_span.shape}")
    assert t_span[0] == 0.0, f"t_span must start at 0, got {t_span[0]}"
    h = float(t_span[1] - t_span[0])
    return t_span, h, len(t_span) - 1


def rhs(x: jnp.ndarray, K_eff: jnp.ndarray, f: jnp.ndarray) -> jnp.ndarray:
    """
    Right-hand side of the logistic ODE, with the same clip as ``utils.solve_system``.

    Parameters
    ----------
    x : jnp.ndarray
        State, shape (B,).
    K_eff : jnp.ndarray
        Effective connectivity, shape (B, B).
    f : jnp.ndarray
        Forcing, shape (B,).

    Returns
    -------
    jnp.ndarray
        ``(1 - x_c) * (K_eff @ x_c + f)`` with ``x_c = clip(x, 0, 1 - 1e-9)``.

    Notes
    -----
    The clip is written with ``jnp.where`` so that its derivative is 1 on the
    closed interval. ``jnp.clip`` would give 0.5 at a tie, which happens at
    ``x = 0`` exactly (e.g. a biomarker with ``f_b = 0``) and would bias
    gradients at the ``f >= 0`` bound.
    """
    hi = 1.0 - CLIP_EPS
    xc = jnp.where(x < 0.0, 0.0, jnp.where(x > hi, hi, x))
    return (1.0 - xc) * (K_eff @ xc + f)


def solve_rk4(
    f: jnp.ndarray, K_eff: jnp.ndarray, h: jnp.ndarray, n_steps: int
) -> jnp.ndarray:
    """
    Classical fixed-step RK4 from ``x0 = 0``.

    Parameters
    ----------
    f : jnp.ndarray
        Forcing, shape (B,).
    K_eff : jnp.ndarray
        Effective connectivity, shape (B, B).
    h : jnp.ndarray
        Step size (float64 scalar).
    n_steps : int
        Number of steps (static).

    Returns
    -------
    jnp.ndarray
        Trajectory of shape (n_steps + 1, B), including ``x0``.
    """
    x0 = jnp.zeros_like(f)

    def step(x, _):
        k1 = rhs(x, K_eff, f)
        k2 = rhs(x + 0.5 * h * k1, K_eff, f)
        k3 = rhs(x + 0.5 * h * k2, K_eff, f)
        k4 = rhs(x + h * k3, K_eff, f)
        x_new = x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return x_new, x_new

    _, xs = jax.lax.scan(step, x0, None, length=n_steps)
    return jnp.concatenate([x0[None, :], xs], axis=0)


def solve_trajectories(
    F: jnp.ndarray,
    kappa: jnp.ndarray,
    scalar_K: jnp.ndarray,
    K: jnp.ndarray,
    h: jnp.ndarray,
    n_steps: int,
) -> jnp.ndarray:
    """
    Solve every subtype's trajectory.

    Parameters
    ----------
    F : jnp.ndarray
        Per-subtype forcing, shape (Z, B).
    kappa : jnp.ndarray
        Self-propagation, shape (B,).
    scalar_K : jnp.ndarray
        Connectivity scale (float64 scalar).
    K : jnp.ndarray
        Connectivity, shape (B, B).
    h : jnp.ndarray
        Step size (float64 scalar).
    n_steps : int
        Number of RK4 steps (static).

    Returns
    -------
    jnp.ndarray
        Trajectories of shape (Z, T, B) with ``T = n_steps + 1``.
    """
    K_eff = scalar_K * K + jnp.diag(kappa)
    return jax.vmap(lambda f: solve_rk4(f, K_eff, h, n_steps))(F)


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------


def interp_rows(
    traj: jnp.ndarray, t: jnp.ndarray, z: jnp.ndarray, h: jnp.ndarray
) -> jnp.ndarray:
    """
    Linear interpolation of each row's subtype trajectory, matching ``np.interp``.

    Parameters
    ----------
    traj : jnp.ndarray
        Trajectories, shape (Z, T, B), on the uniform grid ``i * h``.
    t : jnp.ndarray
        Query times, shape (R,). Values outside ``[0, (T-1) h]`` are clamped
        to the end values, as ``np.interp`` does.
    z : jnp.ndarray
        Subtype index per row, shape (R,), int32.
    h : jnp.ndarray
        Grid spacing (float64 scalar).

    Returns
    -------
    jnp.ndarray
        Interpolated values, shape (R, B).
    """
    T = traj.shape[1]
    u = jnp.clip(t / h, 0.0, T - 1.0)
    i0 = jnp.minimum(jnp.floor(u).astype(jnp.int32), T - 2)
    w = (u - i0)[:, None]
    return (1.0 - w) * traj[z, i0] + w * traj[z, i0 + 1]


# ---------------------------------------------------------------------------
# Data packing
# ---------------------------------------------------------------------------


class PackedData(NamedTuple):
    """Padded, fixed-shape observation arrays (a valid JAX pytree).

    Attributes
    ----------
    X : (R, B) float64
        Observations with missing entries replaced by 0.
    W : (R, B) float64
        Observation weight (1 observed, 0 missing or padding).
    dt : (R,) float64
        Time since first visit (0 for padding).
    ids : (R,) int32
        Patient index per row (a padding patient index for padding rows).
    cog : (R, C) float64
        Cognitive covariates (0 for padding).
    row_valid : (R,) float64
        1 for real rows, 0 for padding.
    patient_valid : (P,) float64
        1 for real patients, 0 for padding.
    """

    X: jnp.ndarray
    W: jnp.ndarray
    dt: jnp.ndarray
    ids: jnp.ndarray
    cog: jnp.ndarray
    row_valid: jnp.ndarray
    patient_valid: jnp.ndarray


def _round_up(n: int, multiple: int) -> int:
    return max(multiple, -(-n // multiple) * multiple)


def pack_data(
    X_obs: np.ndarray,
    obs_weight: Optional[np.ndarray],
    dt: np.ndarray,
    ids: np.ndarray,
    cog: np.ndarray,
    n_patients: int,
    pad_rows: Optional[int] = None,
    pad_patients: Optional[int] = None,
) -> Tuple[PackedData, dict]:
    """
    Pad stacked observations to a fixed shape and convert them to JAX arrays.

    Parameters
    ----------
    X_obs : np.ndarray
        Stacked observations, shape (n_rows, B). NaN marks a missing entry.
    obs_weight : np.ndarray or None
        Optional (n_rows, B) weight (1 observed, 0 missing). Combined with
        the NaN mask of ``X_obs``.
    dt : np.ndarray
        Time since first visit, shape (n_rows,).
    ids : np.ndarray
        Patient index per row in ``0..n_patients-1``, shape (n_rows,).
    cog : np.ndarray
        Cognitive covariates, shape (n_rows,) or (n_rows, C).
    n_patients : int
        Number of real patients.
    pad_rows : int, optional
        Padded row count ``R``. Defaults to ``n_rows`` rounded up to a
        multiple of 256. Pass the same value to every CV fold so all folds
        share one compiled function.
    pad_patients : int, optional
        Padded patient count ``P``. Defaults to ``n_patients`` rounded up to
        a multiple of 64.

    Returns
    -------
    data : PackedData
        Padded JAX arrays.
    meta : dict
        Plain Python ``n_rows``, ``n_patients``, ``R``, ``P``. Keep it out of
        jitted code.

    Notes
    -----
    Padding rows use ``ids = n_patients`` (the first padding patient), or
    ``P - 1`` when there are no padding patients, so every gather stays in
    bounds. Their weight, ``dt``, ``cog`` and ``row_valid`` are all 0.
    """
    X_filled, w_nan = split_observed(X_obs)
    W = w_nan if obs_weight is None else w_nan * np.asarray(obs_weight, dtype=float)
    dt = np.asarray(dt, dtype=float).ravel()
    ids = np.asarray(ids).ravel()
    cog = np.asarray(cog, dtype=float)
    if cog.ndim == 1:
        cog = cog.reshape(-1, 1)

    n_rows, n_biomarkers = X_filled.shape
    n_patients = int(n_patients)
    if not (len(dt) == len(ids) == cog.shape[0] == n_rows):
        raise ValueError(
            f"Shapes disagree: X_obs={X_filled.shape}, dt={dt.shape}, "
            f"ids={ids.shape}, cog={cog.shape}"
        )
    if n_rows and (ids.min() < 0 or ids.max() >= n_patients):
        raise ValueError(f"ids must lie in [0, {n_patients})")

    R = _round_up(n_rows, DEFAULT_ROW_MULTIPLE) if pad_rows is None else int(pad_rows)
    P = (
        _round_up(n_patients, DEFAULT_PATIENT_MULTIPLE)
        if pad_patients is None
        else int(pad_patients)
    )
    if R < n_rows:
        raise ValueError(f"pad_rows={R} is smaller than the number of rows ({n_rows})")
    if P < n_patients:
        raise ValueError(
            f"pad_patients={P} is smaller than the number of patients ({n_patients})"
        )

    pad_id = n_patients if n_patients < P else P - 1

    Xp = np.zeros((R, n_biomarkers))
    Wp = np.zeros((R, n_biomarkers))
    dtp = np.zeros(R)
    idsp = np.full(R, pad_id, dtype=np.int32)
    cogp = np.zeros((R, cog.shape[1]))
    row_valid = np.zeros(R)
    patient_valid = np.zeros(P)

    Xp[:n_rows] = X_filled
    Wp[:n_rows] = W
    dtp[:n_rows] = dt
    idsp[:n_rows] = ids
    cogp[:n_rows] = cog
    row_valid[:n_rows] = 1.0
    patient_valid[:n_patients] = 1.0

    data = PackedData(
        X=f64(Xp),
        W=f64(Wp),
        dt=f64(dtp),
        ids=i32(idsp),
        cog=f64(cogp),
        row_valid=f64(row_valid),
        patient_valid=f64(patient_valid),
    )
    meta = dict(n_rows=n_rows, n_patients=n_patients, R=R, P=P)
    return data, meta


def pad_beta(beta: np.ndarray, P: int) -> jnp.ndarray:
    """
    Real betas followed by zeros, length ``P``, float64.

    Parameters
    ----------
    beta : np.ndarray
        Real-patient betas, shape (n_patients,).
    P : int
        Padded patient count.

    Returns
    -------
    jnp.ndarray
    """
    out = np.zeros(P)
    beta = np.asarray(beta, dtype=float).ravel()
    out[: len(beta)] = beta
    return f64(out)


def pad_assignments(assignments: np.ndarray, P: int) -> jnp.ndarray:
    """
    Real assignments followed by zeros, length ``P``, int32.

    Parameters
    ----------
    assignments : np.ndarray
        Real-patient subtype indices, shape (n_patients,).
    P : int
        Padded patient count.

    Returns
    -------
    jnp.ndarray
    """
    out = np.zeros(P, dtype=np.int32)
    assignments = np.asarray(assignments).ravel()
    out[: len(assignments)] = assignments
    return i32(out)


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------


class Hyp(NamedTuple):
    """Penalty weights and prior constants, each a float64 scalar array."""

    lambda_s: jnp.ndarray
    lambda_scalar: jnp.ndarray
    scalar_K_center: jnp.ndarray
    lambda_kappa: jnp.ndarray
    lambda_f: jnp.ndarray
    lambda_cog: jnp.ndarray
    lambda_beta: jnp.ndarray
    beta_mean: jnp.ndarray
    beta_var: jnp.ndarray
    lambda_jsd: jnp.ndarray
    lambda_c: jnp.ndarray


_HYP_DEFAULTS = dict(
    lambda_s=0.0,
    lambda_scalar=0.0,
    scalar_K_center=0.25,
    lambda_kappa=0.0,
    lambda_f=0.0,
    lambda_cog=0.0,
    lambda_beta=0.0,
    beta_mean=0.0,
    beta_var=1.0,
    lambda_jsd=0.0,
    lambda_c=0.0,
)


def make_hyp(**kwargs) -> Hyp:
    """
    Build a ``Hyp`` with every value converted to a float64 JAX scalar.

    Parameters
    ----------
    **kwargs
        Any ``Hyp`` field. Missing fields default to 0.0, except
        ``scalar_K_center = 0.25`` and ``beta_var = 1.0``.

    Returns
    -------
    Hyp
    """
    unknown = set(kwargs) - set(_HYP_DEFAULTS)
    if unknown:
        raise TypeError(f"Unknown hyperparameter(s): {sorted(unknown)}")
    vals = {**_HYP_DEFAULTS, **kwargs}
    return Hyp(**{k: f64(vals[k]) for k in Hyp._fields})


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def recon_residuals(
    traj: jnp.ndarray,
    s: jnp.ndarray,
    beta_full: jnp.ndarray,
    assign_full: jnp.ndarray,
    data: PackedData,
    h: jnp.ndarray,
) -> jnp.ndarray:
    """
    Weighted reconstruction residuals for every (padded) row.

    Parameters
    ----------
    traj : jnp.ndarray
        Trajectories, shape (Z, T, B).
    s : jnp.ndarray
        Output scale, shape (B,).
    beta_full : jnp.ndarray
        Padded betas, shape (P,).
    assign_full : jnp.ndarray
        Padded int32 assignments, shape (P,).
    data : PackedData
    h : jnp.ndarray
        Grid spacing.

    Returns
    -------
    jnp.ndarray
        ``(X - s * x_z(dt + beta)) * W``, shape (R, B).
    """
    t = data.dt + beta_full[data.ids]
    z = assign_full[data.ids]
    pred = s * interp_rows(traj, t, z, h)
    return (data.X - pred) * data.W


def recon_sse(
    traj: jnp.ndarray,
    s: jnp.ndarray,
    beta_full: jnp.ndarray,
    assign_full: jnp.ndarray,
    data: PackedData,
    h: jnp.ndarray,
) -> jnp.ndarray:
    """
    Plain weighted reconstruction SSE (no penalties).

    Parameters
    ----------
    See ``recon_residuals``.

    Returns
    -------
    jnp.ndarray
        Scalar ``sum(r**2)``.
    """
    r = recon_residuals(traj, s, beta_full, assign_full, data, h)
    return jnp.sum(r ** 2)


def globals_loss(
    s: jnp.ndarray,
    kappa: jnp.ndarray,
    scalar_K: jnp.ndarray,
    F: jnp.ndarray,
    beta_full: jnp.ndarray,
    assign_full: jnp.ndarray,
    data: PackedData,
    K: jnp.ndarray,
    hyp: Hyp,
    h: jnp.ndarray,
    n_steps: int,
    s_free: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    Loss for the globals step, matching ``theta_s_loss_multi``.

    Parameters
    ----------
    s, kappa : jnp.ndarray
        Shape (B,).
    scalar_K : jnp.ndarray
        Float64 scalar.
    F : jnp.ndarray
        Per-subtype forcing, shape (Z, B), held fixed.
    beta_full, assign_full : jnp.ndarray
        Padded betas (P,) and int32 assignments (P,).
    data : PackedData
    K : jnp.ndarray
        Connectivity, shape (B, B).
    hyp : Hyp
    h : jnp.ndarray
        Grid spacing.
    n_steps : int
        Number of RK4 steps (static).
    s_free : jnp.ndarray, optional
        Float mask (B,), 1 where ``s`` is free. The ridge penalty
        ``lambda_s * sum(s**2)`` only covers free entries, mirroring
        ``_s_ridge_penalty``. ``None`` treats every entry as free.

    Returns
    -------
    jnp.ndarray
        ``sum(r**2) + lambda_s * sum(s_free * s**2)
        + 0.5 * lambda_scalar * (log(max(scalar_K, 1e-12)) - log(center))**2
        + lambda_kappa * sum(kappa)``. There is no f penalty.
    """
    traj = solve_trajectories(F, kappa, scalar_K, K, h, n_steps)
    sse = recon_sse(traj, s, beta_full, assign_full, data, h)
    s_sq = s ** 2 if s_free is None else s_free * s ** 2
    scalar_K_safe = jnp.maximum(scalar_K, SCALAR_K_MIN)
    lognorm = 0.5 * hyp.lambda_scalar * (
        jnp.log(scalar_K_safe) - jnp.log(hyp.scalar_K_center)
    ) ** 2
    return (
        sse
        + hyp.lambda_s * jnp.sum(s_sq)
        + lognorm
        + hyp.lambda_kappa * jnp.sum(kappa)
    )


def shift_data(data: PackedData, c: jnp.ndarray) -> PackedData:
    """
    Subtract a per-biomarker intercept from the observations.

    The model with an intercept predicts ``c + s * x``. Its residual
    ``X - c - s * x`` equals the no-intercept residual on ``X - c``, so every
    step except the one that fits ``c`` runs unchanged on the shifted data.
    Missing and padding entries have weight 0, so shifting them is harmless.

    Parameters
    ----------
    data : PackedData
    c : jnp.ndarray
        Intercept, shape (B,).

    Returns
    -------
    PackedData
        Same arrays, with ``X - c``.
    """
    return data._replace(X=data.X - c[None, :])


def globals_loss_intercept(
    s: jnp.ndarray,
    kappa: jnp.ndarray,
    scalar_K: jnp.ndarray,
    c: jnp.ndarray,
    F: jnp.ndarray,
    beta_full: jnp.ndarray,
    assign_full: jnp.ndarray,
    data: PackedData,
    K: jnp.ndarray,
    hyp: Hyp,
    h: jnp.ndarray,
    c_center: jnp.ndarray,
    n_steps: int,
    s_free: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    ``globals_loss`` with a per-biomarker intercept ``c`` fitted jointly.

    Prediction for row r: ``c + s * x_z(dt + beta)``. ``data`` holds the
    unshifted observations. Adds the optional ridge
    ``lambda_c * sum((c - c_center)**2)`` that anchors ``c`` softly.
    """
    loss = globals_loss(
        s, kappa, scalar_K, F, beta_full, assign_full, shift_data(data, c), K,
        hyp, h, n_steps=n_steps, s_free=s_free,
    )
    return loss + hyp.lambda_c * jnp.sum((c - c_center) ** 2)


def cluster_loss(
    F: jnp.ndarray,
    s: jnp.ndarray,
    kappa: jnp.ndarray,
    scalar_K: jnp.ndarray,
    beta_full: jnp.ndarray,
    assign_full: jnp.ndarray,
    data: PackedData,
    K: jnp.ndarray,
    hyp: Hyp,
    h: jnp.ndarray,
    n_steps: int,
) -> jnp.ndarray:
    """
    Joint loss for every subtype's forcing ``f_z``.

    Parameters
    ----------
    F : jnp.ndarray
        Per-subtype forcing, shape (Z, B).
    s, kappa, scalar_K : jnp.ndarray
        Fixed globals.
    beta_full, assign_full : jnp.ndarray
        Padded betas (P,) and int32 assignments (P,).
    data : PackedData
    K : jnp.ndarray
        Connectivity, shape (B, B).
    hyp : Hyp
    h : jnp.ndarray
        Grid spacing.
    n_steps : int
        Number of RK4 steps (static).

    Returns
    -------
    jnp.ndarray
        ``sum(r**2) + lambda_f * sum(F)``.

    Notes
    -----
    The per-subtype problems of ``fit_theta_cluster`` are separable: each row
    only sees the trajectory of its own patient's subtype, and ``x_z`` depends
    only on ``f_z``. So this summed loss equals the sum of the per-subtype
    ``theta_cluster_loss`` values (over non-empty subtypes, plus constant
    ``lambda_f * sum(f_z)`` terms for empty ones), and its gradient for
    ``f_z`` equals the per-subtype gradient.
    """
    traj = solve_trajectories(F, kappa, scalar_K, K, h, n_steps)
    sse = recon_sse(traj, s, beta_full, assign_full, data, h)
    return sse + hyp.lambda_f * jnp.sum(F)


def _gaussian_kernel(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.exp(-0.5 * x ** 2) / jnp.sqrt(2.0 * jnp.pi)


def kernel_jsd(
    beta_full: jnp.ndarray,
    assign_full: jnp.ndarray,
    patient_valid: jnp.ndarray,
    Z: int,
    jsd_lo: jnp.ndarray,
    jsd_hi: jnp.ndarray,
    bandwidth: jnp.ndarray,
    n_bins: int,
    auto_bandwidth: bool,
) -> jnp.ndarray:
    """
    Kernel JSD between subtype beta distributions, matching ``KernelJSDMulti.jsd()``.

    Parameters
    ----------
    beta_full : jnp.ndarray
        Padded betas, shape (P,).
    assign_full : jnp.ndarray
        Padded int32 assignments, shape (P,).
    patient_valid : jnp.ndarray
        1 for real patients, 0 for padding, shape (P,).
    Z : int
        Number of subtypes (Python int, e.g. from an array shape).
    jsd_lo, jsd_hi : jnp.ndarray
        Bin range.
    bandwidth : jnp.ndarray
        Kernel bandwidth; ignored when ``auto_bandwidth``.
    n_bins : int
        Number of bins (static).
    auto_bandwidth : bool
        If True use ``0.9 * std(real betas) * n_real**(-1/5)`` (population
        std), differentiated through beta (static).

    Returns
    -------
    jnp.ndarray
        Scalar JSD over the present subtypes (``count_z > 0``); 0 when fewer
        than two subtypes are present.
    """
    eps = JSD_EPS
    pv = patient_valid
    bins = jnp.linspace(jsd_lo, jsd_hi, n_bins)

    if auto_bandwidth:
        n_real = jnp.sum(pv)
        mean = jnp.sum(pv * beta_full) / n_real
        var = jnp.sum(pv * (beta_full - mean) ** 2) / n_real
        bw = 0.9 * jnp.sqrt(var) * n_real ** (-0.2)
    else:
        bw = bandwidth

    m = pv[None, :] * (assign_full[None, :] == jnp.arange(Z)[:, None])  # (Z, P)
    count = jnp.sum(m, axis=1)  # (Z,)
    present = (count > 0).astype(jnp.float64)
    n_present = jnp.sum(present)

    kern = _gaussian_kernel((bins[:, None] - beta_full[None, :]) / bw)  # (n_bins, P)
    P_z = (m @ kern.T) / (jnp.maximum(count, 1.0)[:, None] * bw)  # (Z, n_bins)
    P_z = P_z / (jnp.sum(P_z, axis=1, keepdims=True) + eps)
    P_z = P_z + eps

    denom = jnp.maximum(n_present, 1.0)
    M = jnp.sum(present[:, None] * P_z, axis=0) / denom + eps
    kl = jnp.sum(P_z * jnp.log(P_z / M[None, :]), axis=1)  # (Z,)
    jsd = jnp.sum(present * kl) / denom
    return jnp.where(n_present >= 2, jsd, 0.0)


def beta_loss(
    beta_full: jnp.ndarray,
    traj: jnp.ndarray,
    s: jnp.ndarray,
    assign_full: jnp.ndarray,
    cog_a: jnp.ndarray,
    cog_b: jnp.ndarray,
    data: PackedData,
    hyp: Hyp,
    h: jnp.ndarray,
    jsd_lo: jnp.ndarray,
    jsd_hi: jnp.ndarray,
    bandwidth: jnp.ndarray,
    n_bins: int,
    use_jsd: bool,
    auto_bandwidth: bool,
) -> jnp.ndarray:
    """
    Joint beta loss, matching ``_vectorized_beta_loss_and_grad``.

    Parameters
    ----------
    beta_full : jnp.ndarray
        Padded betas, shape (P,).
    traj : jnp.ndarray
        Precomputed trajectories, shape (Z, T, B). The ODE is not solved here.
    s : jnp.ndarray
        Output scale, shape (B,).
    assign_full : jnp.ndarray
        Padded int32 assignments, shape (P,).
    cog_a : jnp.ndarray
        Per-subtype cognitive slopes, shape (Z, C).
    cog_b : jnp.ndarray
        Per-subtype cognitive intercepts, shape (Z,).
    data : PackedData
    hyp : Hyp
    h : jnp.ndarray
        Grid spacing.
    jsd_lo, jsd_hi, bandwidth : jnp.ndarray
        JSD bin range and bandwidth override.
    n_bins : int
        JSD bin count (static).
    use_jsd : bool
        Include the JSD term (static).
    auto_bandwidth : bool
        Use the data-driven JSD bandwidth (static).

    Returns
    -------
    jnp.ndarray
        Reconstruction SSE + ``lambda_cog`` cognitive term + ``lambda_beta``
        prior + (optionally) ``lambda_jsd * kernel_jsd``.
    """
    sse = recon_sse(traj, s, beta_full, assign_full, data, h)

    t = data.dt + beta_full[data.ids]
    z = assign_full[data.ids]
    cog_pred = jnp.sum(data.cog * cog_a[z], axis=1) + cog_b[z]
    cog_term = jnp.sum(data.row_valid * (t - cog_pred) ** 2)

    prior = jnp.sum(data.patient_valid * (beta_full - hyp.beta_mean) ** 2) / hyp.beta_var

    loss = sse + hyp.lambda_cog * cog_term + hyp.lambda_beta * prior
    if use_jsd:
        loss = loss + hyp.lambda_jsd * kernel_jsd(
            beta_full,
            assign_full,
            data.patient_valid,
            cog_a.shape[0],
            jsd_lo,
            jsd_hi,
            bandwidth,
            n_bins,
            auto_bandwidth,
        )
    return loss


def sse_matrix_jax(
    traj: jnp.ndarray,
    s: jnp.ndarray,
    beta_full: jnp.ndarray,
    data: PackedData,
    cog_a: jnp.ndarray,
    cog_b: jnp.ndarray,
    lambda_cog: jnp.ndarray,
    h: jnp.ndarray,
) -> jnp.ndarray:
    """
    Per-patient, per-subtype SSE, matching ``assignments.sse_matrix``.

    Parameters
    ----------
    traj : jnp.ndarray
        Trajectories, shape (Z, T, B).
    s : jnp.ndarray
        Output scale, shape (B,).
    beta_full : jnp.ndarray
        Padded betas, shape (P,).
    data : PackedData
    cog_a : jnp.ndarray
        Shape (Z, C).
    cog_b : jnp.ndarray
        Shape (Z,).
    lambda_cog : jnp.ndarray
        Weight of the cognitive term (0 disables it).
    h : jnp.ndarray
        Grid spacing.

    Returns
    -------
    jnp.ndarray
        Shape (P, Z). Padding patients get 0 in every column.
    """
    P = beta_full.shape[0]
    per_row = sse_rows_jax(traj, s, beta_full, data, cog_a, cog_b, lambda_cog, h)
    return jax.ops.segment_sum(per_row, data.ids, num_segments=P)


def sse_rows_jax(
    traj: jnp.ndarray,
    s: jnp.ndarray,
    beta_full: jnp.ndarray,
    data: PackedData,
    cog_a: jnp.ndarray,
    cog_b: jnp.ndarray,
    lambda_cog: jnp.ndarray,
    h: jnp.ndarray,
) -> jnp.ndarray:
    """
    Per-row (visit-level) loss against every subtype; ``sse_matrix_jax`` sums it per patient.

    Parameters
    ----------
    See ``sse_matrix_jax``.

    Returns
    -------
    jnp.ndarray
        Shape (R, Z): the visit's weighted reconstruction SSE plus
        ``lambda_cog * (t - cog_pred_z)**2``. Padding rows get 0.
    """
    Z = traj.shape[0]
    t = data.dt + beta_full[data.ids]
    zeros = jnp.zeros_like(data.ids)

    def per_subtype(zi):
        pred = s * interp_rows(traj, t, zeros + zi, h)
        r = (data.X - pred) * data.W
        rec = jnp.sum(r ** 2, axis=1)
        cog_pred = data.cog @ cog_a[zi] + cog_b[zi]
        return rec + lambda_cog * data.row_valid * (t - cog_pred) ** 2

    return jax.vmap(per_subtype)(jnp.arange(Z, dtype=jnp.int32)).T


def joint_grid_losses_jax(
    traj: jnp.ndarray,
    s: jnp.ndarray,
    candidates: jnp.ndarray,
    data: PackedData,
    cog_a: jnp.ndarray,
    cog_b: jnp.ndarray,
    lambda_cog: jnp.ndarray,
    h: jnp.ndarray,
) -> jnp.ndarray:
    """
    ``sse_matrix_jax`` at every row of a candidate-β table.

    Parameters
    ----------
    traj : jnp.ndarray
        Trajectories, shape (Z, T, B).
    s : jnp.ndarray
        Output scale, shape (B,).
    candidates : jnp.ndarray
        Padded candidate βs, shape (C, P). Traced; C enters only through the
        shape, so each candidate count compiles once.
    data : PackedData
    cog_a, cog_b, lambda_cog, h
        As in ``sse_matrix_jax``.

    Returns
    -------
    jnp.ndarray
        Shape (C, P, Z).
    """
    return jax.vmap(
        lambda beta_c: sse_matrix_jax(traj, s, beta_c, data, cog_a, cog_b, lambda_cog, h)
    )(candidates)


def joint_grid_select(
    traj: jnp.ndarray,
    s: jnp.ndarray,
    candidates: jnp.ndarray,
    data: PackedData,
    cog_a: jnp.ndarray,
    cog_b: jnp.ndarray,
    lambda_cog: jnp.ndarray,
    beta_upper_full: jnp.ndarray,
    h: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Joint (candidate β, subtype) argmin per patient, matching ``assignments.select_joint_grid``.

    Parameters
    ----------
    traj : jnp.ndarray
        Trajectories, shape (Z, T, B).
    s : jnp.ndarray
        Output scale, shape (B,).
    candidates : jnp.ndarray
        Padded candidate βs, shape (C, P). For the E-step row 0 is the current
        β and rows 1.. are the grid (C = G + 1); for ``transform`` it is the
        grid only.
    data : PackedData
    cog_a, cog_b : jnp.ndarray
        Shapes (Z, C_cog) and (Z,).
    lambda_cog : jnp.ndarray
        Weight of the cognitive term (0 disables it).
    beta_upper_full : jnp.ndarray
        Padded per-patient upper bound, shape (P,). Candidates above it get
        loss ``+inf``. Padding patients should get ``t_max``; their losses are
        0 and never affect real patients.
    h : jnp.ndarray
        Grid spacing.

    Returns
    -------
    best_beta : jnp.ndarray
        Chosen candidate β, shape (P,).
    best_z : jnp.ndarray
        Chosen subtype, shape (P,), int32.

    Notes
    -----
    The argmin runs over the candidate-major flattening of the (C, Z) table,
    so ties go to the lower candidate index, then the lower subtype index.
    """
    losses = joint_grid_losses_jax(traj, s, candidates, data, cog_a, cog_b, lambda_cog, h)
    valid = candidates <= beta_upper_full[None, :]
    losses = jnp.where(valid[:, :, None], losses, jnp.inf)
    C, P, Z = losses.shape
    flat = jnp.transpose(losses, (1, 0, 2)).reshape(P, C * Z)
    idx = jnp.argmin(flat, axis=1).astype(jnp.int32)
    best_c = idx // Z
    best_z = idx % Z
    best_beta = jnp.take_along_axis(candidates, best_c[None, :], axis=0)[0]
    return best_beta, best_z


# ---------------------------------------------------------------------------
# Module-level jitted functions (see "Recompile rules" above)
# ---------------------------------------------------------------------------

_traj_jit = jax.jit(solve_trajectories, static_argnames=("n_steps",))
_globals_vg = jax.jit(
    jax.value_and_grad(globals_loss, argnums=(0, 1, 2)),
    static_argnames=("n_steps",),
)
_globals_c_vg = jax.jit(
    jax.value_and_grad(globals_loss_intercept, argnums=(0, 1, 2, 3)),
    static_argnames=("n_steps",),
)
_shift_data_jit = jax.jit(shift_data)
_cluster_vg = jax.jit(
    jax.value_and_grad(cluster_loss, argnums=0),
    static_argnames=("n_steps",),
)
_beta_vg = jax.jit(
    jax.value_and_grad(beta_loss, argnums=0),
    static_argnames=("n_bins", "use_jsd", "auto_bandwidth"),
)
_recon_sse_jit = jax.jit(recon_sse)
_sse_matrix_jit = jax.jit(sse_matrix_jax)
_joint_grid_select_jit = jax.jit(joint_grid_select)
_joint_grid_losses_jit = jax.jit(joint_grid_losses_jax)
_sse_rows_jit = jax.jit(sse_rows_jax)

JITTED_FUNCTIONS = {
    "_traj_jit": _traj_jit,
    "_globals_vg": _globals_vg,
    "_globals_c_vg": _globals_c_vg,
    "_shift_data_jit": _shift_data_jit,
    "_cluster_vg": _cluster_vg,
    "_beta_vg": _beta_vg,
    "_recon_sse_jit": _recon_sse_jit,
    "_sse_matrix_jit": _sse_matrix_jit,
    "_joint_grid_select_jit": _joint_grid_select_jit,
    "_joint_grid_losses_jit": _joint_grid_losses_jit,
    "_sse_rows_jit": _sse_rows_jit,
}

TRANSFORM_ONLY_JITTED = ("_joint_grid_losses_jit", "_sse_rows_jit")
