"""
SubtypingEM: EM algorithm for COMIND disease progression modeling with subtype discovery.

sklearn-compatible (BaseEstimator, TransformerMixin).

Missing biomarker entries are indicated by NaN in a patient's ``X_obs``
array (shape (n_visits, n_biomarkers)). Because the likelihood is a
product of independent per-entry Gaussians, the EM treatment of a
missing y^k_ij is to omit its term from the sum of squares (equivalent
to taking the expectation of the log-likelihood under the model's own
current prediction). This is implemented as an elementwise weight
(1.0 = observed, 0.0 = missing) multiplied into every residual. A whole
missing visit is already handled by omitting that row; this mechanism
is for missing entries within an otherwise-present visit. Assumes
MAR/MCAR missingness (not MNAR).

fit() is organized into four helpers:

    _prepare_data     flatten patient dicts → stacked numpy arrays
    _initialize_state build initial EM state (params + history arrays)
    _run_em_loop      outer EM loop: checkpoint / solver escalation / accept
    _finalize_fit     trim histories, store fitted attributes on self

E-step and BIC helpers live in assignments.py and model_selection.py.
"""

import os
import signal
import time
import warnings
import numpy as np
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin

from .assignments import (
    precompute_trajectories,
    sse_matrix,
    update_assignments_hard,
    update_assignments_joint_grid,
    update_assignments_jitter,
)
from .model_selection import (
    compute_sse_per_biomarker,
    count_bic_params,
    compute_bic,
)
from .optimizer_theta_globals import fit_theta_globals, resolve_s_mask
from .optimizer_theta_cluster import fit_theta_cluster
from .optimizer_beta import estimate_beta, reconstruction_sse
from .optimizer_cognitive_regression import fit_linear_cog_regression_multi
from .utils import (
    beta_upper_from_followup,
    check_beta_grid_within_cap,
    make_beta_grid,
    solve_system,
    initialize_beta,
    ensure_2d_cog,
    match_labels_best_overlap,
    split_observed,
)


_BACKENDS = {"numpy", "jax"}
_ESTEP_MODES = {"current", "joint_grid"}
_AT_UPPER_TOL = 1e-3


def _lse_improvement_acceptable(best, lse_val, eps, rel_tol):
    """True when the trial LSE improves enough over the running best."""
    if not np.isfinite(lse_val):
        return False
    if not np.isfinite(best) or best == np.inf:
        return True
    delta = best - lse_val
    if delta >= eps:
        return True
    return delta / max(abs(best), 1e-12) >= rel_tol


class SubtypingEM(BaseEstimator, TransformerMixin):
    """
    EM algorithm for recovering disease progression model parameters
    and patient-specific time shifts from cross-sectional observations.

    Missing data: NaN in a patient's ``X_obs`` marks a missing biomarker
    at that visit. The likelihood is a product of independent per-entry
    Gaussians (paper eq. 4), so the EM treatment of a missing y^k_ij is
    to omit its squared-residual term — equivalent to taking the
    expectation of that term under N(current prediction, sigma_k^2),
    which is a constant independent of theta and beta. Implemented as
    an elementwise ``obs_weight`` (1.0 = observed, 0.0 = missing)
    multiplied into every residual. Assumes missingness is MAR/MCAR
    (not MNAR: e.g. not "segmentation failed because atrophy was too
    severe to parcellate"). A whole missing visit is handled by
    omitting that row; this mechanism is for missing entries within an
    otherwise-present visit.

    Global (s, kappa, scalar_K) and cluster (f) θ-steps use staged optimizers
    from ``theta_solver_stages`` (default ``lbfgs_approx`` → ``nelder_mead``).
    Add ``lbfgs_exact`` for exact LSODA sensitivities (slower), or ``lbfgs_rk4`` for
    fixed-step RK4 sensitivities on the forward grid.

    Each outer iteration starts from the cheapest solver. If reconstruction LSE
    improvement is too small, the iteration retries with the next stage. After
    an accepted update the next iteration starts cheaply again.

    If, after exhausting all stages, LSE still does not strictly improve vs
    ``best_lse``, EM exits early and restores the checkpoint from that outer
    iteration.

    Warm-start: pass ``initial_f``, ``initial_s``, ``initial_scalar_K``,
    ``initial_kappa``, ``initial_assignments``, per-subtype cognitive params,
    ``initial_beta``, or use
    :func:`COMIND_transformer.warm_start.load_warm_start_from_npz`.

    ``s_mask`` (optional boolean vector): True = optimize that biomarker's
    scale ``s``; False = hold it at 1. Use this for ceiling-normalized
    clinical scores whose scale is already known. ``fit_s=False`` fixes
    every entry regardless of ``s_mask``. ``s_lower_percentile`` (e.g. 95)
    bounds each free ``s`` (the biomarker's asymptote) from below by that
    percentile of the biomarker's observed training values (floored at 0);
    initial ``s`` is raised to the bound. ``None`` (default) keeps the lower
    bound at 0. The bound is stored as ``s_lower_`` after ``fit``.

    ``backend="jax"`` replaces the globals, E-step, cluster-f and beta steps
    with JAX versions (RK4 forward solve on ``t_span``, exact reverse-mode
    gradients, scipy L-BFGS-B). ``theta_solver_stages`` is ignored: each
    outer iteration runs exactly one stage. Cognitive regression, BIC and
    checkpointing stay NumPy/LSODA, so the final BIC is an LSODA consistency
    check on the JAX-fitted parameters. ``transform`` and ``transform_soft``
    also stay NumPy/LSODA unless ``transform_backend="jax"``.
    Pass the same ``jax_pad_rows`` / ``jax_pad_patients`` to every CV fold so
    all folds share one compiled function. Requires ``pip install "jax[cpu]"``.

    ``cap_beta_by_followup=True`` bounds patient i's beta to
    ``[0, t_max - max(dt_i)]`` in the beta step and in ``transform`` /
    ``transform_soft``, so every visit stays inside ``t_span`` instead of
    reading the clamped end value of the trajectory. Initial betas are clipped
    to that range. The bounds are stored as ``beta_upper_`` after ``fit``.

    ``estep_mode="joint_grid"`` replaces the hard E-step with a joint search
    over (beta, subtype): each patient's candidates are their current beta and
    the ``estep_grid_size``-point grid ``make_beta_grid(t_max, n)`` (default
    ``n_beta_grid`` points), and the beta step then warm-starts from the
    chosen beta. On iterations where the jittered E-step runs
    (``assignments_jitter``), it runs exactly as before and ``estep_mode`` is
    ignored for that iteration. See
    ``assignments.update_assignments_joint_grid`` for the guarantees.

    ``transform_backend="jax"`` runs ``transform`` / ``transform_soft`` on
    JAX (RK4 trajectories, ``joint_grid_select``, one joint L-BFGS-B
    refinement per call). It is independent of ``backend``: the default
    ``None`` keeps the NumPy/LSODA transform for both backends, so held-out
    scores of past fits are unchanged. Pass the same
    ``transform_pad_rows`` / ``transform_pad_patients`` to every CV fold to
    share one compiled shape.

    Diagnostics recorded every iteration (index 0 = initial state):
    ``subtype_counts_history_`` (n_subtypes, n_iter + 1),
    ``beta_at_upper_fraction_history_`` (fraction of patients with
    ``beta_i >= beta_upper_i - 1e-3``) and ``estep_beta_moves_history_``
    (patients whose beta the E-step moved; always 0 in ``"current"`` mode).
    """

    def __init__(
        self,
        max_iter: int = 50,
        t_max: float = 30,
        step: float = 0.01,
        K: np.ndarray = None,
        rng: np.random.Generator = None,
        lambda_f: float = 0.01,
        lambda_cog: float = 0,
        lambda_scalar: float = 0.0,
        scalar_K_center: float = 0.25,
        lambda_jsd: float = 0.0,
        lambda_beta: float = 0.0,
        lambda_kappa: float = 0.0,
        fit_s: bool = True,
        s_mask: np.ndarray = None,
        initial_f: np.ndarray = None,
        initial_s: np.ndarray = None,
        initial_scalar_K: float = None,
        initial_kappa: np.ndarray = None,
        initial_assignments: np.ndarray = None,
        initial_cluster_cog_a: np.ndarray = None,
        initial_cluster_cog_b: np.ndarray = None,
        initial_beta: np.ndarray = None,
        theta_solver_stages=None,
        epsilon: float = 1e-2,
        relative_tolerance: float = 1e-3,
        n_subtypes: int = 2,
        assignments_jitter: bool = False,
        jitter_iter: int = 1,
        jitter_temperature: float = 1.0,
        verbose: int = 1,
        n_beta_grid: int = 20,
        strict_tol: bool = False,
        ode_method: str = "LSODA",
        n_anneal_iters: int = 0,
        kappa_anneal_strength: float = 0.05,
        f_anneal_strength: float = 0.01,
        anneal_decay: float = 0.7,
        backend: str = "numpy",
        jax_pad_rows: int = None,
        jax_pad_patients: int = None,
        cap_beta_by_followup: bool = False,
        estep_mode: str = "current",
        estep_grid_size: int = None,
        transform_backend: str = None,
        transform_pad_rows: int = None,
        transform_pad_patients: int = None,
        s_lower_percentile: float = None,
        fit_intercept: bool = False,
        initial_c: np.ndarray = None,
        c_bounds=None,
        c_mask: np.ndarray = None,
        lambda_c: float = 0.0,
        c_center: np.ndarray = None,
    ):
        self.max_iter = max_iter
        self.t_max = t_max
        self.step = step
        self.K = K
        self.rng = np.random.default_rng(75) if rng is None else rng

        self.lambda_f = lambda_f
        self.lambda_cog = lambda_cog
        self.lambda_scalar = lambda_scalar
        self.scalar_K_center = scalar_K_center
        self.lambda_jsd = lambda_jsd
        self.lambda_beta = lambda_beta
        self.lambda_kappa = lambda_kappa
        self.fit_s = fit_s
        self.s_mask = s_mask

        self.initial_f = initial_f
        self.initial_s = initial_s
        self.initial_scalar_K = initial_scalar_K
        self.initial_kappa = initial_kappa
        self.initial_assignments = initial_assignments
        self.initial_cluster_cog_a = initial_cluster_cog_a
        self.initial_cluster_cog_b = initial_cluster_cog_b
        self.initial_beta = initial_beta

        if theta_solver_stages is not None:
            self.theta_solver_stages = tuple(theta_solver_stages)
        else:
            self.theta_solver_stages = ("lbfgs_approx", "nelder_mead")

        self.epsilon = epsilon
        self.relative_tolerance = relative_tolerance

        self.n_subtypes = n_subtypes
        self.assignments_jitter = assignments_jitter
        self.jitter_iter = jitter_iter
        self.jitter_temperature = jitter_temperature

        self.verbose = verbose
        self.n_beta_grid = n_beta_grid
        self.strict_tol = strict_tol
        self.ode_method = ode_method
        self.n_anneal_iters = n_anneal_iters
        self.kappa_anneal_strength = kappa_anneal_strength
        self.f_anneal_strength = f_anneal_strength
        self.anneal_decay = anneal_decay
        self.backend = backend
        self.jax_pad_rows = jax_pad_rows
        self.jax_pad_patients = jax_pad_patients
        self.cap_beta_by_followup = cap_beta_by_followup
        self.estep_mode = estep_mode
        self.estep_grid_size = estep_grid_size
        self.transform_backend = transform_backend
        self.transform_pad_rows = transform_pad_rows
        self.transform_pad_patients = transform_pad_patients
        self.s_lower_percentile = s_lower_percentile
        # Per-biomarker intercept: prediction = c + s * x (JAX backend only).
        self.fit_intercept = fit_intercept
        self.initial_c = initial_c
        self.c_bounds = c_bounds
        self.c_mask = c_mask
        self.lambda_c = lambda_c
        self.c_center = c_center
        self._validate_backend()

    def _validate_backend(self):
        if self.backend not in _BACKENDS:
            raise ValueError(
                f"backend must be one of {sorted(_BACKENDS)}, got {self.backend!r}"
            )
        if getattr(self, "fit_intercept", False) and self.backend != "jax":
            raise ValueError("fit_intercept=True requires backend='jax'")
        if self.estep_mode not in _ESTEP_MODES:
            raise ValueError(
                f"estep_mode must be one of {sorted(_ESTEP_MODES)}, got {self.estep_mode!r}"
            )
        if self.transform_backend not in (None, "numpy", "jax"):
            raise ValueError(
                f"transform_backend must be None, 'numpy' or 'jax', "
                f"got {self.transform_backend!r}"
            )
        if self.estep_grid_size is not None and int(self.estep_grid_size) < 1:
            raise ValueError(
                f"estep_grid_size must be >= 1 or None, got {self.estep_grid_size!r}"
            )

    def _resolve_c_bounds(self, n_biomarkers: int) -> np.ndarray:
        """(B, 2) intercept bounds. ``c_mask`` False entries are fixed at their initial c."""
        if self.c_bounds is None:
            b = np.tile([-np.inf, np.inf], (n_biomarkers, 1))
        else:
            b = np.asarray(self.c_bounds, dtype=float)
            b = np.tile(b.reshape(1, 2), (n_biomarkers, 1)) if b.size == 2 else b.reshape(n_biomarkers, 2).copy()
        return b

    def _initial_c(self, n_biomarkers: int) -> np.ndarray:
        if self.initial_c is None:
            return np.zeros(n_biomarkers)
        c = np.asarray(self.initial_c, dtype=float).ravel().copy()
        if c.shape != (n_biomarkers,):
            raise ValueError(f"initial_c must have shape ({n_biomarkers},), got {c.shape}")
        return c

    def _final_c_or_zero(self, n_biomarkers: int) -> np.ndarray:
        c = getattr(self, "final_c", None)
        return np.zeros(n_biomarkers) if c is None else np.asarray(c, dtype=float)

    def _shift_patients(self, X: list) -> list:
        """Copies of the patient dicts with ``X_obs - final_c`` (no-op without an intercept)."""
        if not self.fit_intercept or getattr(self, "final_c", None) is None:
            return X
        c = np.asarray(self.final_c, dtype=float)
        return [{**p, "X_obs": np.asarray(p["X_obs"], dtype=float) - c[None, :]} for p in X]

    def _estep_beta_grid(self) -> np.ndarray:
        """beta grid for the joint-grid E-step (``n_beta_grid`` points by default)."""
        n = self.n_beta_grid if self.estep_grid_size is None else int(self.estep_grid_size)
        return make_beta_grid(self.t_max, n)

    def fit(self, X: list, y=None, checkpoint_path=None):
        """
        Fit on patient list ``X``.

        Parameters
        ----------
        checkpoint_path : str or None
            If set, write an atomic overwrite checkpoint after each accepted EM
            iteration (for recovery after walltime limits). Not used during CV
            unless you pass it on fold fits (not recommended).
        """
        self._validate_backend()
        self.t_span = np.linspace(0.0, self.t_max, int(self.t_max / self.step))
        self.assignment_probabilities_ = None
        self._checkpoint_path = checkpoint_path
        self._checkpoint_X = X
        self._checkpoint_snapshot = None

        flat = self._prepare_data(X)
        if self.backend == "jax":
            from . import jax_core
            from .optimizer_jax import nonfinite_events

            packed, meta = jax_core.pack_data(
                flat["X_obs"],
                flat["obs_weight"],
                flat["dt"],
                flat["ids"],
                flat["cog"],
                flat["n_patients"],
                pad_rows=self.jax_pad_rows,
                pad_patients=self.jax_pad_patients,
            )
            flat["jax"] = dict(packed=(packed, meta), packed_raw=(packed, meta), meta=meta)
            self.jax_padded_shape_ = (meta["R"], meta["P"])
            nonfinite_start = nonfinite_events()
        state = self._initialize_state(flat, X)
        self._register_checkpoint_signal_handler()
        try:
            self._run_em_loop(state, flat, X)
            self._finalize_fit(state, flat, X)
            if self.backend == "jax":
                self.jax_nonfinite_events_ = nonfinite_events() - nonfinite_start
            if self._checkpoint_path:
                self._save_fit_checkpoint(state, flat, X, fit_complete=True)
        finally:
            self._unregister_checkpoint_signal_handler()
        return self

    def _transform_backend_resolved(self) -> str:
        """``"jax"`` only when ``transform_backend="jax"``; otherwise NumPy/LSODA."""
        tb = getattr(self, "transform_backend", None)
        return "jax" if tb == "jax" else "numpy"

    def _transform_trajectories_numpy(self) -> list:
        """Per-subtype (B, T) LSODA trajectories at the fitted parameters."""
        n_biomarkers = np.ravel(self.cluster_f[0]).shape[0]
        return [
            solve_system(
                np.zeros(n_biomarkers),
                np.ravel(self.cluster_f[z]),
                self.K,
                self.t_span,
                self.final_scalar_K,
                self.final_kappa,
                ode_method=self.ode_method,
            )
            for z in range(self.n_subtypes)
        ]

    def transform(self, X: list, use_cognitive_prior: bool = True, _shifted: bool = False) -> np.ndarray:
        """
        Estimate beta (time-shift) and subtype assignment for each patient.

        Uses the JAX implementation (``_transform_jax``) when
        ``transform_backend="jax"`` and the NumPy/LSODA one otherwise.

        Returns a structured array with dtype [('beta', 'f8'), ('subtype', 'i4')].
        """
        if not hasattr(self, "cluster_f") or not hasattr(self, "cluster_cog_a"):
            raise RuntimeError("fit() must be called before transform()")
        if not _shifted:
            X = self._shift_patients(X)
        if self._transform_backend_resolved() == "jax":
            return self._transform_jax(X, use_cognitive_prior)
        return self._transform_numpy(X, use_cognitive_prior)

    def _transform_numpy(
        self, X: list, use_cognitive_prior: bool = True, *, _trajectories=None
    ) -> np.ndarray:
        """NumPy ``transform``. ``_trajectories`` (list of (B, T)) replaces the LSODA solve."""
        lam_cog = self.lambda_cog if use_cognitive_prior else 0.0
        results = np.zeros(len(X), dtype=[("beta", "f8"), ("subtype", "i4")])

        X_preds = (
            self._transform_trajectories_numpy() if _trajectories is None
            else [np.asarray(x, dtype=float) for x in _trajectories]
        )
        beta_grid = make_beta_grid(self.t_max, self.n_beta_grid)
        caps = self._transform_beta_caps(X, beta_grid)

        for idx, p in enumerate(tqdm(X, desc="Estimating beta and subtype assignments")):
            X_obs_i, obs_weight_i = split_observed(p["X_obs"])
            dt_i = p["dt"]
            cog_i = p["cog"]
            if cog_i.ndim == 1:
                cog_i = cog_i.reshape(-1, 1)

            sse_grid = np.zeros((self.n_beta_grid, self.n_subtypes))
            for z in range(self.n_subtypes):
                cog_pred_z = cog_i @ self.cluster_cog_a[z] + self.cluster_cog_b[z]
                for gi, beta_g in enumerate(beta_grid):
                    sse_grid[gi, z] = reconstruction_sse(
                        beta_g, X_obs_i, dt_i, X_preds[z], self.t_span, self.final_s,
                        obs_weight_i=obs_weight_i,
                    )
                    if lam_cog > 0:
                        sse_grid[gi, z] += lam_cog * np.sum(
                            (dt_i + beta_g - cog_pred_z) ** 2
                        )
            if caps is not None:
                sse_grid[beta_grid > caps[idx], :] = np.inf

            best_gi, best_z = np.unravel_index(np.argmin(sse_grid), sse_grid.shape)

            beta_vec, _ = estimate_beta(
                beta_all=np.array([beta_grid[best_gi]], dtype=float),
                X_obs=X_obs_i,
                dt=dt_i,
                ids=np.zeros(len(dt_i), dtype=int),
                cog=cog_i,
                t_span=self.t_span,
                cluster_f=self.cluster_f,
                scalar_K=self.final_scalar_K,
                s=self.final_s,
                assignments=np.array([best_z], dtype=int),
                K=self.K,
                cog_a=self.cluster_cog_a,
                cog_b=self.cluster_cog_b,
                lambda_cog=lam_cog,
                lambda_jsd=0.0,
                lambda_beta=0.0,
                beta_mean=None,
                beta_var=None,
                t_max=self.t_max,
                kappa=self.final_kappa,
                obs_weight=obs_weight_i,
                beta_upper=None if caps is None else caps[idx:idx + 1],
                X_preds=None if _trajectories is None else X_preds,
            )
            results[idx]["beta"] = float(beta_vec[0])
            results[idx]["subtype"] = best_z

        self.beta_val = results["beta"]
        self.transform_assignments = results["subtype"]
        return results

    def transform_soft(
        self,
        X: list,
        use_cognitive_prior: bool = True,
        temperature: float = 1.0,
        per_timepoint: bool = False,
    ) -> np.ndarray:
        """
        Per-patient beta under each subtype, plus softmax probabilities from LSE.

        For every patient and subtype ``z``:
          1. Grid-search beta for that fixed subtype
          2. Refine with ``estimate_beta`` (assignment locked to ``z``)
          3. Record reconstruction (+ optional cog) LSE at the refined beta

        Probabilities are ``softmax(-LSE / temperature)``. ``subtype_best`` is
        the argmax over those probabilities. ``beta`` / ``subtype`` match the
        regular ``transform`` joint grid-search path (best subtype after refine).

        If ``per_timepoint`` is True, beta is still estimated from all visits
        (one shift per subject, so observed ``dt`` spacing is preserved). The
        return array is then expanded to one row per visit. ``t_obs`` is
        ``beta + dt`` at that visit. Visit-level ``proba_*`` use only that
        visit's residual against each subtype trajectory at ``t_obs`` (optional
        cog term uses that visit only). Each visit's ``proba_*`` sum to 1.

        Returns a structured array with fields:
          beta, subtype, subtype_best,
          beta_1 .. beta_K, proba_1 .. proba_K
        (1-indexed subtype columns). With ``per_timepoint=True`` also:
          patient_idx, visit, t_obs, id.

        Uses the JAX implementation (``_transform_soft_jax``) when
        ``transform_backend="jax"`` and the NumPy/LSODA one otherwise.
        """
        if not hasattr(self, "cluster_f") or not hasattr(self, "cluster_cog_a"):
            raise RuntimeError("fit() must be called before transform_soft()")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        X = self._shift_patients(X)
        impl = (
            self._transform_soft_jax
            if self._transform_backend_resolved() == "jax"
            else self._transform_soft_numpy
        )
        return impl(X, use_cognitive_prior, temperature, per_timepoint)

    def _soft_cog_lists(self):
        """Per-subtype cognitive slopes and intercepts as lists."""
        n_subtypes = self.n_subtypes
        if isinstance(self.cluster_cog_a, list):
            return self.cluster_cog_a, self.cluster_cog_b
        return (
            [np.asarray(self.cluster_cog_a[z], dtype=float) for z in range(n_subtypes)],
            [float(self.cluster_cog_b[z]) for z in range(n_subtypes)],
        )

    def _soft_results(self, X: list, per_timepoint: bool) -> np.ndarray:
        """Empty structured output array for ``transform_soft``."""
        dtype = [
            ("beta", "f8"),
            ("subtype", "i4"),
            ("subtype_best", "i4"),
        ]
        if per_timepoint:
            dtype.extend([
                ("patient_idx", "i4"),
                ("visit", "i4"),
                ("t_obs", "f8"),
                ("id", "O"),
            ])
        for z in range(self.n_subtypes):
            dtype.append((f"beta_{z + 1}", "f8"))
        for z in range(self.n_subtypes):
            dtype.append((f"proba_{z + 1}", "f8"))
        n_rows = (
            sum(np.asarray(p["dt"]).reshape(-1).size for p in X)
            if per_timepoint else len(X)
        )
        return np.zeros(n_rows, dtype=dtype)

    @staticmethod
    def _softmax_from_lse(lse, temperature):
        log_p = -np.asarray(lse, dtype=float) / temperature
        log_p -= np.max(log_p)
        proba = np.exp(log_p)
        proba /= proba.sum()
        return proba

    @staticmethod
    def _write_soft_row(results, row, beta, subtype, subtype_best, beta_by_z, proba,
                        patient_idx=None, visit=None, t_obs=None, pid=None):
        results[row]["beta"] = float(beta)
        results[row]["subtype"] = int(subtype)
        results[row]["subtype_best"] = int(subtype_best)
        for z in range(len(beta_by_z)):
            results[row][f"beta_{z + 1}"] = float(beta_by_z[z])
            results[row][f"proba_{z + 1}"] = float(proba[z])
        if patient_idx is not None:
            results[row]["patient_idx"] = int(patient_idx)
            results[row]["visit"] = int(visit)
            results[row]["t_obs"] = float(t_obs)
            results[row]["id"] = pid

    def _transform_soft_numpy(
        self,
        X: list,
        use_cognitive_prior: bool = True,
        temperature: float = 1.0,
        per_timepoint: bool = False,
        *,
        _trajectories=None,
    ) -> np.ndarray:
        """NumPy ``transform_soft``. ``_trajectories`` (list of (B, T)) replaces the LSODA solve."""
        n_patients = len(X)
        n_subtypes = self.n_subtypes
        lam_cog = self.lambda_cog if use_cognitive_prior else 0.0

        # estimate_beta expects list-of-per-subtype arrays (not a stacked ndarray).
        cluster_f = [
            np.ravel(self.cluster_f[z]) for z in range(n_subtypes)
        ]
        cluster_cog_a, cluster_cog_b = self._soft_cog_lists()

        results = self._soft_results(X, per_timepoint)
        beta_val = np.zeros(n_patients, dtype=float)
        transform_assignments = np.zeros(n_patients, dtype=np.int32)

        X_preds = (
            self._transform_trajectories_numpy() if _trajectories is None
            else [np.asarray(x, dtype=float) for x in _trajectories]
        )
        given_preds = None if _trajectories is None else X_preds
        beta_grid = make_beta_grid(self.t_max, self.n_beta_grid)
        caps = self._transform_beta_caps(X, beta_grid)

        def _softmax_from_lse(lse):
            return self._softmax_from_lse(lse, temperature)

        def _write_row(row, *args, **kwargs):
            self._write_soft_row(results, row, *args, **kwargs)

        row = 0
        for idx, p in enumerate(
            tqdm(X, desc="Estimating per-subtype beta and probabilities")
        ):
            X_obs_i, obs_weight_i = split_observed(p["X_obs"])
            dt_i = np.asarray(p["dt"], dtype=float).reshape(-1)
            cog_i = p["cog"]
            if cog_i.ndim == 1:
                cog_i = cog_i.reshape(-1, 1)

            sse_grid = np.zeros((self.n_beta_grid, n_subtypes))
            for z in range(n_subtypes):
                cog_pred_z = cog_i @ cluster_cog_a[z] + cluster_cog_b[z]
                for gi, beta_g in enumerate(beta_grid):
                    sse_grid[gi, z] = reconstruction_sse(
                        beta_g, X_obs_i, dt_i, X_preds[z], self.t_span, self.final_s,
                        obs_weight_i=obs_weight_i,
                    )
                    if lam_cog > 0:
                        sse_grid[gi, z] += lam_cog * np.sum(
                            (dt_i + beta_g - cog_pred_z) ** 2
                        )
            if caps is not None:
                sse_grid[beta_grid > caps[idx], :] = np.inf
            upper_i = None if caps is None else caps[idx:idx + 1]

            beta_by_z = np.zeros(n_subtypes, dtype=float)
            lse_by_z = np.zeros(n_subtypes, dtype=float)
            for z in range(n_subtypes):
                best_gi = int(np.argmin(sse_grid[:, z]))
                beta_vec, lse_z = estimate_beta(
                    beta_all=np.array([beta_grid[best_gi]], dtype=float),
                    X_obs=X_obs_i,
                    dt=dt_i,
                    ids=np.zeros(len(dt_i), dtype=int),
                    cog=cog_i,
                    t_span=self.t_span,
                    cluster_f=cluster_f,
                    scalar_K=self.final_scalar_K,
                    s=self.final_s,
                    assignments=np.array([z], dtype=int),
                    K=self.K,
                    cog_a=cluster_cog_a,
                    cog_b=cluster_cog_b,
                    lambda_cog=lam_cog,
                    lambda_jsd=0.0,
                    lambda_beta=0.0,
                    beta_mean=None,
                    beta_var=None,
                    t_max=self.t_max,
                    kappa=self.final_kappa,
                    ode_method=self.ode_method,
                    obs_weight=obs_weight_i,
                    beta_upper=upper_i,
                    X_preds=given_preds,
                )
                beta_by_z[z] = float(beta_vec[0])
                lse_by_z[z] = float(lse_z)

            # Regular transform: joint best (grid) then refine under that subtype.
            best_gi, best_z = np.unravel_index(np.argmin(sse_grid), sse_grid.shape)
            beta_vec, _ = estimate_beta(
                beta_all=np.array([beta_grid[best_gi]], dtype=float),
                X_obs=X_obs_i,
                dt=dt_i,
                ids=np.zeros(len(dt_i), dtype=int),
                cog=cog_i,
                t_span=self.t_span,
                cluster_f=cluster_f,
                scalar_K=self.final_scalar_K,
                s=self.final_s,
                assignments=np.array([best_z], dtype=int),
                K=self.K,
                cog_a=cluster_cog_a,
                cog_b=cluster_cog_b,
                lambda_cog=lam_cog,
                lambda_jsd=0.0,
                lambda_beta=0.0,
                beta_mean=None,
                beta_var=None,
                t_max=self.t_max,
                kappa=self.final_kappa,
                ode_method=self.ode_method,
                obs_weight=obs_weight_i,
                beta_upper=upper_i,
                X_preds=given_preds,
            )
            beta_joint = float(beta_vec[0])
            best_z = int(best_z)
            beta_val[idx] = beta_joint
            transform_assignments[idx] = best_z
            pid = p.get("id", idx)

            if per_timepoint:
                n_t = len(dt_i)
                t_obs = dt_i + beta_joint
                for t in range(n_t):
                    lse_t = np.zeros(n_subtypes, dtype=float)
                    for z in range(n_subtypes):
                        lse_t[z] = reconstruction_sse(
                            beta_joint,
                            X_obs_i[t:t + 1],
                            dt_i[t:t + 1],
                            X_preds[z],
                            self.t_span,
                            self.final_s,
                            obs_weight_i=obs_weight_i[t:t + 1],
                        )
                        if lam_cog > 0:
                            cog_pred_tz = float(
                                cog_i[t] @ cluster_cog_a[z] + cluster_cog_b[z]
                            )
                            lse_t[z] += lam_cog * (t_obs[t] - cog_pred_tz) ** 2
                    proba = _softmax_from_lse(lse_t)
                    _write_row(
                        row,
                        beta_joint,
                        best_z,
                        int(np.argmax(proba)),
                        beta_by_z,
                        proba,
                        patient_idx=idx,
                        visit=t,
                        t_obs=t_obs[t],
                        pid=pid,
                    )
                    row += 1
            else:
                proba = _softmax_from_lse(lse_by_z)
                _write_row(
                    idx,
                    beta_joint,
                    best_z,
                    int(np.argmax(proba)),
                    beta_by_z,
                    proba,
                )

        self.beta_val = beta_val
        self.transform_assignments = transform_assignments
        self.assignment_probabilities_ = np.column_stack(
            [results[f"proba_{z + 1}"] for z in range(n_subtypes)]
        )
        return results

    def _transform_jax_setup(self, X: list, use_cognitive_prior: bool, _trajectories):
        """Stack and pack new patients, solve trajectories, and build the beta grid."""
        from . import jax_core as jc
        from .optimizer_jax import _stack_cog, solve_trajectories_jax, trajectories_to_jax

        n_patients = len(X)
        dts = [np.asarray(p["dt"], dtype=float).reshape(-1) for p in X]
        X_raw = np.vstack([np.asarray(p["X_obs"], dtype=float) for p in X])
        dt = np.concatenate(dts)
        ids = np.concatenate([np.full(len(d), i) for i, d in enumerate(dts)])
        cog = np.vstack([ensure_2d_cog(p["cog"], len(d)) for p, d in zip(X, dts)])
        X_obs, obs_weight = split_observed(X_raw)
        data, meta = jc.pack_data(
            X_raw, None, dt, ids, cog, n_patients,
            pad_rows=getattr(self, "transform_pad_rows", None),
            pad_patients=getattr(self, "transform_pad_patients", None),
        )
        if _trajectories is None:
            traj = solve_trajectories_jax(
                self.cluster_f, self.final_kappa, self.final_scalar_K, self.K, self.t_span
            )
        else:
            traj = trajectories_to_jax(_trajectories)
        cog_a, cog_b = self._soft_cog_lists()
        A, b = _stack_cog(cog_a, cog_b, self.n_subtypes, data.cog.shape[1])
        beta_grid = make_beta_grid(self.t_max, self.n_beta_grid)
        _, h, _ = jc.grid_from_t_span(self.t_span)
        return dict(
            n_patients=n_patients, n_rows=len(dt), dts=dts,
            X_obs=X_obs, obs_weight=obs_weight, dt=dt, ids=ids, cog=cog,
            packed=(data, meta), traj=traj, cog_a=cog_a, cog_b=cog_b,
            A=jc.f64(A), b=jc.f64(b), s=jc.f64(np.ravel(self.final_s)), h=jc.f64(h),
            lam_cog=self.lambda_cog if use_cognitive_prior else 0.0,
            beta_grid=beta_grid, caps=self._transform_beta_caps(X, beta_grid),
        )

    def _transform_jax_select(self, st: dict):
        """Joint (grid beta, subtype) selection with ``joint_grid_select`` (grid only)."""
        from . import jax_core as jc
        from .optimizer_jax import _grid_candidates, _pad_upper

        data, meta = st["packed"]
        n, P = st["n_patients"], meta["P"]
        best_beta, best_z = jc._joint_grid_select_jit(
            st["traj"], st["s"], _grid_candidates(st["beta_grid"], P), data,
            st["A"], st["b"], jc.f64(st["lam_cog"]),
            _pad_upper(st["caps"], n, P, self.t_max), st["h"],
        )
        return (
            np.asarray(best_beta, dtype=np.float64)[:n].copy(),
            np.asarray(best_z)[:n].astype(np.int32),
        )

    def _transform_jax_refine(self, st: dict, beta0: np.ndarray, assignments: np.ndarray):
        """One joint L-BFGS-B refinement of every new patient's beta, subtypes fixed."""
        from .optimizer_jax import estimate_beta_jax

        beta, _ = estimate_beta_jax(
            beta_all=beta0,
            X_obs=st["X_obs"],
            dt=st["dt"],
            ids=st["ids"],
            cog=st["cog"],
            t_span=self.t_span,
            trajectories=st["traj"],
            s=self.final_s,
            assignments=assignments,
            cog_a=st["cog_a"],
            cog_b=st["cog_b"],
            lambda_cog=st["lam_cog"],
            lambda_jsd=0.0,
            lambda_beta=0.0,
            beta_mean=None,
            beta_var=None,
            t_max=self.t_max,
            strict_tol=True,
            obs_weight=st["obs_weight"],
            beta_upper=st["caps"],
            packed=st["packed"],
            maxiter=500,
        )
        return beta

    def _transform_jax(
        self, X: list, use_cognitive_prior: bool = True, *, _trajectories=None
    ) -> np.ndarray:
        """
        JAX ``transform``: joint grid selection, then one joint beta refinement.

        Subtype and starting beta come from ``jax_core.joint_grid_select`` over
        ``make_beta_grid(t_max, n_beta_grid)`` (candidates above a patient's
        cap are skipped when ``cap_beta_by_followup``). Every patient's beta is
        then refined in a single L-BFGS-B call with subtypes fixed,
        ``lambda_jsd = lambda_beta = 0``, bounds ``(0, cap_i)`` (or
        ``(0, t_max)``), strict tolerances and ``maxiter = 500``. That loss is
        a sum of independent per-patient terms, so the joint call is
        equivalent to per-patient refinement up to the stopping tolerance; it
        replaces n separate scipy calls with one.

        ``_trajectories`` (list of (B, T) or a (Z, T, B) array) replaces the
        RK4 solve, so tests can compare backends on identical trajectories.
        """
        st = self._transform_jax_setup(X, use_cognitive_prior, _trajectories)
        beta0, best_z = self._transform_jax_select(st)
        beta = self._transform_jax_refine(st, beta0, best_z)

        results = np.zeros(len(X), dtype=[("beta", "f8"), ("subtype", "i4")])
        results["beta"] = beta
        results["subtype"] = best_z
        self.beta_val = results["beta"]
        self.transform_assignments = results["subtype"]
        return results

    def _transform_soft_jax(
        self,
        X: list,
        use_cognitive_prior: bool = True,
        temperature: float = 1.0,
        per_timepoint: bool = False,
        *,
        _trajectories=None,
    ) -> np.ndarray:
        """
        JAX ``transform_soft`` with Z + 1 joint refinements instead of n * (Z + 1).

        For each subtype z, every patient is assigned to z and refined jointly
        (as in ``_transform_jax``) from that subtype's best grid point; the
        per-patient reconstruction LSE at the refined betas is column z of
        ``sse_matrix_jax`` (no cognitive term, as ``estimate_beta``'s returned
        LSE in the NumPy version). ``beta`` / ``subtype`` come from the same
        joint selection and refinement as ``_transform_jax``. Output fields,
        probabilities, ``subtype_best`` and the ``per_timepoint`` expansion
        match ``_transform_soft_numpy``; per-visit probabilities use
        ``sse_rows_jax`` at the joint beta.
        """
        from . import jax_core as jc
        from .optimizer_jax import _grid_candidates

        st = self._transform_jax_setup(X, use_cognitive_prior, _trajectories)
        data, meta = st["packed"]
        n, P, Z = st["n_patients"], meta["P"], self.n_subtypes
        beta_grid = st["beta_grid"]

        losses = np.array(jc._joint_grid_losses_jit(
            st["traj"], st["s"], _grid_candidates(beta_grid, P), data,
            st["A"], st["b"], jc.f64(st["lam_cog"]), st["h"],
        ), dtype=np.float64)[:, :n, :]
        if st["caps"] is not None:
            losses[beta_grid[:, None] > st["caps"][None, :], :] = np.inf
        best_gi = np.argmin(losses, axis=0)  # (n, Z)

        beta_by_z = np.zeros((n, Z))
        lse_by_z = np.zeros((n, Z))
        for z in range(Z):
            beta_z = self._transform_jax_refine(
                st, beta_grid[best_gi[:, z]], np.full(n, z, dtype=np.int32)
            )
            beta_by_z[:, z] = beta_z
            lse = jc._sse_matrix_jit(
                st["traj"], st["s"], jc.pad_beta(beta_z, P), data,
                st["A"], st["b"], jc.f64(0.0), st["h"],
            )
            lse_by_z[:, z] = np.asarray(lse, dtype=np.float64)[:n, z]

        beta0, best_z = self._transform_jax_select(st)
        beta_joint = self._transform_jax_refine(st, beta0, best_z)

        results = self._soft_results(X, per_timepoint)
        if per_timepoint:
            row_sse = np.asarray(jc._sse_rows_jit(
                st["traj"], st["s"], jc.pad_beta(beta_joint, P), data,
                st["A"], st["b"], jc.f64(st["lam_cog"]), st["h"],
            ), dtype=np.float64)[: st["n_rows"]]
            row = 0
            for idx, (p, dt_i) in enumerate(zip(X, st["dts"])):
                pid = p.get("id", idx)
                for t in range(len(dt_i)):
                    proba = self._softmax_from_lse(row_sse[row], temperature)
                    self._write_soft_row(
                        results, row, beta_joint[idx], best_z[idx], int(np.argmax(proba)),
                        beta_by_z[idx], proba, patient_idx=idx, visit=t,
                        t_obs=dt_i[t] + beta_joint[idx], pid=pid,
                    )
                    row += 1
        else:
            for idx in range(n):
                proba = self._softmax_from_lse(lse_by_z[idx], temperature)
                self._write_soft_row(
                    results, idx, beta_joint[idx], best_z[idx], int(np.argmax(proba)),
                    beta_by_z[idx], proba,
                )

        self.beta_val = beta_joint
        self.transform_assignments = best_z.astype(np.int32)
        self.assignment_probabilities_ = np.column_stack(
            [results[f"proba_{z + 1}"] for z in range(Z)]
        )
        return results

    def _transform_beta_caps(self, X: list, beta_grid: np.ndarray):
        """Per-patient beta caps for ``transform``, or None when the cap is off."""
        if not getattr(self, "cap_beta_by_followup", False):
            return None
        caps = beta_upper_from_followup([p["dt"] for p in X], self.t_max)
        check_beta_grid_within_cap(beta_grid, caps)
        return caps

    def score(self, X: list, y=None) -> float:
        """Negative per-subtype reconstruction LSE on X (higher = better, for sklearn CV).

        Each patient is scored on the trajectory of the subtype ``transform`` assigns,
        not on a single shared curve.
        """
        X = self._shift_patients(X)
        transformed = self.transform(X, _shifted=True)
        return -self._compute_val_score(
            X, transformed["beta"], transformed["subtype"]
        )

    def compute_subtype_mapping(self, true_f_list, verbose=True):
        """Compute fitted→true subtype mapping by nearest-f matching."""
        from .utils import get_subtype_mapping_from_f

        self.subtype_mapping = get_subtype_mapping_from_f(self.cluster_f, true_f_list)
        if verbose:
            print(f"\nSubtype mapping (fitted → true): {self.subtype_mapping}")
            for fitted in range(len(self.subtype_mapping)):
                print(
                    f"  Fitted subtype {fitted} → True subtype "
                    f"{int(self.subtype_mapping[fitted])}"
                )
        return self.subtype_mapping

    def _prepare_data(self, X):
        """Flatten patient dicts into stacked numpy arrays."""
        n_patients = len(X)
        n_biomarkers = X[0]["X_obs"].shape[1]

        X_obs_list, dt_list, ids_list, cog_list, ibeta_list = [], [], [], [], []
        for i, patient in enumerate(X):
            n = len(patient["dt"])
            X_obs_list.append(patient["X_obs"])
            dt_list.append(patient["dt"])
            ids_list.append(np.full(n, i))
            cog_list.append(ensure_2d_cog(patient["cog"], n))
            if "initial_beta" in patient:
                ibeta_list.append(patient["initial_beta"])

        X_obs_raw = np.vstack(X_obs_list)
        dt = np.concatenate(dt_list)
        ids = np.concatenate(ids_list)
        cog = np.vstack(cog_list)

        if not (len(dt) == len(ids) == X_obs_raw.shape[0] == cog.shape[0]):
            raise ValueError(
                f"Stacked shapes disagree: X_obs={X_obs_raw.shape}, dt={dt.shape}, "
                f"ids={ids.shape}, cog={cog.shape}"
            )
        if cog.ndim == 1:
            cog = np.atleast_2d(cog).T

        n_obs_per_b = np.sum(np.isfinite(X_obs_raw), axis=0)
        if np.any(n_obs_per_b < 2):
            bad = np.where(n_obs_per_b < 2)[0]
            raise ValueError(
                f"Biomarker(s) {bad.tolist()} have fewer than 2 observed values "
                f"cohort-wide; null variance for BIC cannot be estimated."
            )
        var_per_biomarker_null = np.maximum(
            np.nanvar(X_obs_raw, axis=0, ddof=1), 1e-12
        )
        s_lower = None
        if self.s_lower_percentile is not None:
            s_lower = np.maximum(
                np.nanpercentile(X_obs_raw, float(self.s_lower_percentile), axis=0), 0.0
            )
        X_obs, obs_weight = split_observed(X_obs_raw)
        n_obs = int(obs_weight.sum())

        for i in range(n_patients):
            if obs_weight[ids == i].sum() == 0:
                warnings.warn(
                    f"Patient index {i} has zero observed biomarker entries; "
                    f"beta is unidentified from reconstruction alone.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        if self.initial_beta is not None:
            beta = np.asarray(self.initial_beta, dtype=float).copy()
            if beta.shape != (n_patients,):
                raise ValueError(
                    f"initial_beta must have shape ({n_patients},), got {beta.shape}"
                )
        elif ibeta_list:
            if len(ibeta_list) != n_patients:
                raise ValueError(
                    f"initial_beta on patients: {len(ibeta_list)} values, "
                    f"expected {n_patients}"
                )
            beta = np.array(ibeta_list)
        else:
            beta = initialize_beta(
                ids=np.arange(n_patients), beta_range=(0, self.t_max), rng=self.rng
            )

        max_val = np.max(beta)
        beta[(beta > max_val - 1) & (beta < max_val)] -= 2

        if self.cap_beta_by_followup:
            beta_upper = beta_upper_from_followup(dt_list, self.t_max)
            clipped = (beta < 0) | (beta > beta_upper)
            n_clipped = int(np.sum(clipped))
            if n_clipped > 0:
                warnings.warn(
                    f"cap_beta_by_followup: clipped the initial beta of {n_clipped} "
                    f"patient(s) to [0, t_max - max(dt_i)].",
                    RuntimeWarning,
                    stacklevel=2,
                )
                beta = np.clip(beta, 0.0, beta_upper)
        else:
            beta_upper = np.full(n_patients, float(self.t_max))

        beta_mean = np.mean(beta)
        beta_var = max(np.var(beta), 1e-8)

        return dict(
            X_obs=X_obs,
            dt=dt,
            ids=ids,
            cog=cog,
            n_patients=n_patients,
            n_biomarkers=n_biomarkers,
            initial_beta=beta,
            beta_upper=beta_upper,
            s_lower=s_lower,
            beta_mean=beta_mean,
            beta_var=beta_var,
            n_obs=n_obs,
            var_per_biomarker_null=var_per_biomarker_null,
            obs_weight=obs_weight,
        )

    def _initialize_state(self, flat, X):
        """Build initial EM state dict (params, histories, loop control)."""
        n_patients = flat["n_patients"]
        n_biomarkers = flat["n_biomarkers"]
        n_cog = flat["cog"].shape[1]
        rng = self.rng

        if self.initial_s is not None:
            s = np.asarray(self.initial_s, dtype=float).ravel().copy()
            if s.shape != (n_biomarkers,):
                raise ValueError(f"initial_s must have shape ({n_biomarkers},), got {s.shape}")
        else:
            s = rng.uniform(0.1, 3, size=n_biomarkers)

        s_mask_bool = resolve_s_mask(n_biomarkers, self.fit_s, self.s_mask)
        if flat["s_lower"] is not None:
            s = np.maximum(s, flat["s_lower"])
        s[~s_mask_bool] = 1.0
        self.s_mask_ = s_mask_bool
        self.s_lower_ = None if flat["s_lower"] is None else flat["s_lower"].copy()

        scalar_K = (
            float(self.initial_scalar_K)
            if self.initial_scalar_K is not None
            else 1.0
        )

        if self.initial_kappa is not None:
            kappa = np.asarray(self.initial_kappa, dtype=float).ravel().copy()
            if kappa.shape != (n_biomarkers,):
                raise ValueError(
                    f"initial_kappa must have shape ({n_biomarkers},), got {kappa.shape}"
                )
        else:
            kappa = rng.uniform(0.0, 1.0, size=n_biomarkers)

        cluster_f, cluster_cog_a, cluster_cog_b = [], [], []
        for z in range(self.n_subtypes):
            if self.initial_f is not None:
                f0 = np.asarray(self.initial_f)
                if f0.ndim == 2 and f0.shape[0] == self.n_subtypes:
                    cluster_f.append(np.ravel(f0[z]).copy())
                else:
                    cluster_f.append(
                        np.ravel(f0).copy() + rng.uniform(-0.01, 0.01, size=n_biomarkers)
                    )
            else:
                cluster_f.append(rng.uniform(0, 0.1, size=n_biomarkers))

            if self.initial_cluster_cog_a is not None:
                src = np.asarray(self.initial_cluster_cog_a, dtype=float)
                cluster_cog_a.append(np.ravel(src[z] if src.ndim == 2 else src).copy())
            else:
                cluster_cog_a.append(np.ones(n_cog))

            if self.initial_cluster_cog_b is not None:
                src_b = np.asarray(self.initial_cluster_cog_b, dtype=float).ravel()
                cluster_cog_b.append(float(src_b[0] if src_b.size == 1 else src_b[z]))
            else:
                cluster_cog_b.append(0.0)

        if self.initial_assignments is not None:
            if len(self.initial_assignments) != n_patients:
                raise ValueError(
                    f"initial_assignments length ({len(self.initial_assignments)}) "
                    f"must match number of patients ({n_patients})"
                )
            if np.any(self.initial_assignments < 0) or np.any(
                self.initial_assignments >= self.n_subtypes
            ):
                raise ValueError(
                    f"initial_assignments must be in [0, {self.n_subtypes})"
                )
            assignments = self.initial_assignments.copy()
        elif all("initial_subtype" in p for p in X):
            assignments = np.array([p["initial_subtype"] for p in X], dtype=int)
            if np.any(assignments < 0) or np.any(assignments >= self.n_subtypes):
                raise ValueError(
                    f"initial_subtype values must be in [0, {self.n_subtypes})"
                )
        else:
            assignments = rng.integers(0, self.n_subtypes, size=n_patients)

        beta = flat["initial_beta"]
        c = self._initial_c(n_biomarkers)
        if self.fit_intercept:
            cb = self._resolve_c_bounds(n_biomarkers)
            c = np.clip(c, cb[:, 0], cb[:, 1])
        X_preds = precompute_trajectories(
            cluster_f, self.K, self.t_span, scalar_K, kappa,
            ode_method=self.ode_method,
        )
        sse_mat = sse_matrix(
            flat["X_obs"] - c[None, :],
            flat["dt"],
            flat["ids"],
            beta,
            X_preds,
            s,
            self.t_span,
            cog=flat["cog"],
            cluster_cog_a=cluster_cog_a,
            cluster_cog_b=cluster_cog_b,
            lambda_cog=self.lambda_cog,
            obs_weight=flat["obs_weight"],
        )
        initial_lse = float(np.sum(sse_mat[np.arange(n_patients), assignments]))

        rep_theta = np.concatenate([np.ravel(cluster_f[0]), s, [scalar_K]])
        T = self.max_iter + 1

        theta_hist = np.zeros((rep_theta.shape[0], T))
        beta_hist = np.zeros((n_patients, T))
        kappa_hist = np.zeros((n_biomarkers, T))
        lse_hist = np.zeros(T)
        cog_reg_hist = np.zeros((self.n_subtypes, n_cog + 1, T))
        assign_hist = np.zeros((n_patients, T), dtype=int)

        theta_hist[:, 0] = rep_theta
        beta_hist[:, 0] = beta
        kappa_hist[:, 0] = kappa
        lse_hist[0] = initial_lse
        assign_hist[:, 0] = assignments
        for z in range(self.n_subtypes):
            cog_reg_hist[z, :, 0] = np.concatenate(
                [cluster_cog_a[z], [cluster_cog_b[z]]]
            )

        iter_time_hist = np.zeros(T, dtype=float)
        solver_hist = [""] * T
        assign_change_hist = np.zeros(T, dtype=int)

        beta_upper = flat["beta_upper"]
        subtype_counts_hist = np.zeros((self.n_subtypes, T), dtype=int)
        beta_at_upper_hist = np.zeros(T, dtype=float)
        estep_moves_hist = np.zeros(T, dtype=int)
        subtype_counts_hist[:, 0] = np.bincount(assignments, minlength=self.n_subtypes)
        beta_at_upper_hist[0] = np.mean(beta >= beta_upper - _AT_UPPER_TOL)

        return dict(
            current_beta=beta.copy(),
            current_s=s,
            current_c=c,
            c_hist=[c.copy()],
            current_kappa=kappa,
            current_scalar_K=scalar_K,
            cluster_f=cluster_f,
            cluster_cog_a=cluster_cog_a,
            cluster_cog_b=cluster_cog_b,
            assignments=assignments,
            beta_mean=flat["beta_mean"],
            beta_var=flat["beta_var"],
            solver_phases=list(self.theta_solver_stages),
            theta_hist=theta_hist,
            beta_hist=beta_hist,
            kappa_hist=kappa_hist,
            lse_hist=lse_hist,
            cog_reg_hist=cog_reg_hist,
            assign_hist=assign_hist,
            iter_time_hist=iter_time_hist,
            solver_hist=solver_hist,
            assign_change_hist=assign_change_hist,
            subtype_counts_hist=subtype_counts_hist,
            beta_at_upper_hist=beta_at_upper_hist,
            estep_moves_hist=estep_moves_hist,
            estep_beta_moves=0,
            beta_upper=beta_upper,
            best_lse=np.inf,
            loop_iter=0,
            final_lse=initial_lse,
            n_biomarkers=n_biomarkers,
        )

    def _run_em_loop(self, state, flat, X):
        """Outer EM loop with checkpointing and θ-solver escalation."""
        if self.backend == "jax":
            state["solver_phases"] = ["jax"]
        if self.verbose >= 1:
            pbar = tqdm(total=self.max_iter)
        else:

            class _NoPbar:
                def update(self, n=1):
                    pass

            pbar = _NoPbar()

        while state["loop_iter"] < self.max_iter:
            iter_start = time.perf_counter()
            hist_idx = state["loop_iter"] + 1
            ck = self._checkpoint(state)
            solver_phase_idx = 0

            while True:
                self._restore_checkpoint(state, ck)
                current_solver = state["solver_phases"][solver_phase_idx]
                lse = self._em_step(state, flat, current_solver)

                acceptable = _lse_improvement_acceptable(
                    state["best_lse"],
                    lse,
                    self.epsilon,
                    self.relative_tolerance,
                )
                last_stage = solver_phase_idx >= len(state["solver_phases"]) - 1

                if acceptable or last_stage:
                    if not acceptable and last_stage and self.verbose >= 2:
                        print(
                            f"EM iter {state['loop_iter']}: LSE improvement still below "
                            f"threshold after {current_solver}; accepting anyway."
                        )
                    break

                solver_phase_idx += 1
                if self.verbose >= 2:
                    print(
                        f"EM iter {state['loop_iter']}: LSE gain too small with "
                        f"{state['solver_phases'][solver_phase_idx - 1]}; "
                        f"retrying with {state['solver_phases'][solver_phase_idx]}"
                    )

            if np.isfinite(state["best_lse"]) and lse >= state["best_lse"]:
                if self.verbose >= 1:
                    print(
                        f"EM: LSE did not improve vs best ({state['best_lse']:.6g}); "
                        f"trial={lse:.6g}. Exiting early at outer iter "
                        f"{state['loop_iter']}."
                    )
                self._restore_checkpoint(state, ck)
                state["final_lse"] = float(state["best_lse"])
                break

            state["iter_time_hist"][hist_idx] = time.perf_counter() - iter_start
            state["solver_hist"][hist_idx] = current_solver
            state["assign_change_hist"][hist_idx] = int(
                np.sum(state["assignments"] != ck["assignments"])
            )

            self._record_history(state, lse, hist_idx)
            state["best_lse"] = (
                lse
                if not np.isfinite(state["best_lse"])
                else min(state["best_lse"], lse)
            )
            state["final_lse"] = lse
            state["loop_iter"] += 1
            pbar.update(1)
            if self.n_anneal_iters > 0 and state["loop_iter"] <= self.n_anneal_iters:
                decay = self.anneal_decay ** (state["loop_iter"] - 1)
                kappa_noise = self.rng.uniform(
                    0, self.kappa_anneal_strength * decay,
                    size=state["current_kappa"].shape,
                )
                state["current_kappa"] = state["current_kappa"] + kappa_noise
                for z in range(self.n_subtypes):
                    f_scale = 1.0 + self.rng.uniform(
                        -self.f_anneal_strength * decay,
                        self.f_anneal_strength * decay,
                        size=state["cluster_f"][z].shape,
                    )
                    state["cluster_f"][z] = np.maximum(
                        state["cluster_f"][z] * f_scale, 0.0
                    )
            self._save_fit_checkpoint(state, flat, X, fit_complete=False)

    def _em_step(self, state, flat, current_solver):
        """One EM iteration (global θ, E-step, cog regression, cluster f, beta)."""
        if self.backend == "jax":
            return self._em_step_jax(state, flat)
        X_obs = flat["X_obs"]
        dt = flat["dt"]
        ids = flat["ids"]
        cog = flat["cog"]
        unique_ids = np.unique(ids)
        state["estep_beta_moves"] = 0

        # globals_trajectories is solved at the converged globals with this same
        # cluster_f, so the E-step below (hard, jittered or joint-grid) can reuse
        # it. Ordering dependency: the per-subtype fit_theta_cluster loop must
        # stay AFTER the E-step, or this cache goes stale silently (wrong
        # assignments, not a crash).
        (
            state["current_s"],
            state["current_kappa"],
            state["current_scalar_K"],
            globals_trajectories,
        ) = (
            fit_theta_globals(
                X_obs=X_obs,
                dt_obs=dt,
                ids=ids,
                K=self.K,
                t_span=self.t_span,
                method=current_solver,
                beta_pred=state["current_beta"],
                s_guess=state["current_s"],
                kappa_guess=state["current_kappa"],
                scalar_K_guess=state["current_scalar_K"],
                lambda_s=0.0,
                lambda_scalar=self.lambda_scalar,
                lambda_kappa=self.lambda_kappa,
                scalar_K_center=self.scalar_K_center,
                fit_s=self.fit_s,
                s_mask=self.s_mask_,
                s_lower=flat["s_lower"],
                assignments=state["assignments"],
                cluster_f=state["cluster_f"],
                strict_tol=self.strict_tol,
                ode_method=self.ode_method,
                obs_weight=flat["obs_weight"],
            )
        )

        loop_iter = state["loop_iter"]
        use_jitter = self._use_jitter(loop_iter)
        if use_jitter:
            state["assignments"], probs = update_assignments_jitter(
                X_obs,
                dt,
                ids,
                cog,
                state["current_beta"],
                state["cluster_f"],
                state["current_scalar_K"],
                state["current_kappa"],
                state["current_s"],
                self.K,
                self.t_span,
                state["cluster_cog_a"],
                state["cluster_cog_b"],
                self.lambda_cog,
                temperature=self.jitter_temperature,
                rng=self.rng,
                ode_method=self.ode_method,
                obs_weight=flat["obs_weight"],
                X_preds=globals_trajectories,
            )
            self.assignment_probabilities_ = probs
        elif self.estep_mode == "joint_grid":
            state["assignments"], beta_start = update_assignments_joint_grid(
                X_obs,
                dt,
                ids,
                cog,
                state["current_beta"],
                state["cluster_f"],
                state["current_scalar_K"],
                state["current_kappa"],
                state["current_s"],
                self.K,
                self.t_span,
                state["cluster_cog_a"],
                state["cluster_cog_b"],
                self.lambda_cog,
                self._estep_beta_grid(),
                beta_upper=flat["beta_upper"],
                ode_method=self.ode_method,
                obs_weight=flat["obs_weight"],
                X_preds=globals_trajectories,
            )
            self._apply_beta_start(state, beta_start)
        else:
            state["assignments"] = update_assignments_hard(
                X_obs,
                dt,
                ids,
                cog,
                state["current_beta"],
                state["cluster_f"],
                state["current_scalar_K"],
                state["current_kappa"],
                state["current_s"],
                self.K,
                self.t_span,
                state["cluster_cog_a"],
                state["cluster_cog_b"],
                self.lambda_cog,
                ode_method=self.ode_method,
                obs_weight=flat["obs_weight"],
                X_preds=globals_trajectories,
            )

        self._update_cog_regression(state, flat)

        for z in range(self.n_subtypes):
            cl_idx = np.where(state["assignments"] == z)[0]
            if len(cl_idx) == 0:
                if self.verbose >= 2:
                    print(f"Warning: Cluster {z} is empty at iteration {loop_iter}")
                continue

            cl_pids = unique_ids[cl_idx]
            obs_mask = np.isin(ids, cl_pids)
            ids_local = self._reindex_ids(ids[obs_mask])

            state["cluster_f"][z] = np.ravel(
                fit_theta_cluster(
                    X_obs=X_obs[obs_mask, :],
                    dt_obs=dt[obs_mask],
                    ids=ids_local,
                    K=self.K,
                    t_span=self.t_span,
                    s=state["current_s"],
                    scalar_K=state["current_scalar_K"],
                    lambda_f=self.lambda_f,
                    method=current_solver,
                    beta_pred=state["current_beta"][cl_idx],
                    f_guess=np.ravel(state["cluster_f"][z]),
                    rng=self.rng,
                    kappa=state["current_kappa"],
                    strict_tol=self.strict_tol,
                    ode_method=self.ode_method,
                    obs_weight=flat["obs_weight"][obs_mask, :],
                )
            )

        state["current_beta"], lse = estimate_beta(
            beta_all=state["current_beta"],
            X_obs=X_obs,
            dt=dt,
            ids=ids,
            cog=cog,
            t_span=self.t_span,
            cluster_f=state["cluster_f"],
            scalar_K=state["current_scalar_K"],
            s=state["current_s"],
            assignments=state["assignments"],
            K=self.K,
            cog_a=state["cluster_cog_a"],
            cog_b=state["cluster_cog_b"],
            lambda_cog=self.lambda_cog,
            lambda_jsd=self.lambda_jsd,
            lambda_beta=self.lambda_beta,
            beta_mean=state["beta_mean"],
            beta_var=state["beta_var"],
            t_max=self.t_max,
            kappa=state["current_kappa"],
            strict_tol=self.strict_tol,
            ode_method=self.ode_method,
            obs_weight=flat["obs_weight"],
            beta_upper=flat["beta_upper"],
        )

        return lse

    @staticmethod
    def _apply_beta_start(state, beta_start):
        """Adopt the joint-grid E-step's beta before cog regression, f and beta steps."""
        state["estep_beta_moves"] = int(np.sum(beta_start != state["current_beta"]))
        state["current_beta"] = np.asarray(beta_start, dtype=float).copy()

    def _use_jitter(self, loop_iter):
        """Whether this outer iteration uses the jittered (sampled) E-step."""
        return (
            self.assignments_jitter
            and self.jitter_iter > 0
            and (loop_iter % self.jitter_iter == 0)
            and loop_iter > 0
        )

    def _update_cog_regression(self, state, flat):
        """Per-subtype cognitive regression on the current assignments and beta."""
        dt = flat["dt"]
        ids = flat["ids"]
        cog = flat["cog"]
        unique_ids = np.unique(ids)
        for z in range(self.n_subtypes):
            cl_idx = np.where(state["assignments"] == z)[0]
            if len(cl_idx) == 0:
                continue
            cl_pids = unique_ids[cl_idx]
            obs_mask = np.isin(ids, cl_pids)
            cog_a, cog_b = fit_linear_cog_regression_multi(
                cog[obs_mask],
                dt[obs_mask],
                state["current_beta"][cl_idx],
                ids[obs_mask],
            )
            state["cluster_cog_a"][z] = cog_a
            state["cluster_cog_b"][z] = cog_b

    def _em_step_jax(self, state, flat):
        """One EM iteration on the JAX backend (same step order as ``_em_step``)."""
        from .optimizer_jax import (
            estimate_beta_jax,
            fit_theta_cluster_all_jax,
            fit_theta_globals_jax,
            solve_trajectories_jax,
            update_assignments_hard_jax,
            update_assignments_joint_grid_jax,
            update_assignments_jitter_jax,
        )

        X_obs = flat["X_obs"]
        dt = flat["dt"]
        ids = flat["ids"]
        cog = flat["cog"]
        packed = flat["jax"]["packed"]
        state["estep_beta_moves"] = 0

        if self.fit_intercept:
            return self._em_step_jax_intercept(state, flat)

        # globals_trajectories is solved at the converged globals with this same
        # cluster_f, so the E-step below (hard, jittered or joint-grid) can reuse
        # it. Ordering dependency: the joint fit_theta_cluster_all_jax call must
        # stay AFTER the E-step, or this cache goes stale silently (wrong
        # assignments, not a crash).
        (
            state["current_s"],
            state["current_kappa"],
            state["current_scalar_K"],
            globals_trajectories,
        ) = fit_theta_globals_jax(
            X_obs=X_obs,
            dt_obs=dt,
            ids=ids,
            K=self.K,
            t_span=self.t_span,
            beta_pred=state["current_beta"],
            s_guess=state["current_s"],
            kappa_guess=state["current_kappa"],
            scalar_K_guess=state["current_scalar_K"],
            lambda_s=0.0,
            lambda_scalar=self.lambda_scalar,
            lambda_kappa=self.lambda_kappa,
            scalar_K_center=self.scalar_K_center,
            fit_s=self.fit_s,
            s_mask=self.s_mask_,
            s_lower=flat["s_lower"],
            assignments=state["assignments"],
            cluster_f=state["cluster_f"],
            strict_tol=self.strict_tol,
            obs_weight=flat["obs_weight"],
            packed=packed,
        )

        return self._em_step_jax_after_globals(state, flat, globals_trajectories)

    def _em_step_jax_after_globals(self, state, flat, globals_trajectories):
        """E-step, cog regression, cluster f and beta steps of a JAX EM iteration."""
        from .optimizer_jax import (
            estimate_beta_jax,
            fit_theta_cluster_all_jax,
            solve_trajectories_jax,
            update_assignments_hard_jax,
            update_assignments_joint_grid_jax,
            update_assignments_jitter_jax,
        )

        X_obs = flat["X_obs"]
        dt = flat["dt"]
        ids = flat["ids"]
        cog = flat["cog"]
        packed = flat["jax"]["packed"]
        loop_iter = state["loop_iter"]
        estep_args = (
            X_obs,
            dt,
            ids,
            cog,
            state["current_beta"],
            state["cluster_f"],
            state["current_scalar_K"],
            state["current_kappa"],
            state["current_s"],
            self.K,
            self.t_span,
            state["cluster_cog_a"],
            state["cluster_cog_b"],
            self.lambda_cog,
        )
        if self._use_jitter(loop_iter):
            state["assignments"], probs = update_assignments_jitter_jax(
                *estep_args,
                temperature=self.jitter_temperature,
                rng=self.rng,
                obs_weight=flat["obs_weight"],
                X_preds=globals_trajectories,
                packed=packed,
            )
            self.assignment_probabilities_ = probs
        elif self.estep_mode == "joint_grid":
            state["assignments"], beta_start = update_assignments_joint_grid_jax(
                *estep_args,
                self._estep_beta_grid(),
                beta_upper=flat["beta_upper"],
                obs_weight=flat["obs_weight"],
                X_preds=globals_trajectories,
                packed=packed,
            )
            self._apply_beta_start(state, beta_start)
        else:
            state["assignments"] = update_assignments_hard_jax(
                *estep_args,
                obs_weight=flat["obs_weight"],
                X_preds=globals_trajectories,
                packed=packed,
            )

        self._update_cog_regression(state, flat)

        if self.verbose >= 2:
            for z in range(self.n_subtypes):
                if not np.any(state["assignments"] == z):
                    print(f"Warning: Cluster {z} is empty at iteration {loop_iter}")

        state["cluster_f"] = fit_theta_cluster_all_jax(
            X_obs,
            dt,
            ids,
            self.K,
            self.t_span,
            cluster_f=state["cluster_f"],
            assignments=state["assignments"],
            s=state["current_s"],
            scalar_K=state["current_scalar_K"],
            lambda_f=self.lambda_f,
            beta_pred=state["current_beta"],
            kappa=state["current_kappa"],
            strict_tol=self.strict_tol,
            obs_weight=flat["obs_weight"],
            packed=packed,
        )

        # Solve once at the new cluster_f; the beta step reuses these trajectories.
        beta_trajectories = solve_trajectories_jax(
            state["cluster_f"],
            state["current_kappa"],
            state["current_scalar_K"],
            self.K,
            self.t_span,
        )

        state["current_beta"], lse = estimate_beta_jax(
            beta_all=state["current_beta"],
            X_obs=X_obs,
            dt=dt,
            ids=ids,
            cog=cog,
            t_span=self.t_span,
            trajectories=beta_trajectories,
            s=state["current_s"],
            assignments=state["assignments"],
            cog_a=state["cluster_cog_a"],
            cog_b=state["cluster_cog_b"],
            lambda_cog=self.lambda_cog,
            lambda_jsd=self.lambda_jsd,
            lambda_beta=self.lambda_beta,
            beta_mean=state["beta_mean"],
            beta_var=state["beta_var"],
            t_max=self.t_max,
            strict_tol=self.strict_tol,
            obs_weight=flat["obs_weight"],
            beta_upper=flat["beta_upper"],
            packed=packed,
        )

        return lse

    def _em_step_jax_intercept(self, state, flat):
        """
        One JAX EM iteration with a fitted intercept ``c`` (prediction ``c + s * x``).

        The globals step fits (s, kappa, scalar_K, c) on the unshifted data. All
        later steps of the iteration run unchanged on ``X - c``.
        """
        from . import jax_core as jc
        from .optimizer_jax import fit_theta_globals_intercept_jax

        raw_data, meta = flat["jax"]["packed_raw"]
        B = flat["n_biomarkers"]
        c_bounds = self._resolve_c_bounds(B)
        if self.c_mask is not None:
            fixed = ~np.asarray(self.c_mask, dtype=bool)
            c_bounds[fixed, 0] = c_bounds[fixed, 1] = state["current_c"][fixed]
        (
            state["current_s"],
            state["current_kappa"],
            state["current_scalar_K"],
            state["current_c"],
            globals_trajectories,
        ) = fit_theta_globals_intercept_jax(
            X_obs=flat["X_obs"],
            dt_obs=flat["dt"],
            ids=flat["ids"],
            K=self.K,
            t_span=self.t_span,
            cluster_f=state["cluster_f"],
            assignments=state["assignments"],
            beta_pred=state["current_beta"],
            s_guess=state["current_s"],
            scalar_K_guess=state["current_scalar_K"],
            kappa_guess=state["current_kappa"],
            c_guess=state["current_c"],
            c_bounds=c_bounds,
            c_center=self.c_center,
            lambda_c=self.lambda_c,
            lambda_scalar=self.lambda_scalar,
            lambda_kappa=self.lambda_kappa,
            scalar_K_center=self.scalar_K_center,
            fit_s=self.fit_s,
            strict_tol=self.strict_tol,
            obs_weight=flat["obs_weight"],
            s_mask=self.s_mask_,
            s_lower=flat["s_lower"],
            packed=(raw_data, meta),
        )
        c = state["current_c"]
        shifted_flat = dict(flat)
        shifted_flat["X_obs"] = flat["X_obs"] - c[None, :]
        shifted_flat["jax"] = dict(
            flat["jax"], packed=(jc._shift_data_jit(raw_data, jc.f64(c)), meta)
        )
        return self._em_step_jax_after_globals(state, shifted_flat, globals_trajectories)

    def _finalize_fit(self, state, flat, X):
        """Trim history arrays and copy fitted attributes onto self."""
        _h = state["loop_iter"] + 1

        self.theta_history = state["theta_hist"][:, :_h]
        self.beta_history = state["beta_hist"][:, :_h]
        self.kappa_history = state["kappa_hist"][:, :_h]
        self.lse_history = state["lse_hist"][:_h]
        self.lse_final = state["final_lse"]
        self.cog_regression_history = state["cog_reg_hist"][:, :, :_h]
        self.assignment_history = state["assign_hist"][:, :_h]

        self.iter_times = state["iter_time_hist"][1:_h]
        self.accepted_solver_stages = state["solver_hist"][1:_h]
        self.assign_changes = state["assign_change_hist"][1:_h]
        self.subtype_counts_history_ = state["subtype_counts_hist"][:, :_h]
        self.beta_at_upper_fraction_history_ = state["beta_at_upper_hist"][:_h]
        self.estep_beta_moves_history_ = state["estep_moves_hist"][:_h]

        self.cluster_f = state["cluster_f"]
        self.final_scalar_K = state["current_scalar_K"]
        self.final_s = state["current_s"]
        self.final_c = np.copy(state["current_c"])
        self.c_history_ = np.array(state["c_hist"])
        self.final_kappa = state["current_kappa"]
        self.final_assignments = state["assignments"]
        self.beta_upper_ = np.copy(flat["beta_upper"])
        self.subtype_mapping = None

        has_true = X is not None and len(X) > 0 and "subtype_true" in X[0]
        true_labels = (
            np.array([p.get("subtype_true", -1) for p in X])
            if has_true
            else np.full(flat["n_patients"], -1)
        )
        if np.all(true_labels >= 0):
            try:
                self.final_assignments_matched = match_labels_best_overlap(
                    state["assignments"], true_labels
                )
                self.label_mapping_applied = True
            except Exception as exc:
                if self.verbose >= 1:
                    print(f"Warning: could not match labels: {exc}")
                self.final_assignments_matched = state["assignments"].copy()
                self.label_mapping_applied = False
        else:
            self.final_assignments_matched = state["assignments"].copy()
            self.label_mapping_applied = False

        self.theta = np.concatenate(
            [np.ravel(state["cluster_f"][0]), state["current_s"], [state["current_scalar_K"]]]
        )
        self.cog_a = state["cluster_cog_a"][0]
        self.cog_b = state["cluster_cog_b"][0]
        self.cluster_cog_a = state["cluster_cog_a"]
        self.cluster_cog_b = state["cluster_cog_b"]
        self.final_f = np.ravel(state["cluster_f"][0]).copy()
        self.scalar_K_ = state["current_scalar_K"]

        n_biomarkers = flat["n_biomarkers"]
        self.X_pred = solve_system(
            np.zeros(n_biomarkers),
            self.final_f,
            self.K,
            self.t_span,
            state["current_scalar_K"],
            self.final_kappa,
            ode_method=self.ode_method,
        )

        self.n_obs_ = flat["n_obs"]
        self._var_per_biomarker_null = flat["var_per_biomarker_null"]
        self._n_obs_rows = flat["X_obs"].shape[0]
        self._sse_per_biomarker = compute_sse_per_biomarker(
            flat["X_obs"] - self.final_c[None, :],
            flat["dt"],
            flat["ids"],
            state["current_beta"],
            state["assignments"],
            state["cluster_f"],
            state["current_s"],
            state["current_scalar_K"],
            state["current_kappa"],
            self.K,
            self.t_span,
            ode_method=self.ode_method,
            obs_weight=flat["obs_weight"],
        )
        k = count_bic_params(
            self.final_s,
            self.final_kappa,
            self.cluster_f,
            self.n_subtypes,
            self.lambda_cog,
            self.cluster_cog_a,
            fit_s=self.fit_s,
            s_mask=getattr(self, "s_mask_", self.s_mask),
        )
        k += self._n_free_c(n_biomarkers)
        self.bic_n_params_ = k
        self.bic_ = compute_bic(
            self._sse_per_biomarker,
            self._var_per_biomarker_null,
            self.n_obs_,
            k,
        )

    def _n_free_c(self, n_biomarkers: int) -> int:
        """Number of free intercept entries counted in BIC."""
        if not self.fit_intercept:
            return 0
        bounds = self._resolve_c_bounds(n_biomarkers)
        free = bounds[:, 0] < bounds[:, 1]
        if self.c_mask is not None:
            free &= np.asarray(self.c_mask, dtype=bool)
        return int(free.sum())

    def _bic_n_params(self):
        """Backward-compatible BIC parameter count (same as ``bic_n_params_`` after ``fit``)."""
        if hasattr(self, "bic_n_params_"):
            return int(self.bic_n_params_)
        return count_bic_params(
            self.final_s,
            self.final_kappa,
            self.cluster_f,
            self.n_subtypes,
            self.lambda_cog,
            self.cluster_cog_a,
            fit_s=self.fit_s,
            s_mask=getattr(self, "s_mask_", self.s_mask),
        )

    @staticmethod
    def _reindex_ids(ids_global):
        """Remap patient IDs to contiguous 0..n_unique-1 for cluster θ fit."""
        unique = np.unique(ids_global)
        id_map = {orig: local for local, orig in enumerate(unique)}
        return np.array([id_map[i] for i in ids_global])

    @staticmethod
    def _checkpoint(state):
        """Snapshot mutable EM params (not histories)."""
        return dict(
            s=np.copy(state["current_s"]),
            c=np.copy(state["current_c"]),
            kappa=np.asarray(state["current_kappa"], dtype=float).copy(),
            scalar_K=float(state["current_scalar_K"]),
            cluster_f=[np.ravel(fc).copy() for fc in state["cluster_f"]],
            beta=np.copy(state["current_beta"]),
            assignments=np.copy(state["assignments"]),
            cog_a=[np.copy(a) for a in state["cluster_cog_a"]],
            cog_b=list(state["cluster_cog_b"]),
        )

    @staticmethod
    def _restore_checkpoint(state, ck):
        state["current_s"] = np.copy(ck["s"])
        state["current_c"] = np.copy(ck["c"])
        state["current_kappa"] = np.copy(ck["kappa"])
        state["current_scalar_K"] = ck["scalar_K"]
        state["cluster_f"] = [fc.copy() for fc in ck["cluster_f"]]
        state["current_beta"] = np.copy(ck["beta"])
        state["assignments"] = np.copy(ck["assignments"])
        state["cluster_cog_a"] = [np.copy(a) for a in ck["cog_a"]]
        state["cluster_cog_b"] = list(ck["cog_b"])

    def _register_checkpoint_signal_handler(self):
        """Save latest checkpoint on SIGTERM (e.g. PBS walltime warning)."""
        if not self._checkpoint_path:
            return

        def _on_term(signum, frame):
            del signum, frame
            snap = self._checkpoint_snapshot
            if snap is None:
                return
            st, fl, x_list = snap
            try:
                self._save_fit_checkpoint(st, fl, x_list, fit_complete=False)
            except Exception as exc:
                print(f"Checkpoint on SIGTERM failed: {exc}")

        self._checkpoint_term_handler = _on_term
        signal.signal(signal.SIGTERM, _on_term)

    def _unregister_checkpoint_signal_handler(self):
        if getattr(self, "_checkpoint_term_handler", None) is not None:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            self._checkpoint_term_handler = None

    def _save_fit_checkpoint(self, state, flat, X, *, fit_complete: bool):
        """Atomically overwrite ``checkpoint_path`` with warm-startable progress."""
        path = self._checkpoint_path
        if not path:
            return

        self._checkpoint_snapshot = (state, flat, X)
        _h = state["loop_iter"] + 1
        train_ids = np.array([p["id"] for p in X])

        payload = dict(
            fit_complete=fit_complete,
            loop_iter=int(state["loop_iter"]),
            best_lse=float(state["best_lse"]),
            final_lse=float(state["final_lse"]),
            n_subtypes=int(self.n_subtypes),
            train_ids=train_ids,
            train_assignments=np.copy(state["assignments"]),
            cluster_f=np.array(state["cluster_f"], dtype=float),
            cluster_cog_a=np.array(state["cluster_cog_a"], dtype=float),
            cluster_cog_b=np.array(state["cluster_cog_b"], dtype=float),
            final_s=np.copy(state["current_s"]),
            final_c=np.copy(state["current_c"]),
            final_kappa=np.copy(state["current_kappa"]),
            final_scalar_K=float(state["current_scalar_K"]),
            beta_history=np.copy(state["beta_hist"][:, :_h]),
            assignment_history=np.copy(state["assign_hist"][:, :_h]),
            lse_history=np.copy(state["lse_hist"][:_h]),
            kappa_history=np.copy(state["kappa_hist"][:, :_h]),
            cog_history=np.copy(state["cog_reg_hist"][:, :, :_h]),
            theta_history=np.copy(state["theta_hist"][:, :_h]),
            iter_times=np.copy(state["iter_time_hist"][1:_h]),
            accepted_solver_stages=np.array(state["solver_hist"][1:_h], dtype=object),
            assign_changes=np.copy(state["assign_change_hist"][1:_h]),
            theta_solver_stages=np.array(self.theta_solver_stages, dtype=object),
            max_iter=int(self.max_iter),
            beta_upper=np.copy(flat["beta_upper"]),
            subtype_counts_history=np.copy(state["subtype_counts_hist"][:, :_h]),
            beta_at_upper_fraction_history=np.copy(state["beta_at_upper_hist"][:_h]),
            estep_beta_moves_history=np.copy(state["estep_moves_hist"][:_h]),
        )

        # np.savez appends ".npz"; use a .tmp stem so the temp file is <path>.tmp.npz
        tmp_base = (path[:-4] + ".tmp") if path.endswith(".npz") else (path + ".tmp")
        tmp_npz = tmp_base + ".npz"
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        np.savez(tmp_base, **payload)
        os.replace(tmp_npz, path)
        if self.verbose >= 1:
            tag = "complete" if fit_complete else "progress"
            print(
                f"Checkpoint ({tag}): {path}  "
                f"(outer iter {state['loop_iter']}, LSE {state['final_lse']:.6g})"
            )

    def _record_history(self, state, lse, hist_idx):
        state["beta_hist"][:, hist_idx] = state["current_beta"]
        state["assign_hist"][:, hist_idx] = state["assignments"]
        state["theta_hist"][:, hist_idx] = np.concatenate(
            [
                np.ravel(state["cluster_f"][0]),
                state["current_s"],
                [state["current_scalar_K"]],
            ]
        )
        state["kappa_hist"][:, hist_idx] = state["current_kappa"]
        state["c_hist"].append(np.copy(state["current_c"]))
        state["lse_hist"][hist_idx] = lse
        state["subtype_counts_hist"][:, hist_idx] = np.bincount(
            state["assignments"], minlength=self.n_subtypes
        )
        state["beta_at_upper_hist"][hist_idx] = np.mean(
            state["current_beta"] >= state["beta_upper"] - _AT_UPPER_TOL
        )
        state["estep_moves_hist"][hist_idx] = state["estep_beta_moves"]
        for z in range(self.n_subtypes):
            state["cog_reg_hist"][z, :, hist_idx] = np.concatenate(
                [state["cluster_cog_a"][z], [state["cluster_cog_b"][z]]]
            )

    def _compute_val_score(self, X, beta, assignments=None):
        """Reconstruction LSE on X. Each patient uses their assigned subtype trajectory.

        ``assignments`` indexes ``cluster_f``. If omitted, the assignments stored by
        the most recent ``transform`` call are used.
        """
        if assignments is None:
            assignments = getattr(self, "transform_assignments", None)
        if assignments is None:
            raise ValueError(
                "assignments are required. Pass them explicitly, or call transform() first."
            )
        assignments = np.asarray(assignments, dtype=int)
        beta = np.asarray(beta, dtype=float)
        if len(assignments) != len(X) or len(beta) != len(X):
            raise ValueError(
                f"beta and assignments must have one entry per patient "
                f"(len(X)={len(X)}, len(beta)={len(beta)}, len(assignments)={len(assignments)})"
            )

        n_biomarkers = X[0]["X_obs"].shape[1]
        X_preds = [
            solve_system(
                np.zeros(n_biomarkers),
                np.ravel(self.cluster_f[z]),
                self.K,
                self.t_span,
                self.final_scalar_K,
                self.final_kappa,
                ode_method=self.ode_method,
            )
            for z in range(self.n_subtypes)
        ]
        lse = 0.0
        for i, p in enumerate(X):
            z = int(assignments[i])
            X_obs_i, obs_weight_i = split_observed(p["X_obs"])
            lse += reconstruction_sse(
                float(beta[i]),
                X_obs_i,
                np.asarray(p["dt"]),
                X_preds[z],
                self.t_span,
                self.final_s,
                obs_weight_i=obs_weight_i,
            )
        return lse
