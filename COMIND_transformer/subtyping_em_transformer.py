"""
SubtypingEM: EM algorithm for COMIND disease progression modeling with subtype discovery.

sklearn-compatible (BaseEstimator, TransformerMixin).

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
import numpy as np
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin

from .assignments import (
    precompute_trajectories,
    sse_matrix,
    update_assignments_hard,
    update_assignments_jitter,
)
from .model_selection import (
    compute_sse_per_biomarker,
    count_bic_params,
    compute_bic,
)
from .optimizer_theta_globals import fit_theta_globals
from .optimizer_theta_cluster import fit_theta_cluster
from .optimizer_beta import estimate_beta, reconstruction_sse
from .optimizer_cognitive_regression import fit_linear_cog_regression_multi
from .utils import (
    solve_system,
    initialize_beta,
    ensure_2d_cog,
    match_labels_best_overlap,
)


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
        lambda_jsd: float = 0.0,
        lambda_beta: float = 0.0,
        lambda_kappa: float = 0.0,
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
    ):
        self.max_iter = max_iter
        self.t_max = t_max
        self.step = step
        self.K = K
        self.rng = np.random.default_rng(75) if rng is None else rng

        self.lambda_f = lambda_f
        self.lambda_cog = lambda_cog
        self.lambda_scalar = lambda_scalar
        self.lambda_jsd = lambda_jsd
        self.lambda_beta = lambda_beta
        self.lambda_kappa = lambda_kappa

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
        self.t_span = np.linspace(0.0, self.t_max, int(self.t_max / self.step))
        self.assignment_probabilities_ = None
        self._checkpoint_path = checkpoint_path
        self._checkpoint_X = X
        self._checkpoint_snapshot = None

        flat = self._prepare_data(X)
        state = self._initialize_state(flat, X)
        self._register_checkpoint_signal_handler()
        try:
            self._run_em_loop(state, flat, X)
            self._finalize_fit(state, flat, X)
            if self._checkpoint_path:
                self._save_fit_checkpoint(state, flat, X, fit_complete=True)
        finally:
            self._unregister_checkpoint_signal_handler()
        return self

    def transform(self, X: list, use_cognitive_prior: bool = True) -> np.ndarray:
        """
        Estimate beta (time-shift) and subtype assignment for each patient.

        Returns a structured array with dtype [('beta', 'f8'), ('subtype', 'i4')].
        """
        if not hasattr(self, "cluster_f") or not hasattr(self, "cluster_cog_a"):
            raise RuntimeError("fit() must be called before transform()")

        n_biomarkers = X[0]["X_obs"].shape[1]
        lam_cog = self.lambda_cog if use_cognitive_prior else 0.0
        results = np.zeros(len(X), dtype=[("beta", "f8"), ("subtype", "i4")])

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
        beta_grid = np.linspace(0.1 * self.t_max, 0.9 * self.t_max, self.n_beta_grid)

        for idx, p in enumerate(tqdm(X, desc="Estimating beta and subtype assignments")):
            X_obs_i = p["X_obs"]
            dt_i = p["dt"]
            cog_i = p["cog"]
            if cog_i.ndim == 1:
                cog_i = cog_i.reshape(-1, 1)

            sse_grid = np.zeros((self.n_beta_grid, self.n_subtypes))
            for z in range(self.n_subtypes):
                cog_pred_z = cog_i @ self.cluster_cog_a[z] + self.cluster_cog_b[z]
                for gi, beta_g in enumerate(beta_grid):
                    sse_grid[gi, z] = reconstruction_sse(
                        beta_g, X_obs_i, dt_i, X_preds[z], self.t_span, self.final_s
                    )
                    if lam_cog > 0:
                        sse_grid[gi, z] += lam_cog * np.sum(
                            (dt_i + beta_g - cog_pred_z) ** 2
                        )

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
            )
            results[idx]["beta"] = float(beta_vec[0])
            results[idx]["subtype"] = best_z

        self.beta_val = results["beta"]
        self.transform_assignments = results["subtype"]
        return results

    def score(self, X: list, y=None) -> float:
        """Negative reconstruction LSE on X (higher = better, for sklearn CV)."""
        return -self._compute_val_score(X, self.transform(X)["beta"])

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

        X_obs = np.vstack(X_obs_list)
        dt = np.concatenate(dt_list)
        ids = np.concatenate(ids_list)
        cog = np.vstack(cog_list)

        if not (len(dt) == len(ids) == X_obs.shape[0] == cog.shape[0]):
            raise ValueError(
                f"Stacked shapes disagree: X_obs={X_obs.shape}, dt={dt.shape}, "
                f"ids={ids.shape}, cog={cog.shape}"
            )
        if cog.ndim == 1:
            cog = np.atleast_2d(cog).T

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
            beta_mean=beta_mean,
            beta_var=beta_var,
            n_obs=X_obs.shape[0] * X_obs.shape[1],
            var_per_biomarker_null=np.maximum(np.var(X_obs, axis=0, ddof=1), 1e-12),
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

        scalar_K = (
            float(self.initial_scalar_K)
            if self.initial_scalar_K is not None
            else float(np.max(flat["X_obs"]))
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
        X_preds = precompute_trajectories(
            cluster_f, self.K, self.t_span, scalar_K, kappa,
            ode_method=self.ode_method,
        )
        sse_mat = sse_matrix(
            flat["X_obs"],
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

        return dict(
            current_beta=beta.copy(),
            current_s=s,
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
            best_lse=np.inf,
            loop_iter=0,
            final_lse=initial_lse,
            n_biomarkers=n_biomarkers,
        )

    def _run_em_loop(self, state, flat, X):
        """Outer EM loop with checkpointing and θ-solver escalation."""
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
        X_obs = flat["X_obs"]
        dt = flat["dt"]
        ids = flat["ids"]
        cog = flat["cog"]
        unique_ids = np.unique(ids)

        state["current_s"], state["current_kappa"], state["current_scalar_K"] = (
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
                assignments=state["assignments"],
                cluster_f=state["cluster_f"],
                strict_tol=self.strict_tol,
                ode_method=self.ode_method,
            )
        )

        loop_iter = state["loop_iter"]
        use_jitter = (
            self.assignments_jitter
            and self.jitter_iter > 0
            and (loop_iter % self.jitter_iter == 0)
            and loop_iter > 0
        )
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
            )
            self.assignment_probabilities_ = probs
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
            )

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
        )

        return lse

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

        self.cluster_f = state["cluster_f"]
        self.final_scalar_K = state["current_scalar_K"]
        self.final_s = state["current_s"]
        self.final_kappa = state["current_kappa"]
        self.final_assignments = state["assignments"]
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
            flat["X_obs"],
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
        )
        k = count_bic_params(
            self.final_s,
            self.final_kappa,
            self.cluster_f,
            self.n_subtypes,
            self.lambda_cog,
            self.cluster_cog_a,
        )
        self.bic_n_params_ = k
        self.bic_ = compute_bic(
            self._sse_per_biomarker,
            self._var_per_biomarker_null,
            self.n_obs_,
            k,
        )

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
        )

        tmp_path = path + ".tmp"
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        np.savez(tmp_path, **payload)
        os.replace(tmp_path, path)
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
        state["lse_hist"][hist_idx] = lse
        for z in range(self.n_subtypes):
            state["cog_reg_hist"][z, :, hist_idx] = np.concatenate(
                [state["cluster_cog_a"][z], [state["cluster_cog_b"][z]]]
            )

    def _compute_val_score(self, X, beta):
        """Reconstruction LSE on validation set X given beta estimates."""
        n_biomarkers = X[0]["X_obs"].shape[1]
        f = self.theta[:n_biomarkers]
        s = self.theta[n_biomarkers : 2 * n_biomarkers]
        scalar_K = self.theta[-1]
        X_pred = solve_system(
            np.zeros(n_biomarkers), f, self.K, self.t_span, scalar_K, self.final_kappa
        )
        lse = 0.0
        for i, p in enumerate(X):
            tp = beta[i] + p["dt"]
            X_interp = np.vstack(
                [
                    np.interp(tp, self.t_span, X_pred[b]) * s[b]
                    for b in range(n_biomarkers)
                ]
            ).T
            lse += np.sum((p["X_obs"] - X_interp) ** 2)
        return lse
