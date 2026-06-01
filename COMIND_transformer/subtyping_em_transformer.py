import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin
from .optimizer_theta_globals import fit_theta_globals
from .optimizer_theta_cluster import fit_theta_cluster
from .optimizer_beta import estimate_beta, reconstruction_sse
from .optimizer_cognitive_regression import fit_linear_cog_regression_multi
from .kernel_jsd_multi import KernelJSDMulti
from .model_selection import count_bic_params, compute_bic
from .utils import *

class SubtypingEM(BaseEstimator, TransformerMixin):
    """
    EM algorithm for recovering disease progression model parameters
    and patient-specific time shifts from cross-sectional observations.

    Global (s, kappa, scalar_K) and cluster (f) θ-steps use a staged optimizer when
    ``jac_toggle`` is True: ``lbfgs_approx`` (SciPy finite differences on loss), then
    ``lbfgs_exact`` (LSODA sensitivity equations), then ``nelder_mead``.
    When ``jac_toggle`` is False, only ``nelder_mead`` is used for θ.

    Each EM outer iteration starts from the cheapest stage. If the reconstruction LSE
    does not improve enough vs. ``best_lse`` (``epsilon`` or ``relative_tolerance``),
    the same iteration is retried from a checkpoint with the next stage; after an
    accepted update, the next iteration again starts from the cheapest stage.

    If, after exhausting those θ stages, the trial LSE is still not strictly below the
    best LSE so far, EM exits early and restores parameters to the start of that outer
    iteration (so ``final_*`` match the last improving state).

    Global kappa is initialized with ``Uniform(0, 1)`` per biomarker unless
    ``initial_kappa`` is set. ``kappa_history`` stores kappa after the initial setup
    (column 0) and after each accepted outer iteration, aligned with ``beta_history`` /
    ``lse_history`` indexing.

    Warm-start: pass ``initial_f`` (``n_subtypes`` × ``n_biomarkers``), ``initial_s``,
    ``initial_scalar_K``, ``initial_kappa``, ``initial_assignments``, per-subtype cognitive
    params, and ``initial_beta``, or use :func:`COMIND_transformer.warm_start.load_warm_start_from_npz`.
    """

    def __init__(self, 
                 # [model parameters]
                 max_iter: int = 50,
                 t_max: float = 30,
                 step: float = 0.01,
                 K: np.ndarray = None,
                 rng: np.random.Generator = None,
                 
                # [hyperparameters]
                lambda_f: float = 0.01,
                lambda_cog: float = 0,
                lambda_scalar: float = 0.0,
                lambda_jsd: float = 0.0,  # JSD regularization for beta separation
                lambda_beta: float = 0.0,  # L2 regularization on beta values
                lambda_kappa: float = 0.0,
                 
                 # [initial guesses]
                 initial_f: np.ndarray = None,
                 initial_s: np.ndarray = None,
                 initial_scalar_K: float = None,
                 initial_kappa: np.ndarray = None,
                 initial_assignments: np.ndarray = None,
                 initial_cluster_cog_a: np.ndarray = None,
                 initial_cluster_cog_b: np.ndarray = None,
                 initial_beta: np.ndarray = None,
                 
                 # [iterative fitting parameters]
                 jac_toggle: bool = False,
                 epsilon: float = 1e-2,
                 relative_tolerance: float = 1e-3,
                 
                 # [clustering parmaters]
                 n_subtypes = 2,
                 assignments_jitter: bool = False,  # sample assignment from p(SSE) instead of argmax
                 jitter_iter: int = 1,  # when to jitter: every jitter_iter iterations (loop_iter % jitter_iter == 0)
                 jitter_temperature: float = 1.0,  # softmax temperature: <1 sharper, >1 flatter
                  
                 # [misc options]
                 verbose = 1,
                 ):

        # [model settings]
        self.max_iter = max_iter
        self.t_max = t_max
        self.step = step
        self.K = K
        
        if rng is None:
            self.rng = np.random.default_rng(75)
        else:
            self.rng = rng

        # [hyperparameters]
        self.lambda_f = lambda_f
        self.lambda_cog = lambda_cog
        self.lambda_scalar = lambda_scalar
        self.lambda_jsd = lambda_jsd
        self.lambda_beta = lambda_beta
        self.lambda_kappa = lambda_kappa
        
        # [initial guesses]
        self.initial_f = initial_f
        self.initial_s = initial_s
        self.initial_scalar_K = initial_scalar_K
        self.initial_kappa = initial_kappa
        self.initial_assignments = initial_assignments
        self.initial_cluster_cog_a = initial_cluster_cog_a
        self.initial_cluster_cog_b = initial_cluster_cog_b
        self.initial_beta = initial_beta
        
        # [fitting params]
        self.jac_toggle = jac_toggle
        self.epsilon = epsilon
        self.relative_tolerance = relative_tolerance
        
        # [clustering params]
        self.n_subtypes = n_subtypes
        self.assignments_jitter = assignments_jitter
        self.jitter_iter = jitter_iter
        self.jitter_temperature = jitter_temperature

        # [misc options]
        self.verbose = verbose
        
    def fit(self, X: np.ndarray, y: np.ndarray = None):
        
        ## data handling
        patient_ids = [p["id"] for p in X]
        n_patients = len(patient_ids)
        n_biomarkers = X[0]["X_obs"].shape[1]
        self.t_span = np.linspace(0.0, self.t_max, int(self.t_max / self.step))

        X_obs_list = []
        dt_list = []
        ids_list = []
        cog_list = []
        initial_beta_list = []

        for i, patient in enumerate(X):
            n = len(patient["dt"])
            X_obs_list.append(patient["X_obs"])
            dt_list.append(patient["dt"])
            ids_list.append(np.full(n, i))
            cog_list.append(ensure_2d_cog(patient["cog"], n)) 
            if "initial_beta" in patient:
                initial_beta_list.append(patient["initial_beta"])

        X_obs = np.vstack(X_obs_list)
        dt    = np.concatenate(dt_list)
        ids   = np.concatenate(ids_list)
        cog   = np.vstack(cog_list)

        # assert
        if not (len(dt) == len(ids) == X_obs.shape[0] == cog.shape[0]):
            raise ValueError(
                f"Stacked shapes disagree: X_obs={X_obs.shape}, dt={dt.shape}, "
                f"ids={ids.shape}, cog={cog.shape}"
            )

        # Total number of scalar observations (for BIC)
        self.n_obs_ = X_obs.shape[0] * X_obs.shape[1]
        # Null/reference variance per biomarker for BIC (overall variance on training data; avoids SSE canceling)
        self._var_per_biomarker_null = np.var(X_obs, axis=0, ddof=1)
        self._var_per_biomarker_null = np.maximum(self._var_per_biomarker_null, 1e-12)

        # print(X_obs.shape, dt.shape, cog.shape, ids.shape)

        if self.initial_beta is not None:
            initial_beta = np.asarray(self.initial_beta, dtype=float).copy()
            if initial_beta.shape != (n_patients,):
                raise ValueError(
                    f"initial_beta must have shape ({n_patients},), got {initial_beta.shape}"
                )
        elif initial_beta_list:
            if len(initial_beta_list) != n_patients:
                raise ValueError(
                    f"initial_beta on patients: expected {n_patients} values, got {len(initial_beta_list)}"
                )
            initial_beta = np.array(initial_beta_list)
        else:
            initial_beta = initialize_beta(ids=np.arange(n_patients), beta_range=(0, self.t_max), rng=self.rng)

        max_val = np.max(initial_beta)
        mask = (initial_beta > max_val - 1) & (initial_beta < max_val)
        initial_beta[mask] -= 2
        
        # Compute beta statistics for L2 regularization
        beta_mean = np.mean(initial_beta)
        beta_var = np.var(initial_beta)
        beta_var = max(beta_var, 1e-8)  # Add small epsilon to avoid division by zero
        
        K = self.K
        rng = self.rng
        n_obs = X_obs.shape[0]
        
        if cog.ndim == 1: # cog.shape = (n_obs, n_cog_features)
            cog = np.atleast_2d(cog)
            cog = cog.T
    
        n_cog_features = cog.shape[1]
        
        ## staged θ optimizers (globals + cluster)
        best_lse = np.inf
        if self.jac_toggle:
            solver_phases = ["lbfgs_approx", "lbfgs_exact", "nelder_mead"]
        else:
            solver_phases = ["nelder_mead"]
        
        ## initialize guesses
        # theta
        initial_x0 = np.zeros(n_biomarkers)
        
        # forcing term, random initialization if None
        if self.initial_f is not None:
            init_f = np.asarray(self.initial_f)
            if init_f.ndim == 2 and init_f.shape[0] == self.n_subtypes:
                initial_f = np.ravel(init_f[0])  # use first for bootstrap
            else:
                initial_f = np.ravel(init_f)
        else:
            initial_f = rng.uniform(0, 0.1, size=n_biomarkers)

        if self.initial_s is not None:
            initial_s = np.asarray(self.initial_s, dtype=float).ravel().copy()
            if initial_s.shape != (n_biomarkers,):
                raise ValueError(f"initial_s must have shape ({n_biomarkers},), got {initial_s.shape}")
        else:
            initial_s = rng.uniform(0.1, 3, size=n_biomarkers)

        if self.initial_scalar_K is not None:
            initial_scalar_K = float(self.initial_scalar_K)
        else:
            initial_scalar_K = float(np.max(X_obs))

        if self.initial_kappa is not None:
            current_kappa = np.asarray(self.initial_kappa, dtype=float).ravel().copy()
            if current_kappa.shape != (n_biomarkers,):
                raise ValueError(
                    f"initial_kappa must have shape ({n_biomarkers},), got {current_kappa.shape}"
                )
        else:
            current_kappa = rng.uniform(0.0, 1.0, size=n_biomarkers)

        current_scalar_K = initial_scalar_K

        # Initialize cluster-level parameters (f and cognitive regression per subtype)
        cluster_f = []
        cluster_cog_a = []
        cluster_cog_b = []
        default_cog_a = np.ones(n_cog_features)
        default_cog_b = 0.0
        for subtype in range(self.n_subtypes):
            if self.initial_f is not None:
                init_f = np.asarray(self.initial_f)
                if init_f.ndim == 2 and init_f.shape[0] == self.n_subtypes:
                    cluster_f.append(np.ravel(init_f[subtype]).copy())
                else:
                    initial_f_flat = np.ravel(init_f)
                    cluster_f.append(initial_f_flat.copy() + rng.uniform(-0.01, 0.01, size=n_biomarkers))
            else:
                cluster_f.append(rng.uniform(0, 0.1, size=n_biomarkers))

            if self.initial_cluster_cog_a is not None:
                cog_a_src = np.asarray(self.initial_cluster_cog_a, dtype=float)
                if cog_a_src.ndim == 1:
                    cluster_cog_a.append(cog_a_src.copy())
                else:
                    cluster_cog_a.append(np.ravel(cog_a_src[subtype]).copy())
            else:
                cluster_cog_a.append(default_cog_a.copy())

            if self.initial_cluster_cog_b is not None:
                cog_b_src = np.asarray(self.initial_cluster_cog_b, dtype=float).ravel()
                if cog_b_src.size == 1:
                    cluster_cog_b.append(float(cog_b_src[0]))
                else:
                    cluster_cog_b.append(float(cog_b_src[subtype]))
            else:
                cluster_cog_b.append(default_cog_b)

        # Cluster assignments (warm-start, patient dict, or random)
        if self.initial_assignments is not None:
            if len(self.initial_assignments) != n_patients:
                raise ValueError(
                    f"initial_assignments length ({len(self.initial_assignments)}) "
                    f"must match number of patients ({n_patients})"
                )
            if np.any(self.initial_assignments < 0) or np.any(self.initial_assignments >= self.n_subtypes):
                raise ValueError(
                    f"initial_assignments must be in range [0, {self.n_subtypes})"
                )
            assignments = self.initial_assignments.copy()
        elif all("initial_subtype" in p for p in X):
            assignments = np.array([p["initial_subtype"] for p in X], dtype=int)
            if np.any(assignments < 0) or np.any(assignments >= self.n_subtypes):
                raise ValueError(
                    f"initial_subtype values must be in range [0, {self.n_subtypes})"
                )
        else:
            assignments = rng.integers(0, self.n_subtypes, size=n_patients)

        representative_theta = np.concatenate(
            [np.ravel(cluster_f[0]), initial_s, [initial_scalar_K]]
        )

        ## initialize histories
        theta_history = np.zeros((representative_theta.shape[0], self.max_iter + 1))
        beta_history = np.zeros((n_patients, self.max_iter + 1))
        kappa_history = np.zeros((n_biomarkers, self.max_iter + 1))
        lse_history = np.zeros(self.max_iter + 1)
        cog_regression_history = np.zeros((self.n_subtypes, n_cog_features + 1, self.max_iter + 1))

        theta_history[:, 0] = representative_theta
        beta_history[:, 0] = initial_beta
        kappa_history[:, 0] = current_kappa
        for subtype in range(self.n_subtypes):
            cog_regression_history[subtype, :, 0] = np.concatenate(
                [cluster_cog_a[subtype], [cluster_cog_b[subtype]]]
            )

        ## Initial LSE (subtype-specific trajectories when assignments differ)
        X_pred_by_cluster_init = []
        for subtype in range(self.n_subtypes):
            f_cluster = np.ravel(cluster_f[subtype])
            X_pred_by_cluster_init.append(
                solve_system(initial_x0, f_cluster, K, self.t_span, current_scalar_K, current_kappa)
            )
        initial_lse = 0.0
        for idx, pid in enumerate(np.unique(ids)):
            mask = (ids == pid)
            X_obs_i = X_obs[mask, :]
            dt_i = dt[mask]
            beta_i = initial_beta[idx]
            subtype = assignments[idx]
            X_pred = X_pred_by_cluster_init[subtype]
            t_pred_i = dt_i + beta_i
            X_interp_i = np.array([
                np.interp(t_pred_i, self.t_span, initial_s[b] * X_pred[b])
                for b in range(n_biomarkers)
            ])
            residuals = X_obs_i.T - X_interp_i
            initial_lse += np.sum(residuals ** 2)

        lse_history[0] = initial_lse

        current_beta = initial_beta
        current_s = initial_s

        assignment_history = np.zeros((n_patients, self.max_iter + 1), dtype=int)
        assignment_history[:, 0] = assignments
        
        ### MAIN LOOP ###
        loop_iter = 0
        
        if self.verbose >= 1:
            pbar = tqdm(total=self.max_iter)
        else:
            # Create a dummy progress bar that does nothing
            class DummyProgressBar:
                def update(self, n=1):
                    pass
            pbar = DummyProgressBar()

        self.assignment_probabilities_ = None

        # Patient IDs in same order as current_beta/assignments (index i <-> unique_ids[i])
        unique_ids = np.unique(ids)
            
        def _lse_improvement_acceptable(best, lse_val, eps, rel_tol):
            """True if this EM trial improves reconstruction LSE enough vs. best so far."""
            if not np.isfinite(lse_val):
                return False
            if not np.isfinite(best) or best == np.inf:
                return True
            delta = best - lse_val
            if delta >= eps:
                return True
            denom = max(abs(best), 1e-12)
            if delta / denom >= rel_tol:
                return True
            return False

        while loop_iter < self.max_iter:
            hist_idx = loop_iter + 1

            # Checkpoint: retry same outer iteration with a stronger θ-solver if LSE gain is too small
            ck_s = np.copy(current_s)
            ck_kappa = np.asarray(current_kappa, dtype=float).copy()
            ck_scalar_K = float(current_scalar_K)
            ck_cluster_f = [np.ravel(fc).copy() for fc in cluster_f]
            ck_beta = np.copy(current_beta)
            ck_assignments = np.copy(assignments)
            ck_cog_a = [np.copy(a) for a in cluster_cog_a]
            ck_cog_b = [float(b) for b in cluster_cog_b]

            solver_phase_idx = 0

            while True:
                current_s = np.copy(ck_s)
                current_kappa = np.copy(ck_kappa)
                current_scalar_K = ck_scalar_K
                cluster_f = [fc.copy() for fc in ck_cluster_f]
                current_beta = np.copy(ck_beta)
                assignments = np.copy(ck_assignments)
                cluster_cog_a = [np.copy(a) for a in ck_cog_a]
                cluster_cog_b = list(ck_cog_b)

                current_solver = solver_phases[solver_phase_idx]

                ## STEP 1: GLOBAL LEVEL --> update s, kappa, and scalar_K
                current_s, current_kappa, current_scalar_K = fit_theta_globals(
                    X_obs=X_obs, dt_obs=dt, ids=ids, K=K,
                    t_span=self.t_span,
                    solver_stage=current_solver,
                    beta_pred=current_beta,
                    s_guess=current_s,
                    kappa_guess=current_kappa,
                    scalar_K_guess=current_scalar_K,
                    lambda_s=0.0, lambda_scalar=self.lambda_scalar,
                    lambda_kappa=self.lambda_kappa,
                    assignments=assignments,
                    cluster_f=cluster_f,
                )

                ## STEP 2: RECOMPUTE CLUSTER ASSIGNMENTS (hard or jittered)
                use_jitter_this_iter = (
                    self.assignments_jitter
                    and self.jitter_iter > 0
                    and (loop_iter % self.jitter_iter == 0)
                    and loop_iter > 0
                )
                if use_jitter_this_iter:
                    assignments, probs = self._update_assignments_jitter(
                        X_obs, dt, ids, cog, current_beta,
                        cluster_f, current_scalar_K, current_kappa, current_s,
                        K, self.t_span, cluster_cog_a, cluster_cog_b,
                        self.lambda_cog
                    )
                    self.assignment_probabilities_ = probs
                else:
                    assignments = self._update_assignments(
                        X_obs, dt, ids, cog, current_beta,
                        cluster_f, current_scalar_K, current_kappa, current_s,
                        K, self.t_span, cluster_cog_a, cluster_cog_b,
                        self.lambda_cog
                    )

                ## STEP 2.5: UPDATE COGNITIVE REGRESSION PARAMS PER SUBTYPE
                for subtype in range(self.n_subtypes):
                    cluster_mask = (assignments == subtype)
                    if np.sum(cluster_mask) == 0:
                        continue
                    cluster_patient_indices = np.where(cluster_mask)[0]
                    cluster_patient_ids = unique_ids[cluster_patient_indices]
                    cluster_patient_mask = np.isin(ids, cluster_patient_ids)

                    cog_subtype = cog[cluster_patient_mask]
                    dt_subtype = dt[cluster_patient_mask]
                    ids_subtype = ids[cluster_patient_mask]
                    beta_subtype = current_beta[cluster_patient_indices]

                    cluster_cog_a[subtype], cluster_cog_b[subtype] = fit_linear_cog_regression_multi(
                        cog_subtype, dt_subtype, beta_subtype, ids_subtype
                    )

                ## STEP 3: CLUSTER LEVEL --> update f[subtype] for each cluster (scalar_K is now global)
                for subtype in range(self.n_subtypes):
                    cluster_mask = (assignments == subtype)
                    cluster_patient_indices = np.where(cluster_mask)[0]

                    if len(cluster_patient_indices) == 0:
                        if self.verbose >= 2:
                            print(f"Warning: Cluster {subtype} is empty at iteration {loop_iter}")
                        continue

                    cluster_patient_ids = unique_ids[cluster_patient_indices]
                    cluster_patient_mask = np.isin(ids, cluster_patient_ids)
                    X_obs_cluster = X_obs[cluster_patient_mask, :]
                    dt_cluster = dt[cluster_patient_mask]
                    ids_cluster = ids[cluster_patient_mask]

                    unique_cluster_ids = np.unique(ids_cluster)
                    cluster_id_to_local = {
                        orig_id: local_idx for local_idx, orig_id in enumerate(unique_cluster_ids)
                    }
                    ids_cluster_local = np.array([cluster_id_to_local[i] for i in ids_cluster])
                    beta_cluster = current_beta[cluster_patient_indices]

                    f_guess_flat = np.ravel(cluster_f[subtype])
                    f_cluster = fit_theta_cluster(
                        X_obs=X_obs_cluster, dt_obs=dt_cluster, ids=ids_cluster_local, K=K,
                        t_span=self.t_span,
                        s=current_s, scalar_K=current_scalar_K,
                        lambda_f=self.lambda_f,
                        solver_stage=current_solver,
                        beta_pred=beta_cluster,
                        f_guess=f_guess_flat,
                        rng=rng,
                        kappa=current_kappa,
                    )

                    cluster_f[subtype] = np.ravel(f_cluster)

                ## STEP 4: SUBJECT LEVEL BETA
                current_beta, lse = estimate_beta(
                    beta_all=current_beta,
                    X_obs=X_obs,
                    dt=dt,
                    ids=ids,
                    cog=cog,
                    t_span=self.t_span,
                    cluster_f=cluster_f,
                    scalar_K=current_scalar_K,
                    s=current_s,
                    assignments=assignments,
                    K=K,
                    cog_a=cluster_cog_a,
                    cog_b=cluster_cog_b,
                    lambda_cog=self.lambda_cog,
                    lambda_jsd=self.lambda_jsd,
                    lambda_beta=self.lambda_beta,
                    beta_mean=beta_mean,
                    beta_var=beta_var,
                    t_max=self.t_max,
                    kappa=current_kappa
                )

                acceptable = _lse_improvement_acceptable(
                    best_lse, lse, self.epsilon, self.relative_tolerance
                )
                last_stage = solver_phase_idx >= len(solver_phases) - 1

                if acceptable or last_stage:
                    if not acceptable and last_stage and self.verbose >= 2:
                        print(
                            f"EM iter {loop_iter}: LSE improvement still below threshold after "
                            f"{current_solver}; accepting anyway."
                        )
                    break

                solver_phase_idx += 1
                if self.verbose >= 2:
                    print(
                        f"EM iter {loop_iter}: LSE gain too small with {solver_phases[solver_phase_idx - 1]}; "
                        f"retrying with {solver_phases[solver_phase_idx]}"
                    )

            # After exhausting θ solvers (inner loop always ends on acceptable or last stage): if
            # reconstruction LSE did not strictly improve vs best so far, exit EM and keep the
            # pre-iteration state (avoids committing worse parameters and trailing useless iterations).
            if np.isfinite(best_lse) and lse >= best_lse:
                if self.verbose >= 1:
                    print(
                        f"EM: reconstruction LSE did not improve vs best ({best_lse:.6g}); "
                        f"trial LSE={lse:.6g}. Exiting early at outer iter {loop_iter}."
                    )
                current_s = np.copy(ck_s)
                current_kappa = np.copy(ck_kappa)
                current_scalar_K = ck_scalar_K
                cluster_f = [fc.copy() for fc in ck_cluster_f]
                current_beta = np.copy(ck_beta)
                assignments = np.copy(ck_assignments)
                cluster_cog_a = [np.copy(a) for a in ck_cog_a]
                cluster_cog_b = list(ck_cog_b)
                lse = float(best_lse)
                break

            # Accepted this outer iteration (possibly after escalating θ-solver)
            beta_history[:, hist_idx] = current_beta
            assignment_history[:, hist_idx] = assignments
            representative_theta = np.concatenate(
                [np.ravel(cluster_f[0]), current_s, [current_scalar_K]]
            )
            theta_history[:, hist_idx] = representative_theta
            kappa_history[:, hist_idx] = current_kappa

            best_lse = min(best_lse, lse) if np.isfinite(best_lse) else lse
            lse_history[hist_idx] = lse

            for subtype in range(self.n_subtypes):
                cog_regression_history[subtype, :, hist_idx] = np.concatenate(
                    [cluster_cog_a[subtype], [cluster_cog_b[subtype]]]
                )

            loop_iter += 1
            pbar.update(1)
            
            
        # Use loop_iter+1 so early exit (no write for hist_idx=loop_iter+1) does not include a blank column.
        _h = loop_iter + 1
        self.theta_history = theta_history[:, 0:_h]
        self.beta_history = beta_history[:, 0:_h]
        self.kappa_history = kappa_history[:, 0:_h]
        self.lse_history = lse_history[0:_h]
        self.lse_final = lse

        self.cog_regression_history = cog_regression_history[:, 0:_h]
        self.assignment_history = assignment_history[:, 0:_h]
        
        # Store final cluster parameters
        self.cluster_f = cluster_f
        self.final_scalar_K = current_scalar_K  # Global scalar_K
        self.final_s = current_s
        self.final_kappa = current_kappa
        self.final_assignments = assignments
        self.subtype_mapping = None  # Will be set if compute_subtype_mapping is called
        
        # Match EM labels to true labels if available
        # Check if X contains true subtype labels
        if X is not None and len(X) > 0 and "subtype_true" in X[0]:
            true_labels = np.array([p.get("subtype_true", -1) for p in X])
            if np.all(true_labels >= 0):  # Only match if all have valid true labels
                try:
                    self.final_assignments_matched = match_labels_best_overlap(
                        assignments, true_labels
                    )
                    self.label_mapping_applied = True
                except Exception as e:
                    if self.verbose >= 1:
                        print(f"Warning: Could not match labels: {e}")
                    self.final_assignments_matched = assignments.copy()
                    self.label_mapping_applied = False
            else:
                self.final_assignments_matched = assignments.copy()
                self.label_mapping_applied = False
        else:
            self.final_assignments_matched = assignments.copy()
            self.label_mapping_applied = False
        
        # Store representative theta (first cluster)
        self.theta = np.concatenate([np.ravel(cluster_f[0]), current_s, [current_scalar_K]])
        self.cog_a = cluster_cog_a[0]  # Representative (first subtype)
        self.cog_b = cluster_cog_b[0]
        self.cluster_cog_a = cluster_cog_a  # Per-subtype parameters
        self.cluster_cog_b = cluster_cog_b
        
        self.final_f = np.ravel(cluster_f[0]).copy()  # Representative f (ensure 1D)
        self.scalar_K_ = current_scalar_K  # Global scalar_K

        # For transform, use first cluster as default
        f = np.ravel(cluster_f[0])  # Ensure 1D
        scalar_K = current_scalar_K  # Global scalar_K
        self.X_pred = solve_system(np.zeros(n_biomarkers), f, self.K, self.t_span, scalar_K, self.final_kappa)

        # Per-biomarker SSE for BIC (training residuals)
        self._n_obs_rows = X_obs.shape[0]
        self._sse_per_biomarker = self._compute_sse_per_biomarker(
            X_obs, dt, ids, current_beta, assignments,
            cluster_f, current_s, current_scalar_K, current_kappa
        )
        # BIC on training data (lower is better)
        k = count_bic_params(
            self.final_s,
            self.final_kappa,
            self.cluster_f,
            self.n_subtypes,
            self.lambda_cog,
            cluster_cog_a=self.cluster_cog_a,
        )
        self.bic_ = compute_bic(
            self._sse_per_biomarker,
            self._var_per_biomarker_null,
            self.n_obs_,
            k,
        )

        return self
    
    def _update_assignments(self, X_obs, dt, ids, cog, beta, cluster_f, scalar_K, kappa, s,
                            K, t_span, cluster_cog_a, cluster_cog_b, lambda_cog):
        """
        Update cluster assignments using hard assignment based on reconstruction error.
        
        For each patient, compute reconstruction error with each cluster's parameters
        and assign to the cluster with lowest error.
        """
        unique_ids = np.unique(ids)
        n_patients = len(unique_ids)
        n_subtypes = len(cluster_f)
        n_biomarkers = X_obs.shape[1]
        assignments = np.zeros(n_patients, dtype=int)

        # Precompute one trajectory per subtype (reuse across patients)
        X_pred_by_cluster = []
        for subtype in range(n_subtypes):
            f_cluster = np.ravel(cluster_f[subtype])
            X_pred_by_cluster.append(
                solve_system(np.zeros(n_biomarkers), f_cluster, K, t_span, scalar_K, kappa)
            )

        for idx, patient_id in enumerate(unique_ids):
            mask = (ids == patient_id)
            X_obs_i = X_obs[mask, :]
            dt_i = dt[mask]
            cog_i = cog[mask, :]
            beta_i = beta[idx]

            best_error = np.inf
            best_subtype = 0

            for subtype in range(n_subtypes):
                X_pred_cluster = X_pred_by_cluster[subtype]
                error = reconstruction_sse(beta_i, X_obs_i, dt_i, X_pred_cluster, t_span, s)
                cog_pred = cog_i @ cluster_cog_a[subtype] + cluster_cog_b[subtype]
                error += lambda_cog * np.sum((dt_i + beta_i - cog_pred) ** 2)

                if error < best_error:
                    best_error = error
                    best_subtype = subtype

            assignments[idx] = best_subtype

        return assignments

    def _update_assignments_jitter(self, X_obs, dt, ids, cog, beta, cluster_f, scalar_K, kappa, s,
                                  K, t_span, cluster_cog_a, cluster_cog_b, lambda_cog):
        """
        Jitter assignments for each patient form p_k = exp(-SSE_k)/sum_k exp(-SSE_k)
        (reconstruction-only SSE per subtype), then sample one assignment from that
        categorical. Returns assignments and the probability matrix for entropy etc.
        """
        unique_ids = np.unique(ids)
        n_patients = len(unique_ids)
        n_subtypes = len(cluster_f)
        n_biomarkers = X_obs.shape[1]
        assignments = np.zeros(n_patients, dtype=int)
        probabilities = np.zeros((n_patients, n_subtypes))

        # Precompute one trajectory per subtype (reuse across patients)
        X_pred_by_cluster = []
        for subtype in range(n_subtypes):
            f_cluster = np.ravel(cluster_f[subtype])
            X_pred_by_cluster.append(
                solve_system(np.zeros(n_biomarkers), f_cluster, K, t_span, scalar_K, kappa)
            )

        for idx, patient_id in enumerate(unique_ids):
            mask = (ids == patient_id)
            X_obs_i = X_obs[mask, :]
            dt_i = dt[mask]
            beta_i = beta[idx]

            sse_vec = np.zeros(n_subtypes)
            for subtype in range(n_subtypes):
                sse_vec[subtype] = reconstruction_sse(
                    beta_i, X_obs_i, dt_i, X_pred_by_cluster[subtype], t_span, s
                )

            log_p = -sse_vec / self.jitter_temperature
            log_p -= np.max(log_p)
            p = np.exp(log_p)
            p /= p.sum()

            probabilities[idx, :] = p
            assignments[idx] = self.rng.choice(n_subtypes, p=p)

        return assignments, probabilities

    # def _update_assignments_likelihood(self, X_obs, dt, ids, cog, beta, cluster_f, 
    #                                scalar_K, s, K, t_span, cluster_cog_a, cluster_cog_b, 
    #                                lambda_cog, error_scale=1.0):
    #     """Probabilistic assignments assuming Gaussian error distribution."""
    #     n_patients = len(np.unique(ids))
    #     n_subtypes = len(cluster_f)
    #     probabilities = np.zeros((n_patients, n_subtypes))
        
    #     unique_ids = np.unique(ids)
        
    #     for idx, patient_id in enumerate(unique_ids):
    #         mask = (ids == patient_id)
    #         X_obs_i = X_obs[mask, :]
    #         dt_i = dt[mask]
    #         cog_i = cog[mask, :]
    #         beta_i = beta[idx]
            
    #         log_likelihoods = np.zeros(n_subtypes)
            
    #         for subtype in range(n_subtypes):
    #             f_cluster = np.ravel(cluster_f[subtype])
    #             theta_cluster = np.concatenate([f_cluster, s, [scalar_K]])
    #             X_pred_cluster = solve_system(np.zeros(X_obs_i.shape[1]), f_cluster, K, t_span, scalar_K)
                
    #             error = beta_loss(
    #                 beta_i, X_obs_i, dt_i, X_pred_cluster, t_span,
    #                 cog_i, cluster_cog_a[subtype], cluster_cog_b[subtype], theta_cluster, lambda_cog
    #             )
                
    #             # Gaussian log-likelihood: -0.5 * (error/scale)^2
    #             log_likelihoods[subtype] = -0.5 * (error / error_scale) ** 2
            
    #         # Convert to probabilities (with numerical stability)
    #         log_likelihoods -= np.max(log_likelihoods)
    #         probabilities[idx] = np.exp(log_likelihoods)
    #         probabilities[idx] /= np.sum(probabilities[idx])
        
    #         assignments = np.argmax(probabilities, axis=1)
    #         return assignments, probabilities
    
    def _optimize_jsd_redistribution(self, beta, assignments, t_max, iteration=None):
        """
        Optimize JSD redistribution using gradient descent (multiple steps).
        This is called AFTER all betas are optimized to redistribute them based on JSD.
        
        Much faster than computing JSD thousands of times during individual beta optimization.
        Does actual optimization (multiple gradient steps) rather than just one step.
        
        Parameters
        ----------
        beta : np.ndarray
            Current beta values for all patients (already optimized for reconstruction)
        assignments : np.ndarray
            Subtype assignments for each patient
        t_max : float
            Maximum time value (for bounds)
        iteration : int, optional
            Current iteration number for verbose output
        
        Returns
        -------
        np.ndarray
            Redistributed beta values
        """
        # if self.n_subtypes != 2:
        #     return beta

        if self.n_subtypes < 2:
            return beta
        
        # Extract betas for each subtype
        # subtype_0_betas = beta[assignments == 0]
        # subtype_1_betas = beta[assignments == 1]
        
        # if len(subtype_0_betas) == 0 or len(subtype_1_betas) == 0:
        #     return beta
        
        # idx_subtype_0 = np.where(assignments == 0)[0]
        # idx_subtype_1 = np.where(assignments == 1)[0]

        unique_subtypes = np.unique(assignments)
        if len(unique_subtypes) < 2:
            return beta
        
        # Extract betas and indices for each subtype
        subtype_betas_list = []
        subtype_indices_list = []
        for st in unique_subtypes:
            subtype_betas = beta[assignments == st]
            if len(subtype_betas) == 0:
                return beta
            subtype_betas_list.append(subtype_betas)
            subtype_indices_list.append(np.where(assignments == st)[0])
        
        beta_optimized = beta.copy()
        jsd_before = None
        
        # Do multiple gradient descent steps to optimize JSD
        n_jsd_steps = max(1, int(self.lambda_jsd * 0.1)) 
        
        for step in range(n_jsd_steps):
            # Extract current betas for all subtypes
            current_subtype_betas = [beta_optimized[idx] for idx in subtype_indices_list]
            
            # Compute JSD and derivatives
            jsd_calc = KernelJSDMulti(
                distributions_list=current_subtype_betas,
                value_range=(0, t_max)
            )
            
            if step == 0:
                jsd_before = jsd_calc.jsd()
            
            gradients_list = jsd_calc.jsd_derivatives()  # Returns list of gradients
            
            # Step size: smaller for later steps (fine-tuning)
            step_size = self.lambda_jsd * 0.01 * (1.0 / (step + 1))
                        
            # Apply gradient step to MINIMIZE JSD (make distributions similar)
            for subtype_idx, subtype_indices in enumerate(subtype_indices_list):
                d_subtype = gradients_list[subtype_idx]
                beta_optimized[subtype_indices] -= step_size * d_subtype
            
            # Clip to valid range
            beta_optimized = np.clip(beta_optimized, 0, t_max)
        
        # Diagnostic output
        if self.verbose >= 2 and iteration is not None and iteration % 10 == 0:
            current_subtype_betas_after = [beta_optimized[idx] for idx in subtype_indices_list]
            jsd_after_calc = KernelJSDMulti(
                distributions_list=current_subtype_betas_after,
                value_range=(0, t_max)
            )
            jsd_after = jsd_after_calc.jsd()
            beta_change = np.mean(np.abs(beta_optimized - beta))
            print(f"  Iter {iteration}: JSD opt steps={n_jsd_steps}, "
                  f"JSD {jsd_before:.6f} -> {jsd_after:.6f}, "
                  f"mean beta change={beta_change:.6f}")
        
        return beta_optimized
    
    def compute_subtype_mapping(self, true_f_list, verbose=True):
        """
        Compute subtype mapping based on fitted f vs true f parameters.
        
        Parameters
        ----------
        true_f_list : Sequence[np.ndarray]
            List of true f arrays, one per subtype.
        verbose : bool
            Whether to print the mapping.
        """
        from .utils import get_subtype_mapping_from_f
        self.subtype_mapping = get_subtype_mapping_from_f(self.cluster_f, true_f_list)
        if verbose:
            print(f"\nSubtype mapping (fitted -> true): {self.subtype_mapping}")
            for fitted_subtype in range(self.n_subtypes):
                print(f"  Fitted subtype {fitted_subtype} -> True subtype {self.subtype_mapping[fitted_subtype]}")
        return self.subtype_mapping
    
    def transform(self, X: list[dict], use_cognitive_prior: bool = True) -> np.ndarray:
        """
        Estimate beta values (timeshift) and subtype assignments for a list of patient dicts.
        
        For each patient, this method:
        1. Determines the best subtype assignment based on reconstruction error
        2. Uses subtype-specific parameters to estimate timeshift (beta)
        3. Returns both beta and subtype assignment in a structured array
        
        Parameters
        ----------
        X : list[dict]
            List of patient dictionaries, each containing:
            - 'X_obs': (n_visits, n_biomarkers) biomarker observations
            - 'dt': (n_visits,) time deltas
            - 'cog': (n_visits, n_cog_features) cognitive features
        
        Parameters
        ----------
        X : list[dict]
            Patient dictionaries.
        use_cognitive_prior : bool, default=True
            If False, ignore cognitive priors during transform (lambda_cog=0).

        Returns
        -------
        np.ndarray
            Structured array with dtype [('beta', 'f8'), ('subtype', 'i4')]
            containing timeshift (beta) and subtype assignment for each patient.
        """
        if not hasattr(self, 'cluster_f') or not hasattr(self, 'cluster_cog_a'):
            raise RuntimeError("fit() must be called before transform()")
        
        n_patients = len(X)
        n_biomarkers = X[0]["X_obs"].shape[1]
        effective_lambda_cog = self.lambda_cog if use_cognitive_prior else 0.0
        
        # Create structured array for results
        dtype = [('beta', 'f8'), ('subtype', 'i4')]
        results = np.zeros(n_patients, dtype=dtype)
        
        for idx, p in enumerate(tqdm(X, desc="Estimating beta and subtype assignments")):
            X_obs_i = p["X_obs"]
            dt_i = p["dt"]
            cog_i = p["cog"]
            if cog_i.ndim == 1:
                cog_i = cog_i.reshape(-1, 1)

            # Step 1: Determine best subtype assignment
            # Try each subtype and find the one with lowest reconstruction error
            best_error = np.inf
            best_subtype = 0
            best_beta_guess = 0.0
            
            for subtype in range(self.n_subtypes):
                f_cluster = np.ravel(self.cluster_f[subtype])
                X_pred_cluster = solve_system(
                    np.zeros(n_biomarkers), f_cluster, self.K, 
                    self.t_span, self.final_scalar_K, self.final_kappa
                )
                
                beta_guess = 10.0
                # Compute reconstruction error using subtype-specific cognitive params
                error = reconstruction_sse(
                    beta_guess, X_obs_i, dt_i, X_pred_cluster, self.t_span, self.final_s
                )
                cog_pred = cog_i @ self.cluster_cog_a[subtype] + self.cluster_cog_b[subtype]
                error += effective_lambda_cog * np.sum((dt_i + beta_guess - cog_pred) ** 2)
                
                if error < best_error:
                    best_error = error
                    best_subtype = subtype
                    best_beta_guess = beta_guess
            
            # Step 2: Compute beta for this patient with fixed assigned subtype
            patient_ids_local = np.zeros(len(dt_i), dtype=int)
            beta_i_vec, _ = estimate_beta(
                beta_all=np.array([best_beta_guess], dtype=float),
                X_obs=X_obs_i,
                dt=dt_i,
                ids=patient_ids_local,
                cog=cog_i,
                t_span=self.t_span,
                cluster_f=self.cluster_f,
                scalar_K=self.final_scalar_K,
                s=self.final_s,
                assignments=np.array([best_subtype], dtype=int),
                K=self.K,
                cog_a=self.cluster_cog_a,
                cog_b=self.cluster_cog_b,
                lambda_cog=effective_lambda_cog,
                lambda_jsd=0.0,
                lambda_beta=0.0,
                beta_mean=None,
                beta_var=None,
                t_max=self.t_max,
                kappa=self.final_kappa,
            )
            beta_i = float(beta_i_vec[0])
            
            # Store results
            results[idx]['beta'] = beta_i
            results[idx]['subtype'] = best_subtype
        
        # Store for backward compatibility
        self.beta_val = results['beta']
        self.transform_assignments = results['subtype']
        
        return results


    def score(self, X: dict, y=None) -> float:
        """
        Computes timeshifts of validation set using transform,
        evaluates difference between predicted model and obs
        """
        transform_results = self.transform(X)
        # Extract 'beta' field from structured array returned by transform()
        beta_val = transform_results['beta']
        lse = self._compute_val_score(X, beta_val)
        return -lse
    
    def _compute_val_score(self, X: list[dict], beta: np.ndarray) -> float:
        n_biomarkers = X[0]["X_obs"].shape[1]
        f = self.theta[:n_biomarkers]
        s = self.theta[n_biomarkers:2 * n_biomarkers]
        scalar_K = self.theta[-1]

        X_pred = solve_system(np.zeros(n_biomarkers), f, self.K, self.t_span, scalar_K, self.final_kappa)

        lse = 0.0
        for i, p in enumerate(X):
            dt_i = p["dt"]
            X_obs_i = p["X_obs"]
            beta_i = beta[i]
            time_points = beta_i + dt_i

            X_interp = np.vstack([
                np.interp(time_points, self.t_span, X_pred[b]) * s[b]
                for b in range(n_biomarkers)
            ]).T

            lse += np.sum((X_obs_i - X_interp) ** 2)
        return lse

    def _compute_sse_per_biomarker(
        self,
        X_obs: np.ndarray,
        dt: np.ndarray,
        ids: np.ndarray,
        beta: np.ndarray,
        assignments: np.ndarray,
        cluster_f: np.ndarray,
        s: np.ndarray,
        scalar_K: float,
        kappa: np.ndarray,
    ) -> np.ndarray:
        """
        Sum of squared errors per biomarker on training data (same indexing as fit).
        Returns array of shape (n_biomarkers,).
        """
        n_biomarkers = X_obs.shape[1]
        n_subtypes = len(cluster_f)
        sse_per_b = np.zeros(n_biomarkers)
        X_pred_by_cluster = []
        for subtype in range(n_subtypes):
            f_cluster = np.ravel(cluster_f[subtype])
            X_pred_by_cluster.append(
                solve_system(np.zeros(n_biomarkers), f_cluster, self.K, self.t_span, scalar_K, kappa)
            )
        for r in range(X_obs.shape[0]):
            patient_id = ids[r]
            subtype = assignments[patient_id]
            beta_r = beta[patient_id]
            t = beta_r + dt[r]
            X_pred_sub = X_pred_by_cluster[subtype]  # (n_biomarkers, len(t_span))
            pred_r = np.array([
                np.interp(t, self.t_span, X_pred_sub[b] * s[b]) for b in range(n_biomarkers)
            ])
            sse_per_b += (X_obs[r] - pred_r) ** 2
        return sse_per_b