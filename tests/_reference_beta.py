"""
Loop-based reference implementations of the NumPy beta step, kept for equivalence tests.

These are verbatim copies of ``optimizer_beta._vectorized_beta_loss_and_grad``,
``optimizer_beta.estimate_beta`` and ``kernel_jsd_multi.KernelJSDMulti`` from
before the vectorization in Phase 4 (including the ``beta_upper`` and
``X_preds`` arguments added earlier), renamed with a ``_reference`` prefix.
Do not import them from the package.
"""

import numpy as np
from scipy.optimize import minimize

from COMIND_transformer.optimizer_beta import _beta_bounds
from COMIND_transformer.utils import solve_system


class _ReferenceKernelJSDMulti:
    """Original per-sample-loop ``KernelJSDMulti``."""

    def __init__(self, distributions_list, value_range=(0, 1), bandwidth=None, n_bins=None):
        self.distributions = [np.asarray(d) for d in distributions_list]
        self.n_distributions = len(self.distributions)
        self.range = value_range

        if n_bins is None:
            range_size = value_range[1] - value_range[0]
            self.n_bins = max(20, int(range_size * 2))
        else:
            self.n_bins = n_bins
        self.bins = np.linspace(value_range[0], value_range[1], self.n_bins)

        if bandwidth is None:
            combined = np.concatenate(self.distributions)
            std = np.std(combined)
            n = len(combined)
            self.h = 0.9 * std * n**(-1/5)
        else:
            self.h = bandwidth
        self.eps = 1e-10

    def _gaussian_kernel(self, x):
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

    def _gaussian_kernel_derivative(self, x):
        return -x * self._gaussian_kernel(x)

    def _estimate_densities(self):
        densities = []
        for dist in self.distributions:
            dist_scaled = (self.bins[:, None] - dist[None, :]) / self.h
            P = np.mean(self._gaussian_kernel(dist_scaled), axis=1) / self.h
            P = P / (P.sum() + self.eps)
            densities.append(P)
        return densities

    def jsd(self):
        densities = self._estimate_densities()
        densities = [P + self.eps for P in densities]
        M = np.mean(densities, axis=0)
        M = M + self.eps
        jsd = 0.0
        for P in densities:
            kl = np.sum(P * np.log(P / M))
            jsd += kl
        jsd /= self.n_distributions
        return jsd

    def jsd_derivatives(self):
        densities = self._estimate_densities()
        densities = [P + self.eps for P in densities]
        M = np.mean(densities, axis=0)
        M = M + self.eps
        coeffs = []
        for P in densities:
            coeff = (1.0 / self.n_distributions) * np.log(P / M)
            coeffs.append(coeff)
        gradients = []
        for dist_idx, dist in enumerate(self.distributions):
            d_dist = np.zeros(len(dist))
            coeff = coeffs[dist_idx]
            for i, val in enumerate(dist):
                dist_scaled = (self.bins - val) / self.h
                kernel_deriv = -self._gaussian_kernel_derivative(dist_scaled) / self.h
                d_dist[i] = np.sum(coeff * kernel_deriv) / (len(dist) * self.h)
            gradients.append(d_dist)
        return gradients


def _reference_vectorized_beta_loss_and_grad(
    beta_all, X_obs, dt, ids, cog, t_span, cluster_f, scalar_K, s, assignments, K,
    cluster_cog_a, cluster_cog_b, lambda_cog, lambda_jsd, t_max, X_pred_by_cluster,
    dxdt_by_cluster, jsd_n_bins=None, jsd_bandwidth=None, jsd_value_range=None,
    lambda_beta=0.0, beta_mean=None, beta_var=None, obs_weight=None,
):
    unique_ids = np.unique(ids)
    n_patients = len(unique_ids)
    n_biomarkers = X_obs.shape[1]
    total_loss = 0.0
    gradient = np.zeros(n_patients)

    subtype_to_patient_idxs = {
        subtype: np.where(assignments == subtype)[0]
        for subtype in np.unique(assignments)
    }

    for subtype, patient_idxs in subtype_to_patient_idxs.items():
        if patient_idxs.size == 0:
            continue

        X_pred_k = X_pred_by_cluster[subtype]
        dxdt_k = dxdt_by_cluster[subtype]
        cog_a_k = cluster_cog_a[subtype]
        cog_b_k = cluster_cog_b[subtype]

        for idx in patient_idxs:
            patient_id = unique_ids[idx]
            mask = ids == patient_id
            X_obs_i = X_obs[mask, :]
            dt_i = dt[mask]
            cog_i = cog[mask, :]
            beta_i = beta_all[idx]

            t_pred_i = dt_i + beta_i
            X_interp_i = np.array(
                [np.interp(t_pred_i, t_span, s[b] * X_pred_k[b]) for b in range(n_biomarkers)]
            )
            residuals = X_obs_i.T - X_interp_i
            w_i = obs_weight[mask, :] if obs_weight is not None else None
            if w_i is not None:
                residuals = residuals * w_i.T
            total_loss += np.sum(residuals ** 2)

            dxdt_interp_i = np.array(
                [np.interp(t_pred_i, t_span, s[b] * dxdt_k[b]) for b in range(n_biomarkers)]
            )
            gradient[idx] = -2.0 * np.sum(residuals * dxdt_interp_i)

            cog_pred = cog_i @ cog_a_k + cog_b_k
            total_loss += lambda_cog * np.sum((t_pred_i - cog_pred) ** 2)
            gradient[idx] += 2.0 * lambda_cog * np.sum(t_pred_i - cog_pred)

            if lambda_beta > 0 and beta_mean is not None and beta_var is not None:
                total_loss += lambda_beta * (beta_i - beta_mean) ** 2 / beta_var
                gradient[idx] += 2.0 * lambda_beta * (beta_i - beta_mean) / beta_var

    if lambda_jsd > 0:
        unique_subtypes = np.unique(assignments)
        if len(unique_subtypes) >= 2:
            subtype_betas_list = [beta_all[assignments == st] for st in unique_subtypes]
            if all(len(betas) > 0 for betas in subtype_betas_list):
                if jsd_value_range is None:
                    jsd_value_range = (0, t_max)

            jsd_calc = _ReferenceKernelJSDMulti(
                distributions_list=subtype_betas_list,
                value_range=jsd_value_range,
                n_bins=jsd_n_bins,
                bandwidth=jsd_bandwidth,
            )

            jsd_value = jsd_calc.jsd()
            jsd_loss = lambda_jsd * jsd_value

            gradients_list = jsd_calc.jsd_derivatives()
            for subtype_idx, subtype_id in enumerate(unique_subtypes):
                subtype_indices = np.where(assignments == subtype_id)[0]
                for local_idx, patient_idx in enumerate(subtype_indices):
                    gradient[patient_idx] += lambda_jsd * gradients_list[subtype_idx][local_idx]

            total_loss += jsd_loss

    return total_loss, gradient


def _reference_estimate_beta(
    beta_all, X_obs, dt, ids, cog, t_span, cluster_f, scalar_K, s, assignments, K, cog_a,
    cog_b, lambda_cog=0.0, lambda_jsd=0.0, lambda_beta=0.0, beta_mean=None, beta_var=None,
    t_max=12.0, jsd_n_bins=None, jsd_bandwidth=None, jsd_value_range=None, kappa=None,
    strict_tol=False, ode_method="LSODA", obs_weight=None, beta_upper=None, X_preds=None,
):
    unique_ids = np.unique(ids)
    n_patients = len(unique_ids)
    n_biomarkers = X_obs.shape[1]

    if isinstance(cog_a, list):
        cluster_cog_a = cog_a
        cluster_cog_b = cog_b if isinstance(cog_b, list) else [cog_b] * len(cluster_cog_a)
    else:
        cluster_cog_a = [cog_a] * len(cluster_f)
        cluster_cog_b = [cog_b] * len(cluster_f)

    X_pred_by_cluster = {}
    dxdt_by_cluster = {}
    if kappa is not None:
        K_eff = scalar_K * K + np.diag(kappa)
    else:
        K_eff = scalar_K * K

    for subtype in range(len(cluster_f)):
        f_cluster = np.ravel(cluster_f[subtype])
        if X_preds is not None:
            x_k = np.asarray(X_preds[subtype], dtype=float)
        else:
            x_k = solve_system(
                np.zeros(n_biomarkers), f_cluster, K, t_span, scalar_K, kappa,
                ode_method=ode_method,
            )
        X_pred_by_cluster[subtype] = x_k
        Kx_plus_f = K_eff @ x_k + f_cluster[:, None]
        dxdt_by_cluster[subtype] = (1.0 - x_k) * Kx_plus_f

    def loss_and_grad(beta_vec):
        return _reference_vectorized_beta_loss_and_grad(
            beta_vec, X_obs, dt, ids, cog, t_span, cluster_f, scalar_K, s, assignments, K,
            cluster_cog_a, cluster_cog_b, lambda_cog, lambda_jsd, t_max, X_pred_by_cluster,
            dxdt_by_cluster, jsd_n_bins, jsd_bandwidth, jsd_value_range, lambda_beta,
            beta_mean, beta_var, obs_weight,
        )

    lbfgs_options = {"maxiter": 100}
    if strict_tol:
        lbfgs_options.update({"ftol": 2.2e-10, "gtol": 1e-6})

    result = minimize(
        loss_and_grad,
        x0=beta_all.copy(),
        method="L-BFGS-B",
        jac=True,
        bounds=_beta_bounds(n_patients, t_max, beta_upper),
        options=lbfgs_options,
    )
    optimized_beta = result.x

    reconstruction_lse = 0.0
    for idx, patient_id in enumerate(unique_ids):
        mask = ids == patient_id
        X_obs_i = X_obs[mask, :]
        dt_i = dt[mask]
        beta_i = optimized_beta[idx]
        subtype_i = assignments[idx]
        X_pred_cluster_i = X_pred_by_cluster[subtype_i]
        t_pred_i = dt_i + beta_i
        X_interp_i = np.array(
            [np.interp(t_pred_i, t_span, s[b] * X_pred_cluster_i[b]) for b in range(n_biomarkers)]
        )
        residuals = X_obs_i.T - X_interp_i
        w_i = obs_weight[mask, :] if obs_weight is not None else None
        if w_i is not None:
            residuals = residuals * w_i.T
        reconstruction_lse += np.sum(residuals ** 2)

    return optimized_beta, reconstruction_lse
