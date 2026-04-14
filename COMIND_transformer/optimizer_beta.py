import numpy as np
from scipy.optimize import minimize

from .kernel_jsd_multi import KernelJSDMulti
from .utils import solve_system


def reconstruction_sse(
    beta_i: float,
    X_obs_i: np.ndarray,
    dt_i: np.ndarray,
    X_pred: np.ndarray,
    t_span: np.ndarray,
    s: np.ndarray,
) -> float:
    """Reconstruction-only SSE for one patient against one subtype trajectory."""
    t_pred_i = dt_i + beta_i
    X_interp_i = np.array(
        [np.interp(t_pred_i, t_span, s[b] * X_pred[b]) for b in range(X_pred.shape[0])]
    )
    residuals = X_obs_i.T - X_interp_i
    return float(np.sum(residuals ** 2))


def _vectorized_beta_loss_and_grad(
    beta_all: np.ndarray,
    X_obs: np.ndarray,
    dt: np.ndarray,
    ids: np.ndarray,
    cog: np.ndarray,
    t_span: np.ndarray,
    cluster_f: list,
    scalar_K: float,
    s: np.ndarray,
    assignments: np.ndarray,
    K: np.ndarray,
    cluster_cog_a: list,
    cluster_cog_b: list,
    lambda_cog: float,
    lambda_jsd: float,
    t_max: float,
    X_pred_by_cluster: dict,
    dxdt_by_cluster: dict,
    jsd_n_bins: int = None,
    jsd_bandwidth: float = None,
    jsd_value_range: tuple = None,
    lambda_beta: float = 0.0,
    beta_mean: float = None,
    beta_var: float = None,
) -> tuple:
    """Loss/gradient for all patient betas using precomputed subtype trajectories."""
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

            jsd_calc = KernelJSDMulti(
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


def estimate_beta(
    beta_all: np.ndarray,
    X_obs: np.ndarray,
    dt: np.ndarray,
    ids: np.ndarray,
    cog: np.ndarray,
    t_span: np.ndarray,
    cluster_f: list,
    scalar_K: float,
    s: np.ndarray,
    assignments: np.ndarray,
    K: np.ndarray,
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
    kappa: np.ndarray = None,
) -> tuple:
    """Optimize all patient betas jointly with precomputed subtype trajectories."""
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
        x_k = solve_system(np.zeros(n_biomarkers), f_cluster, K, t_span, scalar_K, kappa)
        X_pred_by_cluster[subtype] = x_k
        Kx_plus_f = K_eff @ x_k + f_cluster[:, None]
        dxdt_by_cluster[subtype] = (1.0 - x_k) * Kx_plus_f

    def loss_func(beta_vec):
        loss, _ = _vectorized_beta_loss_and_grad(
            beta_vec,
            X_obs,
            dt,
            ids,
            cog,
            t_span,
            cluster_f,
            scalar_K,
            s,
            assignments,
            K,
            cluster_cog_a,
            cluster_cog_b,
            lambda_cog,
            lambda_jsd,
            t_max,
            X_pred_by_cluster,
            dxdt_by_cluster,
            jsd_n_bins,
            jsd_bandwidth,
            jsd_value_range,
            lambda_beta,
            beta_mean,
            beta_var,
        )
        return loss

    def grad_func(beta_vec):
        _, grad = _vectorized_beta_loss_and_grad(
            beta_vec,
            X_obs,
            dt,
            ids,
            cog,
            t_span,
            cluster_f,
            scalar_K,
            s,
            assignments,
            K,
            cluster_cog_a,
            cluster_cog_b,
            lambda_cog,
            lambda_jsd,
            t_max,
            X_pred_by_cluster,
            dxdt_by_cluster,
            jsd_n_bins,
            jsd_bandwidth,
            jsd_value_range,
            lambda_beta,
            beta_mean,
            beta_var,
        )
        return grad

    result = minimize(
        loss_func,
        x0=beta_all.copy(),
        method="L-BFGS-B",
        jac=grad_func,
        bounds=[(0, t_max)] * n_patients,
        options={"maxiter": 100},
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
        reconstruction_lse += np.sum(residuals ** 2)

    return optimized_beta, reconstruction_lse
