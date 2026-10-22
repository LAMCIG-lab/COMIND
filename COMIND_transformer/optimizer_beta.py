import numpy as np
from scipy.optimize import minimize

from .kernel_jsd_multi import KernelJSDMulti
from .utils import interp_uniform, solve_system


def reconstruction_sse(
    beta_i: float,
    X_obs_i: np.ndarray,
    dt_i: np.ndarray,
    X_pred: np.ndarray,
    t_span: np.ndarray,
    s: np.ndarray,
    obs_weight_i: np.ndarray = None,
) -> float:
    """Reconstruction-only SSE for one patient against one subtype trajectory.

    ``obs_weight_i`` is an optional (n_timepoints, n_biomarkers) array
    (1.0 = observed, 0.0 = missing). ``None`` treats every entry as observed.
    """
    t_pred_i = dt_i + beta_i
    X_interp_i = np.array(
        [np.interp(t_pred_i, t_span, s[b] * X_pred[b]) for b in range(X_pred.shape[0])]
    )
    residuals = X_obs_i.T - X_interp_i
    if obs_weight_i is not None:
        residuals = residuals * obs_weight_i.T
    return float(np.sum(residuals ** 2))


def _beta_bounds(n_patients: int, t_max: float, beta_upper: np.ndarray = None) -> list:
    """L-BFGS-B bounds ``[(0, upper_i)]``; ``beta_upper=None`` means ``t_max`` for all."""
    if beta_upper is None:
        return [(0, t_max)] * n_patients
    beta_upper = np.asarray(beta_upper, dtype=float).ravel()
    if beta_upper.shape != (n_patients,):
        raise ValueError(
            f"beta_upper must have shape ({n_patients},), got {beta_upper.shape}"
        )
    return [(0, float(u)) for u in beta_upper]


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
    obs_weight: np.ndarray = None,
    _row_index: tuple = None,
) -> tuple:
    """Loss/gradient for all patient betas using precomputed subtype trajectories.

    ``obs_weight`` is an optional (n_rows, n_biomarkers) array matching
    stacked ``X_obs`` (1.0 = observed, 0.0 = missing). ``None`` treats
    every entry as observed.

    Vectorized over rows: each subtype's rows are interpolated at once with
    ``utils.interp_uniform`` (bit-identical to ``np.interp``). Per-patient
    sums are taken over the same element order as the original per-patient
    loop, and the loss is accumulated sequentially in the loop's
    subtype-then-patient order, so loss and gradient are bit-identical to it.

    ``_row_index`` is an optional precomputed ``_patient_row_index(ids)``.
    """
    unique_ids, row_pidx, groups = (
        _row_index if _row_index is not None else _patient_row_index(ids)
    )
    n_patients = len(unique_ids)
    n_rows = X_obs.shape[0]
    s = np.asarray(s, dtype=float)

    t_pred = dt + beta_all[row_pidx]
    z_row = assignments[row_pidx]
    sq_rows = np.zeros((n_rows, X_obs.shape[1]))
    prod_rows = np.zeros((n_rows, X_obs.shape[1]))
    cog_pred = np.zeros(n_rows)

    for subtype in np.unique(assignments):
        rows = np.flatnonzero(z_row == subtype)
        if rows.size == 0:
            continue
        t_k = t_pred[rows]
        X_interp = interp_uniform(s[:, None] * X_pred_by_cluster[subtype], t_k, t_span).T
        dxdt_interp = interp_uniform(s[:, None] * dxdt_by_cluster[subtype], t_k, t_span).T
        residuals = X_obs[rows] - X_interp
        if obs_weight is not None:
            residuals = residuals * obs_weight[rows]
        sq_rows[rows] = residuals ** 2
        prod_rows[rows] = residuals * dxdt_interp
        cog_pred[rows] = cog[rows] @ cluster_cog_a[subtype] + cluster_cog_b[subtype]

    cog_res = t_pred - cog_pred
    terms = [
        _per_patient_sum(sq_rows, groups, n_patients),
        lambda_cog * _per_patient_sum(cog_res ** 2, groups, n_patients),
    ]
    gradient = -2.0 * _per_patient_sum(prod_rows, groups, n_patients)
    gradient += 2.0 * lambda_cog * _per_patient_sum(cog_res, groups, n_patients)

    if lambda_beta > 0 and beta_mean is not None and beta_var is not None:
        terms.append(lambda_beta * (beta_all - beta_mean) ** 2 / beta_var)
        gradient += 2.0 * lambda_beta * (beta_all - beta_mean) / beta_var

    order = np.argsort(assignments, kind="stable")
    total_loss = _sequential_sum(np.stack([t[order] for t in terms], axis=1).ravel())

    if lambda_jsd > 0:
        unique_subtypes = np.unique(assignments)
        if len(unique_subtypes) >= 2:
            subtype_betas_list = [beta_all[assignments == st] for st in unique_subtypes]
            if jsd_value_range is None:
                jsd_value_range = (0, t_max)

            jsd_calc = KernelJSDMulti(
                distributions_list=subtype_betas_list,
                value_range=jsd_value_range,
                n_bins=jsd_n_bins,
                bandwidth=jsd_bandwidth,
            )
            total_loss += lambda_jsd * jsd_calc.jsd()

            gradients_list = jsd_calc.jsd_derivatives()
            for subtype_idx, subtype_id in enumerate(unique_subtypes):
                gradient[assignments == subtype_id] += lambda_jsd * gradients_list[subtype_idx]

    return total_loss, gradient


def _patient_row_index(ids: np.ndarray) -> tuple:
    """
    Unique patient ids, each row's patient position, and rows grouped by visit count.

    Parameters
    ----------
    ids : np.ndarray
        Patient id per row, shape (n_rows,).

    Returns
    -------
    unique_ids : np.ndarray
        Sorted unique ids (``np.unique(ids)``).
    row_pidx : np.ndarray
        Position of each row's patient in ``unique_ids``, shape (n_rows,).
    groups : list of tuple
        One ``(patients, rows)`` pair per distinct visit count ``n``:
        ``patients`` has shape (G,) and ``rows`` has shape (G, n), holding each
        patient's rows in their original order (as ``ids == id`` selects them).

    Notes
    -----
    After ``SubtypingEM._prepare_data`` the ids are sorted and contiguous, so
    positions come from the row offsets of each block. Otherwise this falls
    back to ``np.unique(..., return_inverse=True)``.
    """
    ids = np.asarray(ids).ravel()
    if ids.size and np.all(ids[1:] >= ids[:-1]):
        starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
        counts = np.diff(np.r_[starts, ids.size])
        unique_ids, row_pidx = ids[starts], np.repeat(np.arange(starts.size), counts)
        sorted_rows = np.arange(ids.size)
    else:
        unique_ids, row_pidx = np.unique(ids, return_inverse=True)
        row_pidx = row_pidx.ravel()
        counts = np.bincount(row_pidx, minlength=len(unique_ids))
        starts = np.r_[0, np.cumsum(counts)[:-1]]
        sorted_rows = np.argsort(row_pidx, kind="stable")
    groups = []
    for n in np.unique(counts):
        patients = np.flatnonzero(counts == n)
        rows = sorted_rows[starts[patients][:, None] + np.arange(n)[None, :]]
        groups.append((patients, rows))
    return unique_ids, row_pidx, groups


def _per_patient_sum(values: np.ndarray, groups: list, n_patients: int) -> np.ndarray:
    """
    Per-patient ``np.sum`` with the element order of the original per-patient loop.

    Parameters
    ----------
    values : np.ndarray
        Per-row values, shape (n_rows,) or (n_rows, n_biomarkers).
    groups : list of tuple
        ``groups`` from ``_patient_row_index``.
    n_patients : int
        Number of patients.

    Returns
    -------
    np.ndarray
        Shape (n_patients,). For 2-D ``values`` each patient's block is summed
        biomarker-major, like ``np.sum`` of the loop's (n_biomarkers, n_visits)
        array, so the result is bit-identical to it.
    """
    out = np.zeros(n_patients)
    for patients, rows in groups:
        block = values[rows]
        if block.ndim == 3:
            block = block.transpose(0, 2, 1)
        out[patients] = np.ascontiguousarray(block).reshape(len(patients), -1).sum(axis=1)
    return out


def _sequential_sum(terms: np.ndarray) -> float:
    """Left-to-right ``0.0 + t0 + t1 + ...``, the order of a Python ``+=`` loop."""
    return float(np.add.accumulate(np.r_[0.0, terms])[-1])


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
    strict_tol: bool = False,
    ode_method: str = "LSODA",
    obs_weight: np.ndarray = None,
    beta_upper: np.ndarray = None,
    X_preds: list = None,
) -> tuple:
    """Optimize all patient betas jointly with precomputed subtype trajectories.

    ``obs_weight`` is an optional (n_rows, n_biomarkers) array matching
    stacked ``X_obs`` (1.0 = observed, 0.0 = missing). ``None`` treats
    every entry as observed.

    ``beta_upper`` is an optional (n_patients,) per-patient upper bound, so
    patient i is bounded to ``[0, beta_upper[i]]``. ``None`` means ``t_max``
    for everyone (the original bounds).

    ``X_preds`` is an optional list of per-subtype (n_biomarkers, len(t_span))
    trajectories used instead of solving the ODE here. It must match
    ``cluster_f``, ``scalar_K`` and ``kappa``; ``dxdt`` is derived from it.
    """
    row_index = _patient_row_index(ids)
    n_patients = len(row_index[0])
    n_biomarkers = X_obs.shape[1]
    assignments = np.asarray(assignments)

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
        return _vectorized_beta_loss_and_grad(
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
            obs_weight,
            _row_index=row_index,
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

    _, row_pidx, groups = row_index
    t_pred = dt + optimized_beta[row_pidx]
    z_row = assignments[row_pidx]
    s_arr = np.asarray(s, dtype=float)
    sq_rows = np.zeros((X_obs.shape[0], n_biomarkers))
    for subtype in np.unique(assignments):
        rows = np.flatnonzero(z_row == subtype)
        X_interp = interp_uniform(
            s_arr[:, None] * X_pred_by_cluster[subtype], t_pred[rows], t_span
        ).T
        residuals = X_obs[rows] - X_interp
        if obs_weight is not None:
            residuals = residuals * obs_weight[rows]
        sq_rows[rows] = residuals ** 2
    reconstruction_lse = _sequential_sum(_per_patient_sum(sq_rows, groups, n_patients))

    return optimized_beta, reconstruction_lse
