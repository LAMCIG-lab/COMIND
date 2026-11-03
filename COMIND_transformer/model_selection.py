# model_selection.py

from typing import Tuple

import numpy as np

from .utils import solve_system

def count_bic_params(
    final_s,
    final_kappa,
    cluster_f,
    n_subtypes,
    lambda_cog,
    cluster_cog_a=None,
    fit_s=True,
    s_mask=None,
):
    """
    Count free model parameters for BIC.

    Parameter groups:
      - free entries of s (all n_biomarkers if fit_s and s_mask is None;
        only True entries of s_mask otherwise; none if fit_s is False)
      - active kappa entries (|kappa_b| >= 0.01)
      - 1 global scalar_K
      - per-subtype cognitive regression (cog_a + cog_b) if lambda_cog > 0
      - active forcing term entries per subtype (f_b >= 0.01)

    Parameters
    ----------
    final_s       : (n_biomarkers,)
    final_kappa   : (n_biomarkers,)
    cluster_f     : list[(n_biomarkers,)]
    n_subtypes    : int
    lambda_cog    : float
    cluster_cog_a : list[(n_cog,)] or None
    fit_s         : bool
    s_mask        : (n_biomarkers,) bool or None
        True = that s entry is a free parameter. Ignored when fit_s is False.

    Returns
    -------
    k : int
    """
    threshold    = 0.01
    n_biomarkers = final_s.shape[0]

    kappa_active = int(np.sum(np.abs(final_kappa) >= threshold))
    k = kappa_active + 1                    # active kappa + scalar_K
    if fit_s:
        if s_mask is None:
            k += n_biomarkers               # per-biomarker supremum s
        else:
            k += int(np.sum(np.asarray(s_mask, dtype=bool)))

    if lambda_cog > 0 and cluster_cog_a:
        n_cog = np.asarray(cluster_cog_a[0]).shape[0]
        k += n_cog * n_subtypes             # cog_a per subtype
        k += n_subtypes                     # cog_b per subtype (one scalar each)

    for st in range(n_subtypes):
        f_sub = np.ravel(cluster_f[st])
        k += int(np.sum(f_sub >= threshold))

    return k


def compute_bic(sse_per_biomarker, var_per_biomarker_null, n_obs, k):
    """
    BIC = k·ln(n) + 2·SSE_norm  (lower is better).

    SSE_norm = Σ_b ( SSE_b / σ²_b ) where σ²_b is the null variance for
    biomarker b (overall variance on training data), so the fit term varies
    across models instead of cancelling.

    Parameters
    ----------
    sse_per_biomarker    : (n_biomarkers,)
    var_per_biomarker_null : (n_biomarkers,)
    n_obs               : int  — count of OBSERVED scalar entries (not
                                 rows × biomarkers when data is missing).
                                 The caller is responsible for passing
                                 ``int(obs_weight.sum())``.
    k                   : int  — number of free parameters

    When data is missing, ``var_per_biomarker_null`` must be computed with
    a NaN-aware variance (e.g. ``np.nanvar``); that is also the caller's
    responsibility.

    Returns
    -------
    bic : float
    """
    sigma2   = np.maximum(var_per_biomarker_null, 1e-12)
    sse_norm = np.sum(sse_per_biomarker / sigma2)
    return float(k * np.log(n_obs) + 2.0 * sse_norm)


def compute_sse_per_biomarker(
    X_obs,
    dt,
    ids,
    beta,
    assignments,
    cluster_f,
    s,
    scalar_K,
    kappa,
    K,
    t_span,
    ode_method="LSODA",
    obs_weight=None,
):
    """
    Sum of squared errors per biomarker on training data.

    ``obs_weight`` is an optional (n_rows, n_biomarkers) array (1.0 =
    observed, 0.0 = missing). ``None`` treats every entry as observed.

    Returns
    -------
    sse_per_b : (n_biomarkers,)
    """
    n_biomarkers = X_obs.shape[1]
    n_subtypes = len(cluster_f)
    sse_per_b = np.zeros(n_biomarkers)
    x0 = np.zeros(n_biomarkers)
    X_pred_by_cluster = [
        solve_system(
            x0, np.ravel(cluster_f[subtype]), K, t_span, scalar_K, kappa,
            ode_method=ode_method,
        )
        for subtype in range(n_subtypes)
    ]
    for r in range(X_obs.shape[0]):
        patient_id = ids[r]
        subtype = assignments[patient_id]
        beta_r = beta[patient_id]
        t = beta_r + dt[r]
        X_pred_sub = X_pred_by_cluster[subtype]
        pred_r = np.array([
            np.interp(t, t_span, X_pred_sub[b] * s[b]) for b in range(n_biomarkers)
        ])
        resid_r = X_obs[r] - pred_r
        if obs_weight is not None:
            resid_r = resid_r * obs_weight[r]
        sse_per_b += resid_r ** 2
    return sse_per_b


LOG_2PI = float(np.log(2.0 * np.pi))


def gaussian_bic(
    sse_per_biomarker,
    n_obs_per_biomarker,
    k_model,
) -> Tuple[float, float]:
    """Independent Gaussians, MLE variance per biomarker.

    For biomarker b with n_b observed values and SSE_b,
    σ²_b = SSE_b / n_b and

        -2 log L_b = n_b * (log(2π) + log(σ²_b) + 1)

    BIC = (-2 log L) + k_model * log(N), N = Σ n_b.

    Returns
    -------
    bic_gauss, bic_gauss_neg2loglik
    """
    sse = np.asarray(sse_per_biomarker, dtype=float).ravel()
    n = np.asarray(n_obs_per_biomarker, dtype=float).ravel()
    if sse.shape != n.shape:
        raise ValueError(
            f"sse_per_biomarker shape {sse.shape} != n_obs_per_biomarker {n.shape}"
        )
    k = int(k_model)
    ok = np.isfinite(sse) & np.isfinite(n) & (n > 0)
    if not np.any(ok):
        return float("nan"), float("nan")
    sigma2 = np.maximum(sse[ok] / n[ok], 1e-12)
    neg2 = float(np.sum(n[ok] * (LOG_2PI + np.log(sigma2) + 1.0)))
    N = float(np.sum(n[ok]))
    bic = neg2 + k * np.log(max(N, 1.0))
    return float(bic), float(neg2)
