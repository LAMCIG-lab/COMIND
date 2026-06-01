# model_selection.py

import numpy as np

def count_bic_params(final_s, final_kappa, cluster_f, n_subtypes, lambda_cog, cluster_cog_a=None):
    """
    Count free model parameters for BIC.

    Parameter groups:
      - n_biomarkers entries of s
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

    Returns
    -------
    k : int
    """
    threshold    = 0.01
    n_biomarkers = final_s.shape[0]

    kappa_active = int(np.sum(np.abs(final_kappa) >= threshold))
    k = n_biomarkers + kappa_active + 1     # s  +  active kappa  +  scalar_K

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
    n_obs               : int  — total scalar observations (rows × biomarkers)
    k                   : int  — number of free parameters

    Returns
    -------
    bic : float
    """
    sigma2   = np.maximum(var_per_biomarker_null, 1e-12)
    sse_norm = np.sum(sse_per_biomarker / sigma2)
    return float(k * np.log(n_obs) + 2.0 * sse_norm)
