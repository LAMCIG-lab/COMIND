import numpy as np
import pandas as pd
from scipy.optimize import minimize

# TODO: type hints
# TODO: doc strings

def fit_linear_cog_regression_multi(cog: np.ndarray, dt: np.ndarray, 
                                    beta: np.ndarray, ids: np.ndarray) -> tuple[np.ndarray, float]:
    """Available-case OLS of disease time on clinical covariates.

    Rows where ``cog`` has a non-finite value are dropped before the
    regression. This is intentionally a lighter-weight treatment than
    the masked-likelihood used for biomarkers: the cognitive target is
    a single scalar per visit with no per-entry structure to mask
    within that scalar. ``lambda_cog`` is currently always 0, so this
    path is unused in production; the guard exists so a future
    reactivation with missing clinical covariates does not silently
    produce NaN coefficients.
    """
    #assert cog.shape[0] == dt.shape[0]
    unique_ids = np.unique(ids)
    id_to_index = {pid: i for i, pid in enumerate(unique_ids)}
    index_array = np.array([id_to_index[i] for i in ids])  # shape: (n_obs,)
    t_pred = dt + beta[index_array]
    
    if cog.shape[0] == 1:
        cog = cog.T

    cog = np.asarray(cog, dtype=float)
    if cog.ndim == 1:
        cog = cog.reshape(-1, 1)
    finite = np.isfinite(cog).all(axis=1)
    if not np.all(finite):
        cog = cog[finite]
        t_pred = t_pred[finite]
    if cog.shape[0] == 0:
        raise ValueError(
            "fit_linear_cog_regression_multi: no rows with finite cog values"
        )

    X = np.hstack([cog, np.ones((cog.shape[0], 1))])  # (n_obs, n_features + 1)
    #print(X.T.shape, t_ij.shape)
    
    XtX = X.T @ X
    XtY = X.T @ t_pred
    new_cog = np.linalg.pinv(XtX) @ XtY

    a = new_cog[:-1]
    b = new_cog[-1]

    return a, b
