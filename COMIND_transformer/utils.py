import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import linear_sum_assignment
import statsmodels.formula.api as smf
from typing import Sequence

SOLVE_SYSTEM_CALLS = 0


def solve_system(
    x0: np.ndarray,
    f: np.ndarray,
    K: np.ndarray,
    t_span: np.ndarray,
    scalar_K: float = 1.0,
    kappa: np.ndarray = None,
    ode_method: str = "LSODA",
) -> np.ndarray:
    """
    Solves the multivariate logistic ODE system given initial conditions and parameters.

    Parameters
    ----------
    x0 : np.ndarray
        Initial state vector (typically zeros).
    f : np.ndarray
        Forcing term vector.
    K : np.ndarray
        Connectivity matrix.
    t_span : np.ndarray
        Array of time points to solve over.
    scalar_K : float
        Scalar for the connectivity matrix.
    kappa : np.ndarray
        Diag of K, representing self-propagation
    Returns
    -------
    np.ndarray
        Simulated biomarker trajectories of shape (n_biomarkers, len(t_span)).
    """
    global SOLVE_SYSTEM_CALLS
    SOLVE_SYSTEM_CALLS += 1

    if kappa is None:
        kappa = np.zeros(K.shape[0])
    K_eff = scalar_K * K + np.diag(kappa)
    diag_K_eff = np.diag(K_eff)

    if ode_method == "BDF":
        K_sparsity = (np.abs(K_eff) > 0).astype(int)
        K_sparsity[np.arange(K.shape[0]), np.arange(K.shape[0])] = 1
        jac_sparsity_arg = K_sparsity
    else:
        jac_sparsity_arg = None

    def ode_system(t, x):
        # Logistic state must stay in [0, 1); clip to avoid blow-up in K_eff @ x.
        eps = 1e-9
        x = np.clip(x, 0.0, 1.0 - eps)
        dxdt = (1.0 - x) * (K_eff @ x + f)
        return dxdt

    def jacobian_ode(t, x):
        eps = 1e-9
        x = np.clip(x, 0.0, 1.0 - eps)
        g = (K_eff @ x) + f
        J = (1 - x)[:, None] * K_eff
        J[np.diag_indices_from(J)] = (1.0 - x) * diag_K_eff - g
        return J

    ivp_kwargs = dict(
        t_eval=t_span,
        method=ode_method,
        jac=jacobian_ode,
    )
    if jac_sparsity_arg is not None:
        ivp_kwargs["jac_sparsity"] = jac_sparsity_arg

    sol = solve_ivp(
        ode_system,
        [t_span[0], t_span[-1]],
        x0,
        **ivp_kwargs,
    )

    y = sol.y
    if y.shape[1] == len(t_span):
        return y

    # LSODA can stop early on stiff/unstable trajectories; interpolate onto t_span
    # so downstream np.interp callers always see len(t_span) points.
    if not sol.success:
        import warnings
        warnings.warn(
            f"solve_ivp ({ode_method}) did not reach all t_eval points "
            f"({y.shape[1]}/{len(t_span)}); interpolating onto t_span.",
            RuntimeWarning,
            stacklevel=2,
        )
    y_out = np.empty((y.shape[0], len(t_span)), dtype=float)
    for b in range(y.shape[0]):
        y_out[b] = np.interp(t_span, sol.t, y[b], left=y[b, 0], right=y[b, -1])
    return y_out


def split_observed(X_obs):
    """
    Split a (possibly NaN-containing) observation array into a finite
    placeholder array and an elementwise observation weight.

    Parameters
    ----------
    X_obs : array-like
        Observed biomarker values, shape (n_rows, n_biomarkers). Missing
        entries are indicated by NaN (or any non-finite value).

    Returns
    -------
    X_filled : ndarray
        Copy of ``X_obs`` with non-finite entries replaced by 0.0.
    obs_weight : ndarray
        Same shape as ``X_obs``, 1.0 where ``X_obs`` is finite, 0.0 otherwise.

    Implementation: multiply every residual by ``obs_weight`` before
    summing. Residuals MUST be computed on ``X_filled``, then weighted —
    never the reverse (0 * NaN is NaN in IEEE754). The placeholder 0.0
    never affects results because it is always paired with weight 0.0.

    a patient can have one biomarker missing at one visit while everything else
    at that visit is used normally. Assumes missingness is MAR/MCAR 
    """
    X_obs = np.asarray(X_obs, dtype=float)
    obs_weight = np.isfinite(X_obs).astype(float)
    X_filled = np.where(obs_weight > 0, X_obs, 0.0)
    return X_filled, obs_weight


def beta_upper_from_followup(dt_per_patient: Sequence, t_max: float) -> np.ndarray:
    """
    Per-patient β upper bound that keeps every visit inside ``[0, t_max]``.

    Parameters
    ----------
    dt_per_patient : sequence of array-like
        Time since first visit for each patient.
    t_max : float
        End of the model time grid.

    Returns
    -------
    np.ndarray
        ``cap_i = t_max - max(dt_i)``, shape (n_patients,).

    Raises
    ------
    ValueError
        If any ``cap_i <= 0``, i.e. ``t_max`` is shorter than that patient's
        follow-up. The message lists the offending patient indices.
    """
    max_dt = np.array(
        [float(np.max(np.asarray(d, dtype=float))) for d in dt_per_patient], dtype=float
    )
    cap = float(t_max) - max_dt
    bad = np.flatnonzero(cap <= 0)
    if bad.size:
        raise ValueError(
            f"cap_beta_by_followup: t_max={t_max} is shorter than the follow-up of "
            f"patient index(es) {bad.tolist()} (max dt = {max_dt[bad].tolist()}), "
            f"so their beta upper bound t_max - max(dt) is <= 0."
        )
    return cap


def uniform_grid_bracket(t: np.ndarray, t_span: np.ndarray) -> np.ndarray:
    """
    Interval index ``j`` with ``t_span[j] <= t < t_span[j + 1]``, as ``np.interp`` finds it.

    Parameters
    ----------
    t : np.ndarray
        Query times, shape (N,).
    t_span : np.ndarray
        Increasing grid, shape (T,), T >= 2.

    Returns
    -------
    np.ndarray
        Integer indices in ``[0, T - 2]``, shape (N,). Times outside the grid
        get the first or last interval; ``interp_uniform`` clamps them.

    Notes
    -----
    On a uniform grid (``np.linspace``) the index is ``floor(t / h)`` followed
    by a one-step correction against the actual ``t_span`` values, because
    ``t_span[j]`` equals ``j * h`` only up to rounding. Non-uniform grids fall
    back to ``np.searchsorted``.
    """
    t = np.asarray(t, dtype=float)
    T = t_span.shape[0]
    h = t_span[1] - t_span[0]
    diffs = np.diff(t_span)
    if not np.allclose(diffs, h, rtol=1e-9, atol=0.0):
        return np.clip(np.searchsorted(t_span, t, side="right") - 1, 0, T - 2)
    j = np.clip(np.floor((t - t_span[0]) / h), 0, T - 2).astype(np.intp)
    down = (t < t_span[j]) & (j > 0)
    j[down] -= 1
    up = (t >= t_span[j + 1]) & (j < T - 2)
    j[up] += 1
    return j


def interp_uniform(
    fp: np.ndarray,
    t: np.ndarray,
    t_span: np.ndarray,
    slopes: np.ndarray = None,
) -> np.ndarray:
    """
    ``np.interp(t, t_span, fp[b])`` for every row ``b`` of ``fp`` at once.

    Parameters
    ----------
    fp : np.ndarray
        Values on the grid, shape (B, T).
    t : np.ndarray
        Query times, shape (N,).
    t_span : np.ndarray
        Grid, shape (T,).
    slopes : np.ndarray, optional
        Precomputed ``np.diff(fp, axis=1) / np.diff(t_span)``, shape (B, T - 1).

    Returns
    -------
    np.ndarray
        Interpolated values, shape (B, N).

    Notes
    -----
    Uses ``np.interp``'s own formula ``slope_j * (t - t_span[j]) + fp[j]``,
    its exact-hit branch (``t == t_span[j]`` returns ``fp[j]``) and its end
    clamping, so results match ``np.interp`` to rounding (in practice
    bit-for-bit).
    """
    t = np.asarray(t, dtype=float)
    j = uniform_grid_bracket(t, t_span)
    if slopes is None:
        slopes = np.diff(fp, axis=1) / np.diff(t_span)
    x0 = t_span[j]
    out = slopes[:, j] * (t - x0) + fp[:, j]
    exact = t == x0
    if exact.any():
        out[:, exact] = fp[:, j[exact]]
    left = t < t_span[0]
    if left.any():
        out[:, left] = fp[:, :1]
    right = t >= t_span[-1]
    if right.any():
        out[:, right] = fp[:, -1:]
    return out


def make_beta_grid(
    t_max: float, n: int, lo_frac: float = 0.1, hi_frac: float = 0.9
) -> np.ndarray:
    """
    β grid shared by ``transform``, ``transform_soft`` and the joint-grid E-step.

    Parameters
    ----------
    t_max : float
        End of the model time grid.
    n : int
        Number of grid points.
    lo_frac, hi_frac : float
        Grid range as fractions of ``t_max``.

    Returns
    -------
    np.ndarray
        ``np.linspace(lo_frac * t_max, hi_frac * t_max, n)``.
    """
    return np.linspace(lo_frac * t_max, hi_frac * t_max, n)


def check_beta_grid_within_cap(beta_grid: np.ndarray, beta_upper: np.ndarray) -> None:
    """
    Raise if some patient's cap excludes every β grid point.

    Parameters
    ----------
    beta_grid : np.ndarray
        Candidate βs, shape (G,).
    beta_upper : np.ndarray
        Per-patient upper bound, shape (n_patients,).

    Raises
    ------
    ValueError
        Lists the patient indices whose ``beta_upper`` is below every grid
        point, i.e. ``t_max`` is too short for their follow-up.
    """
    bad = np.flatnonzero(np.asarray(beta_upper) < np.min(beta_grid))
    if bad.size:
        raise ValueError(
            f"cap_beta_by_followup: every beta grid point exceeds the cap "
            f"t_max - max(dt) of patient index(es) {bad.tolist()} "
            f"(smallest grid point {float(np.min(beta_grid)):.6g}); t_max is "
            f"too short for their follow-up."
        )


def initialize_beta(ids: np.ndarray, beta_range: tuple = (0, 12), rng: np.random.Generator = None) -> np.ndarray:
    """
    Uniformly randomly initialize beta values for each unique patient ID.

    Returns
    -------
    np.ndarray
        1D array of beta values indexed by patient.
    """
    if rng is None:
        rng = np.random.default_rng(75)
    patient_ids = np.unique(ids)
    initial_beta = rng.uniform(beta_range[0], beta_range[1], size=len(patient_ids))
    
    return initial_beta

def initialize_f_eigen(
    K: np.ndarray,
    n_subtypes: int = 1,
    n_eigs: int = 8,
    jitter_strength: float = 0.05,
    jitter: bool = True,
    sparsity_mask: bool = False,
    sparsity_top_frac: float = 0.33,
    sparsity_zero_frac: float = 0.5,
    rng=None,
) -> np.ndarray:
    """
    Initialize forcing term f for each subtype using top eigenvectors of K.

    Weighting: subtype j weights eigenvector j most (with 0.25 on adjacent).
    Order: weighted sum -> normalize -> jitter -> optional sparsity mask.

    Returns
    -------
    np.ndarray
        Shape (n_subtypes, n_biomarkers). Use initial_f[k] for subtype k.
    """
    if rng is None:
        rng = np.random.default_rng(75)
    w, V = np.linalg.eigh(K)
    order = np.argsort(np.abs(w))[::-1]
    V = V[:, order]
    n_biomarkers = V.shape[0]
    n_eigs = min(n_eigs, V.shape[1])

    f_list = []
    for j in range(n_subtypes):
        # Subtype j: weight eigenvector (j % n_eigs) most, adjacent less
        weights = np.zeros(n_eigs)
        idx_dom = j % n_eigs
        weights[idx_dom] = 1.0
        if idx_dom > 0:
            weights[idx_dom - 1] = 0.25
        if idx_dom < n_eigs - 1:
            weights[idx_dom + 1] = 0.25

        f_j = np.zeros(n_biomarkers)
        for i in range(n_eigs):
            f_j += weights[i] * np.abs(V[:, i])

        # Normalize
        nrm = np.linalg.norm(f_j)
        if nrm > 1e-12:
            f_j = f_j / nrm

        # Jitter (different per subtype via rng)
        if jitter:
            f_j = f_j + rng.normal(0, jitter_strength, size=f_j.shape)
            f_j = np.clip(f_j, 0.0, None)

        # Optional sparsity: preserve top 33%, zero out 50% of the rest
        if sparsity_mask:
            n_top = max(1, int(n_biomarkers * sparsity_top_frac))
            idx_sorted = np.argsort(f_j)[::-1]
            mask_keep = np.zeros(n_biomarkers, dtype=bool)
            mask_keep[idx_sorted[:n_top]] = True
            remaining = np.where(~mask_keep)[0]
            n_zero = int(len(remaining) * sparsity_zero_frac)
            zero_idx = rng.choice(remaining, size=min(n_zero, len(remaining)), replace=False)
            f_j[zero_idx] = 0.0

        f_list.append(f_j)
    return np.array(f_list)

def set_diagonal_K(K: np.ndarray, s: float = 1.0, k: float = 0.05): # TODO: ask BG if K should be zeroed beforehand?
    """
    Params:
    s: global strength of connections? scales everything
    k: 
    
    """
    K = np.array(K, dtype=float, copy=True)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"K needs to be a square matrix, not {K.shape}")
    
    n = K.shape[0]
    I = np.eye(n)
    
    K_diag = s * K + k * I
    return K_diag
    # then return K with a diag
    
def ensure_2d_cog(c, n_rows_expected: int) -> np.ndarray:
    c = np.asarray(c)
    # (n_obs,) -> (n_obs, 1)
    if c.ndim == 1:
        c = c.reshape(-1, 1)

    # (1, n_obs) -> (n_obs, 1)
    if c.ndim == 2 and c.shape[0] == 1 and c.shape[1] == n_rows_expected:
        c = c.T

    # Final check: first dim must match number of observations for this patient
    if c.ndim != 2 or c.shape[0] != n_rows_expected:
        raise ValueError(
            f"cog shape mismatch; expected first dim {n_rows_expected}, got {c.shape}"
        )
    return c


def get_subtype_mapping_from_f(
    fitted_f_list: Sequence[np.ndarray],
    true_f_list: Sequence[np.ndarray],
) -> np.ndarray:
    """
    Create a mapping from fitted subtype indices to true subtype indices based on f parameters.

    Supports unequal counts: multiple fitted subtypes can map to the same true subtype
    (when n_fitted > n_true), or some true subtypes have no fitted counterpart (when n_fitted < n_true).
    Uses Hungarian algorithm on (n_fitted x n_true) cost matrix; any extra fitted subtypes
    are assigned to their closest true subtype.

    Parameters
    ----------
    fitted_f_list : Sequence[np.ndarray]
        List of fitted f arrays, one per subtype. Each should be shape (n_biomarkers,).
    true_f_list : Sequence[np.ndarray]
        List of true f arrays, one per subtype. Each should be shape (n_biomarkers,).

    Returns
    -------
    np.ndarray
        Mapping array where mapping[fitted_subtype] = true_subtype (in 0..n_true-1).
        Shape: (n_fitted,). All values are valid indices into true_f_list.
    """
    fitted_f_list = [np.asarray(f) for f in fitted_f_list]
    true_f_list = [np.asarray(f) for f in true_f_list]

    n_fitted = len(fitted_f_list)
    n_true = len(true_f_list)
    if n_true == 0:
        return np.array([], dtype=int)

    # Cost matrix: (n_fitted, n_true) — cost[i, j] = distance(fitted_i, true_j)
    cost_matrix = np.zeros((n_fitted, n_true))
    for i in range(n_fitted):
        for j in range(n_true):
            cost_matrix[i, j] = np.linalg.norm(fitted_f_list[i] - true_f_list[j])

    # Optimal assignment: assigns min(n_fitted, n_true) pairs
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # mapping[fitted_subtype] = true_subtype; length n_fitted, values in 0..n_true-1
    mapping = np.zeros(n_fitted, dtype=int)
    for fitted_idx, true_idx in zip(row_ind, col_ind):
        mapping[fitted_idx] = true_idx
    # When n_fitted > n_true, some fitted indices are unassigned; assign to closest true
    assigned_rows = set(row_ind)
    for i in range(n_fitted):
        if i not in assigned_rows:
            mapping[i] = int(np.argmin(cost_matrix[i]))

    return mapping


def get_subtype_mapping(fitted_assignments: np.ndarray, true_assignments: np.ndarray) -> np.ndarray:
    """
    Create a mapping from fitted subtype indices to true subtype indices.
    
    Uses Hungarian algorithm to find the best match between fitted and true subtypes
    based on patient overlap. Returns an array where index = fitted subtype, 
    value = corresponding true subtype.
    
    Parameters
    ----------
    fitted_assignments : np.ndarray
        Subtype assignments from fitted model (shape: n_patients,)
    true_assignments : np.ndarray
        True subtype assignments (shape: n_patients,)
    
    Returns
    -------
    np.ndarray
        Mapping array where mapping[fitted_subtype] = true_subtype
        Shape: (n_subtypes,)
    """
    fitted_assignments = np.asarray(fitted_assignments)
    true_assignments = np.asarray(true_assignments)
    
    if len(fitted_assignments) != len(true_assignments):
        raise ValueError(
            f"fitted_assignments and true_assignments must have same length, "
            f"got {len(fitted_assignments)} and {len(true_assignments)}"
        )
    
    n_fitted = len(np.unique(fitted_assignments))
    n_true = len(np.unique(true_assignments))
    n_subtypes = max(n_fitted, n_true)
    
    # Build confusion matrix: cost[i, j] = number of patients with fitted=i and true=j
    # We want to maximize agreement, so we use negative counts as costs
    cost_matrix = np.zeros((n_subtypes, n_subtypes))
    for i in range(n_subtypes):
        for j in range(n_subtypes):
            mask = (fitted_assignments == i) & (true_assignments == j)
            cost_matrix[i, j] = -np.sum(mask)  # Negative because we want to maximize
    
    # Use Hungarian algorithm to find optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Create mapping: mapping[fitted_subtype] = true_subtype
    mapping = np.zeros(n_subtypes, dtype=int)
    for fitted_idx, true_idx in zip(row_ind, col_ind):
        mapping[fitted_idx] = true_idx
    
    return mapping


def match_labels_best_overlap(em_labels: np.ndarray, true_labels: np.ndarray) -> np.ndarray:
    """
    Match EM-assigned labels to true labels based on best overlap using Hungarian algorithm.
    
    This function finds the optimal permutation of EM labels that maximizes agreement
    with true labels. This is necessary because EM assigns arbitrary label numbers
    that don't necessarily correspond to true subtype indices.
    
    Parameters
    ----------
    em_labels : np.ndarray
        Labels assigned by EM algorithm (shape: n_patients,)
    true_labels : np.ndarray
        True subtype labels (shape: n_patients,)
    
    Returns
    -------
    np.ndarray
        Remapped EM labels that best match true labels (shape: n_patients,)
    """
    em_labels = np.asarray(em_labels)
    true_labels = np.asarray(true_labels)
    
    if len(em_labels) != len(true_labels):
        raise ValueError(f"em_labels and true_labels must have same length, got {len(em_labels)} and {len(true_labels)}")
    
    n_em_clusters = len(np.unique(em_labels))
    n_true_clusters = len(np.unique(true_labels))
    n_clusters = max(n_em_clusters, n_true_clusters)
    
    # Build confusion matrix: cost[i, j] = number of patients with em_label=i and true_label=j
    # We want to maximize agreement, so we use negative counts as costs
    cost_matrix = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(n_clusters):
            # Count how many patients have em_label=i and true_label=j
            mask = (em_labels == i) & (true_labels == j)
            cost_matrix[i, j] = -np.sum(mask)  # Negative because we want to maximize
    
    # Use Hungarian algorithm to find optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Create mapping from EM labels to true labels
    label_mapping = np.zeros(n_clusters, dtype=int)
    for em_idx, true_idx in zip(row_ind, col_ind):
        label_mapping[em_idx] = true_idx
    
    # Remap EM labels
    remapped_labels = label_mapping[em_labels]
    
    return remapped_labels


def print_parameter_comparison(
    fitted_f_list: Sequence[np.ndarray],
    fitted_scalar_K: float,
    fitted_s: np.ndarray,
    true_f_list: Sequence[np.ndarray],
    true_scalar_K_list: Sequence[float],
    true_s: np.ndarray,
    subtype_mapping: np.ndarray = None,
    n_subtypes: int = None,
):
    """
    Print a comparison of fitted vs true parameters (f, scalar_K, s).
    
    Parameters
    ----------
    fitted_f_list : Sequence[np.ndarray]
        List of fitted f arrays, one per subtype.
    fitted_scalar_K : float
        Fitted global scalar_K value (shared across all subtypes).
    fitted_s : np.ndarray
        Fitted global s array.
    true_f_list : Sequence[np.ndarray]
        List of true f arrays, one per subtype.
    true_scalar_K_list : Sequence[float]
        List of true scalar_K values, one per subtype (for comparison).
    true_s : np.ndarray
        True global s array.
    subtype_mapping : np.ndarray, optional
        Mapping array where mapping[fitted_subtype] = true_subtype.
        If None, assumes fitted subtype i corresponds to true subtype i.
    n_subtypes : int, optional
        Number of subtypes. If None, inferred from fitted_f_list.
    """
    n_fitted = len(fitted_f_list)
    n_true = len(true_f_list)
    if subtype_mapping is None:
        subtype_mapping = np.arange(min(n_fitted, n_true))

    # Compare f by fitted subtype (handles n_fitted != n_true)
    for fitted_subtype in range(n_fitted):
        if fitted_subtype >= len(subtype_mapping):
            break
        true_subtype = int(subtype_mapping[fitted_subtype])
        if true_subtype < 0 or true_subtype >= n_true:
            print(f"\nFitted Subtype {fitted_subtype} -> (no true match)")
            print(f"  f_fitted:      {np.asarray(fitted_f_list[fitted_subtype])}")
            continue
        f_fitted = np.asarray(fitted_f_list[fitted_subtype])
        f_true = np.asarray(true_f_list[true_subtype])
        f_error = np.mean(np.abs(f_fitted - f_true))
        print(f"\nFitted Subtype {fitted_subtype} -> True Subtype {true_subtype}:")
        print(f"  f_fitted:      {f_fitted}")
        print(f"  f_true:        {f_true}")
    
    # Compare global scalar_K (single value, compare with average of true values)
    true_scalar_K_mean = np.mean(true_scalar_K_list)
    scalar_K_error = np.abs(fitted_scalar_K - true_scalar_K_mean)
    print(f"\nGlobal scalar_K:")
    print(f"  scalar_K_fitted: {fitted_scalar_K:.6f}")
    print(f"  scalar_K_true (mean): {true_scalar_K_mean:.6f}")
    if len(true_scalar_K_list) > 1:
        print(f"  scalar_K_true (per subtype): {true_scalar_K_list}")
    
    # Compare global s
    print(f"\nGlobal s:")
    print(f"  s_fitted:      {fitted_s}")
    print(f"  s_true:        {true_s}")


def _z(x):
    """Z-score normalization helper function."""
    x = np.asarray(x, float)
    m = np.nanmean(x)
    s = np.nanstd(x)
    return (x - m) / (s if np.isfinite(s) and s > 0 else 1.0)


def build_severity_index(df: pd.DataFrame = None, cog: np.ndarray = None) -> np.ndarray:
    """
    Build a severity index from clinical scores.
    
    Works for both real data (with MCATOT, TD_score, PIGD_score) and synthetic data
    (with cognitive_score). For real data, higher severity = worse (lower MoCA, higher TD/PIGD).
    For synthetic data, uses cognitive_score directly.
    
    Parameters
    ----------
    df : pd.DataFrame, optional
        DataFrame with clinical scores. Must have either:
        - For real data: "MCATOT", "TD_score", "PIGD_score" columns
        - For synthetic data: "cognitive_score" column
    cog : np.ndarray, optional
        Alternative input: array of cognitive scores (for synthetic data).
        If provided, df is ignored.
    
    Returns
    -------
    np.ndarray
        Severity index (higher = worse/later disease stage)
    """
    if cog is not None:
        # Synthetic data: use cognitive_score directly
        # For synthetic data, higher cognitive_score typically means later disease
        # We negate it to make higher = worse (consistent with real data)
        cog_array = np.asarray(cog, float)
        return -_z(cog_array)
    
    if df is None:
        raise ValueError("Either df or cog must be provided")
    
    # Check if this is real data (has MCATOT, TD_score, PIGD_score)
    if all(col in df.columns for col in ["MCATOT", "TD_score", "PIGD_score"]):
        moca = _z(df["MCATOT"].to_numpy())        # higher is better
        td = _z(df["TD_score"].to_numpy())
        pigd = _z(df["PIGD_score"].to_numpy())
        S = (-moca + td + pigd) / 3.0             # higher = worse / later
        return S
    # Check if this is synthetic data (has cognitive_score)
    elif "cognitive_score" in df.columns:
        cog_array = df["cognitive_score"].to_numpy()
        return -_z(cog_array)  # Negate to make higher = worse
    else:
        raise ValueError(
            "df must contain either (MCATOT, TD_score, PIGD_score) for real data "
            "or 'cognitive_score' for synthetic data"
        )


def fit_mixedlm_beta_from_clinical(df: pd.DataFrame = None, ids: np.ndarray = None, 
                                   dt: np.ndarray = None, t_max: float = 30,
                                   cog: np.ndarray = None, verbose: bool = False,
                                   rng: np.random.Generator = None) -> tuple:
    """
    Initialize beta values from clinical scores using mixed-effects linear model.
    
    Works for both real data (with MCATOT, TD_score, PIGD_score) and synthetic data
    (with cognitive_score). Fits a mixed-effects model: severity ~ dt + (1|id)
    and uses random intercepts to estimate patient-specific beta values.
    
    Parameters
    ----------
    df : pd.DataFrame, optional
        DataFrame with clinical scores and time information.
        Must have columns: patient id, dt (or time), and clinical scores.
    ids : np.ndarray, optional
        Patient IDs (required if df not provided)
    dt : np.ndarray, optional
        Time since first visit (required if df not provided)
    t_max : float
        Maximum time value for clipping beta
    cog : np.ndarray, optional
        Cognitive scores array (for synthetic data, alternative to df)
    verbose : bool
        Whether to print summary statistics
    rng : np.random.Generator, optional
        Random number generator for jitter
    
    Returns
    -------
    tuple
        (initial_beta, pid_to_beta, result)
        initial_beta : np.ndarray of beta values indexed by unique patient IDs
        pid_to_beta : dict mapping patient ID to beta value
        result : fitted model result (or None if failed)
    """
    if rng is None:
        rng = np.random.default_rng(0)
    
    # Build severity index
    if df is not None:
        S = build_severity_index(df=df)
        if ids is None:
            # Try to infer from df
            if "patient_id" in df.columns:
                ids = df["patient_id"].to_numpy()
            elif "subj_id" in df.columns:
                ids = df["subj_id"].to_numpy()
            else:
                raise ValueError("Cannot infer ids from df. Please provide ids parameter.")
        if dt is None:
            if "dt" in df.columns:
                dt = df["dt"].to_numpy()
            elif "time" in df.columns:
                dt = df["time"].to_numpy()
            else:
                raise ValueError("Cannot infer dt from df. Please provide dt parameter.")
    elif cog is not None and ids is not None and dt is not None:
        S = build_severity_index(cog=cog)
    else:
        raise ValueError("Must provide either (df) or (ids, dt, cog)")
    
    # Create DataFrame for mixed model
    dfm = pd.DataFrame({"id": ids, "dt": dt.astype(float), "S": S})
    
    # Keep longitudinal subjects (at least 2 observations)
    good_ids = dfm.groupby("id").size().pipe(lambda s: s[s >= 2]).index
    dfm = dfm[dfm["id"].isin(good_ids)].copy()
    
    # Clean + tiny jitter on dt to avoid exact duplicate rows
    dfm = dfm.replace([np.inf, -np.inf], np.nan).dropna(subset=["id", "dt", "S"])
    dfm["dt"] = dfm["dt"] + 1e-6 * rng.standard_normal(len(dfm))
    
    # Fit: severity ~ dt + (1|id)
    model = smf.mixedlm("S ~ dt", data=dfm, groups=dfm["id"], re_formula="1")
    result = None
    for meth in ("bfgs", "nm"):
        try:
            result = model.fit(method=meth, reml=True, maxiter=500, disp=False)
            if result.converged:
                break
        except:
            continue
    
    # Get slope k (fixed effect of dt)
    if result is not None and result.converged:
        k = float(result.params.get("dt", np.nan))
    else:
        # Fallback OLS for slope if MixedLM fails
        ols = smf.ols("S ~ dt", data=dfm).fit()
        k = float(ols.params.get("dt", np.nan))
    
    if not np.isfinite(k) or abs(k) < 1e-8:
        # if slope is (near) zero, default to small positive slope to avoid blow-up
        if verbose:
            print("[MixedLM] slope near zero; using fallback k=1.0")
        k = 1.0
    
    # Random intercepts u_i
    re = {}
    if result is not None and hasattr(result, "random_effects"):
        re = {pid: float(eff.get("Group", 0.0)) for pid, eff in result.random_effects.items()}
    
    unique_ids = np.unique(ids)
    beta_raw = np.array([re.get(pid, 0.0) / k for pid in unique_ids], dtype=float)
    
    # shift & clip: make the smallest beta zero, cap at t_max
    beta_shift = beta_raw - np.nanmin(beta_raw)
    initial_beta = np.clip(beta_shift, 0.0, t_max)
    
    pid_to_beta = {pid: initial_beta[i] for i, pid in enumerate(unique_ids)}
    
    if verbose:
        print("beta_init summary:", pd.Series(initial_beta).describe())
    
    return initial_beta, pid_to_beta, result
