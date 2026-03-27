import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import linear_sum_assignment
import statsmodels.formula.api as smf
from typing import Sequence

import jax
import jax.numpy as jnp
from ormatex_py.ode_sys import OdeSys, CustomJacLinOp
from ormatex_py import integrate_wrapper

class MultivariateLogistic(OdeSys):
    # K_eff has scalar_K and kappa already baked in: K_eff = scalar_K * K + kappa * I
    K_eff: jnp.ndarray  # (n, n) effective connectivity matrix
    f: jnp.ndarray      # (n,)   forcing vector

    def __init__(self, K_eff, f, **kwargs):
        super().__init__()
        self.K_eff = jnp.array(K_eff)
        self.f     = jnp.array(f)

    @jax.jit
    def _frhs(self, t, x, **kwargs):
        # dx/dt = (I - diag(x)) @ (K_eff @ x + f)
        #
        # g = total driving force per region:
        #     network input (K_eff @ x) + external forcing (f)
        g = self.K_eff @ x + self.f

        # (1 - x) is the logistic saturation — element-wise scaling
        # when x_i -> 1 (fully damaged), that region stops accumulating
        return (1.0 - x) * g

    @jax.jit
    def _fjac(self, t, x, **kwargs):
        # Jacobian of _frhs with respect to x
        # J = (I - diag(x)) @ K_eff - diag(g)
        #
        # Off-diagonal entry J[i,j] = (1 - x_i) * K_eff[i,j]
        #   — how much region j's damage affects region i's rate of change
        #   — zero if i and j aren't connected in the brain network
        #
        # Diagonal entry J[i,i] = (1 - x_i) * K_eff[i,i] - g_i
        #   — the self-feedback term, always negative (stabilizing)
        n = len(x)
        g = self.K_eff @ x + self.f

        # build off-diagonal part: (1 - x_i) * K_eff[i,j] for all i,j
        J = (1.0 - x)[:, None] * self.K_eff

        # correct the diagonal: replace with -(K_eff @ x + f)_i
        J = J.at[jnp.arange(n), jnp.arange(n)].set(-g)

        # CustomJacLinOp wraps J so ORMATEX can use it in Krylov iterations
        # it needs t, x, and the RHS function alongside the matrix
        return CustomJacLinOp(t, x, self.frhs, J)


def solve_system(x0: np.ndarray, f: np.ndarray, K: np.ndarray, t_span: np.ndarray,
                 scalar_K: float = 1.0, kappa=None) -> np.ndarray:
    n = K.shape[0]

    # scalar kappa only — builds K_eff once before integration
    if kappa is None:
        kappa = 0.0
    K_eff = scalar_K * K + float(kappa) * np.eye(n)

    # convert t_span to ORMATEX format
    # t_span is always uniform from your linspace so dt is constant
    t0     = float(t_span[0])
    dt     = float(t_span[1] - t_span[0])  # = self.step from SubtypingEM
    nsteps = len(t_span) - 1               # = int(t_max / step) - 1

    # create the ODE system and integrate
    sys = MultivariateLogistic(K_eff, f)
    res = integrate_wrapper.integrate(
        sys,
        jnp.array(x0, dtype=jnp.float64),
        t0,
        dt,
        nsteps,
        'exprb2',
        max_krylov_dim=min(20, n),  # 20 is good for n=80, fine for n=3 too
        iom=2,                      # incomplete orthogonalization, faster Arnoldi
    )

    # res.y shape is (nsteps+1, n) — transpose to (n, M) to match original
    return np.array(res.y).T

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
