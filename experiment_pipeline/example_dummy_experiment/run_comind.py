"""
Generic COMIND grid-search experiment template.

Assumes the CSV is already cleaned and transformed (non-negative X_cols,
ready for the model). Cleaning/transformation is a separate, dataset-
specific step and stays out of this file entirely.

No CLI arguments. Candidate index comes from the PBS_ARRAYID environment
variable (set by the torque array job); defaults to 0 for local/interactive
runs. Edit the CONFIG block below to point at your data and customize the
grid. Everything else is generic plumbing.
"""
import os
from itertools import product

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from COMIND_transformer.model_selection import count_bic_params
from COMIND_transformer.preprocessing import build_connectome, parse_data
from COMIND_transformer.subtyping_em_transformer import SubtypingEM

########################################################
# CONFIG: edit this block per experiment
########################################################
script_dir = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(script_dir, "dummy_clean.csv")
CONNECTOME_PATH = os.path.join(script_dir, "dummy_K.csv")

ID_COL = "subj_id"
TIME_COL = "time"  # assumed already in years
X_COLS = ["biomarker_1", "biomarker_2", "biomarker_3", "biomarker_4", "biomarker_5"]
CLINICAL_COLS = []  # beta-init predictors, if any (leave [] if unused)
METADATA_COLS = [
    "meta_categorical_1",
    "meta_categorical_2",
    "meta_categorical_3",
]

T_MAX = 40.0
TEST_SIZE = 0.2
SPLIT_SEED = 75

EXPERIMENT_CONFIG = {
    "name": "example_dummy_experiment",
    "max_iter": 10,
    "theta_solver_stages": ("lbfgs_approx",), # choose solver type: "lbfgs_approx", "lbfgs_exact", "lbfgs_rk4"
    "n_subtypes_list": [1, 2, 3, 4],
    "param_grid_hyper": {
        "lambda_f": [10.0],
        "lambda_kappa": [5.0],
        "lambda_cog": [0.0],
        "lambda_scalar": [10.0],
        "scalar_K_center": [1.0],
        "lambda_jsd": [50.0],
        "lambda_beta": [0.0], 
    },
}
# =============================================================================

out_dir = os.path.join(script_dir, "results")
os.makedirs(out_dir, exist_ok=True)

current_candidate = int(os.environ.get("PBS_ARRAYID", 0))

print(f"=== Experiment: {EXPERIMENT_CONFIG['name']} ===")
print(f"candidate: {current_candidate}")

# --- 1. Load (already-cleaned) data ---
df = pd.read_csv(CSV_PATH)
print(f"Loaded {CSV_PATH}: {df.shape}")

x_mat = df[X_COLS].to_numpy(dtype=float)
if (x_mat < 0).any():
    raise ValueError(
        f"X_COLS contain negative values (min={x_mat.min():.4f}) -- "
        f"this script assumes the CSV is already transformed to be "
        f"non-negative. Clean/transform your data upstream first."
    )

# --- 2. Patient list ---
all_patients = parse_data(
    df,
    subj_id_col=ID_COL,
    time_col=TIME_COL,
    X_cols=X_COLS,
    clinical_cols=CLINICAL_COLS,
    metadata_cols=METADATA_COLS,
)
n_biomarkers = len(X_COLS)
print(f"Patients: {len(all_patients)}, biomarkers: {n_biomarkers}")

# 3. Connectome 
K, disconnected = build_connectome(CONNECTOME_PATH, X_COLS, pad_missing=True)
assert K.shape == (n_biomarkers, n_biomarkers)
print(f"K aligned: {K.shape}, {len(disconnected)} disconnected: {disconnected}")

f_init = np.random.RandomState(SPLIT_SEED).uniform(0.05, 0.15, size=n_biomarkers)

# 4. Train/val split    
X_train, X_val = train_test_split(
    all_patients, test_size=TEST_SIZE, random_state=SPLIT_SEED
)
print(f"Train: {len(X_train)} patients, Val: {len(X_val)} patients")

# 5. Build the grid
N_SUBTYPES_LIST = list(EXPERIMENT_CONFIG["n_subtypes_list"])
param_grid_hyper = dict(EXPERIMENT_CONFIG["param_grid_hyper"])
hyper_names = list(param_grid_hyper.keys())
hyper_values = list(param_grid_hyper.values())
hyper_combinations = list(product(*hyper_values))
n_hyper_per_K = len(hyper_combinations)
total_combinations = len(N_SUBTYPES_LIST) * n_hyper_per_K
print(f"Grid: {n_hyper_per_K} hyper combos x {len(N_SUBTYPES_LIST)} n_subtypes "
      f"= {total_combinations} total candidates")

if current_candidate >= total_combinations:
    print(f"candidate {current_candidate} >= {total_combinations}, nothing to do")
    raise SystemExit(0)

k_idx = current_candidate // n_hyper_per_K
sub_cand = current_candidate % n_hyper_per_K
n_subtypes = N_SUBTYPES_LIST[k_idx]
params = {name: hyper_combinations[sub_cand][i] for i, name in enumerate(hyper_names)}
params["n_subtypes"] = n_subtypes

print(f"\n=== Candidate {current_candidate} (n_subtypes={n_subtypes}) ===")
for name, value in params.items():
    print(f"  {name}: {value}")

# 6. Fit
subtyping_em = SubtypingEM(
    K=K,
    initial_f=f_init,
    initial_scalar_K=1.0,
    n_subtypes=n_subtypes,
    max_iter=EXPERIMENT_CONFIG["max_iter"],
    theta_solver_stages=EXPERIMENT_CONFIG["theta_solver_stages"],
    t_max=T_MAX,
    step=0.01,
    epsilon=1e-2,
    lambda_f=params["lambda_f"],
    lambda_cog=params["lambda_cog"],
    lambda_scalar=params["lambda_scalar"],
    scalar_K_center=params["scalar_K_center"],
    lambda_kappa=params["lambda_kappa"],
    lambda_jsd=params["lambda_jsd"],
    lambda_beta=params["lambda_beta"],
    verbose=1,
    rng=np.random.default_rng(SPLIT_SEED),
)

checkpoint_path = os.path.join(out_dir, f"checkpoint_{current_candidate}.npz")
subtyping_em.fit(X_train, checkpoint_path=checkpoint_path)

transform_results = subtyping_em.transform(X_val, use_cognitive_prior=False)
beta_val = transform_results["beta"]
val_assignments = transform_results["subtype"]
val_lse = subtyping_em._compute_val_score(X_val, beta_val)

# 7. Save literally everything
bic_value = subtyping_em.bic_
lse_final = float(subtyping_em.lse_final)
n_obs = int(subtyping_em.n_obs_)
bic_k = count_bic_params(
    subtyping_em.final_s,
    subtyping_em.final_kappa,
    subtyping_em.cluster_f,
    subtyping_em.n_subtypes,
    subtyping_em.lambda_cog,
    subtyping_em.cluster_cog_a,
)
bic_penalty = bic_k * np.log(n_obs)
bic_neg2_log_L = bic_value - bic_penalty

param_str = "_".join(
    f"{name}{val:.3f}".replace(".", "p") if isinstance(val, float) else f"{name}{val}"
    for name, val in params.items()
)
out_path = os.path.join(out_dir, f"result_{current_candidate:03d}_{param_str}.npz")

np.savez(
    out_path,
    # trajectories / history
    theta_history=np.array(subtyping_em.theta_history),
    beta_history=np.array(subtyping_em.beta_history),
    kappa_history=np.array(subtyping_em.kappa_history),
    lse_history=np.array(subtyping_em.lse_history),
    assignment_history=np.array(subtyping_em.assignment_history),
    # final fitted params
    final_kappa=np.array(subtyping_em.final_kappa),
    final_s=np.array(subtyping_em.final_s),
    final_scalar_K=subtyping_em.final_scalar_K,
    cluster_f=np.array(subtyping_em.cluster_f),
    cluster_cog_a=np.array(subtyping_em.cluster_cog_a),
    cluster_cog_b=np.array(subtyping_em.cluster_cog_b),
    final_assignments=subtyping_em.final_assignments,
    f_init=f_init,
    K=K,
    biomarker_names=np.array(X_COLS, dtype=object),
    disconnected=np.array(disconnected, dtype=object),
    # val results
    beta_val=np.array(beta_val),
    val_assignments=np.array(val_assignments),
    val_lse=val_lse,
    train_ids=np.array([p["id"] for p in X_train], dtype=object),
    val_ids=np.array([p["id"] for p in X_val], dtype=object),
    # provenance (so analyze_results.ipynb stays dumb)
    experiment_name=EXPERIMENT_CONFIG["name"],
    csv_path=CSV_PATH,
    connectome_path=CONNECTOME_PATH,
    id_col=ID_COL,
    time_col=TIME_COL,
    clinical_cols=np.array(CLINICAL_COLS, dtype=object),
    metadata_cols=np.array(METADATA_COLS, dtype=object),
    split_seed=SPLIT_SEED,
    test_size=TEST_SIZE,
    t_max=T_MAX,
    # candidate / grid bookkeeping
    candidate=current_candidate,
    k_idx=k_idx,
    sub_cand=sub_cand,
    n_subtypes=n_subtypes,
    n_subtypes_list=np.array(N_SUBTYPES_LIST),
    param_grid_size=total_combinations,
    n_hyper_per_K=n_hyper_per_K,
    max_iter=EXPERIMENT_CONFIG["max_iter"],
    theta_solver_stages=np.array(EXPERIMENT_CONFIG["theta_solver_stages"], dtype=object),
    # hyperparameters
    lambda_f=params["lambda_f"],
    lambda_cog=params["lambda_cog"],
    lambda_scalar=params["lambda_scalar"],
    scalar_K_center=params["scalar_K_center"],
    lambda_kappa=params["lambda_kappa"],
    lambda_jsd=params["lambda_jsd"],
    lambda_beta=params["lambda_beta"],
    # fit quality
    bic=bic_value,
    bic_neg2_log_L=bic_neg2_log_L,
    bic_penalty=bic_penalty,
    bic_n_params=bic_k,
    n_obs=n_obs,
    lse_final=lse_final,
    sse_per_biomarker=np.array(subtyping_em._sse_per_biomarker),
    n_obs_rows=int(subtyping_em._n_obs_rows),
)

print(f"\nSaved: {out_path}")
print(f"Final LSE: {lse_final:.6f}, BIC: {bic_value:.4f}, val LSE: {val_lse:.6f}")