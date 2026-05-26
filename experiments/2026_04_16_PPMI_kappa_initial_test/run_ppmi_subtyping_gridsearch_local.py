"""
Local-friendly PPMI subtyping grid search runner.

- Resolves COMIND code as a sibling of COMIND_experiments (no hard-coded home path).
- Data paths default from env PPMI_DATA_ROOT or --data-root.
- --skip-cv: skip 3-fold CV (faster smoke test; cv_* fields in npz are NaN / empty).
- --quick: single subtype K=2 and one hyperparameter tuple (minimal grid).

The main cluster script is run_ppmi_subtyping_gridsearch.py; this file is for IDE / laptop runs.
"""
from __future__ import annotations

import argparse
import os
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

# ---------------------------------------------------------------------------
# Paths: experiment dir -> COMIND_experiments -> parent -> sibling COMIND
# ---------------------------------------------------------------------------
EXPERIMENT_DIR = Path(__file__).resolve().parent
# .../<workspace>/COMIND_experiments/<this_experiment>/this_script.py
_WORKSPACE_ROOT = EXPERIMENT_DIR.parent.parent
DEFAULT_COMIND_ROOT = _WORKSPACE_ROOT / "COMIND"
DEFAULT_DATA_ROOT = Path(os.environ.get("PPMI_DATA_ROOT", "/home/dsemchin/data"))

parser = argparse.ArgumentParser(description="PPMI subtyping grid search (local-friendly)")
parser.add_argument("--candidate", type=int, default=0, help="Grid index (same mapping as cluster script)")
parser.add_argument(
    "--comind-root",
    type=Path,
    default=None,
    help=f"Path to COMIND repo (default: {DEFAULT_COMIND_ROOT})",
)
parser.add_argument(
    "--data-root",
    type=Path,
    default=DEFAULT_DATA_ROOT,
    help="Directory containing data_ppmi_pd.csv and connectome CSV tree",
)
parser.add_argument("--skip-cv", action="store_true", help="Skip K-fold CV before final fit")
parser.add_argument(
    "--quick",
    action="store_true",
    help="Minimal grid: n_subtypes=2 only, single hyperparameter combo",
)
args = parser.parse_args()
current_candidate = args.candidate

comind_root = args.comind_root or DEFAULT_COMIND_ROOT
if not comind_root.is_dir():
    raise FileNotFoundError(
        f"COMIND root not found: {comind_root}. Clone COMIND next to COMIND_experiments or pass --comind-root."
    )
sys.path.insert(0, str(comind_root.resolve()))

from COMIND_transformer.subtyping_em_transformer import SubtypingEM
from COMIND_transformer.utils import fit_mixedlm_beta_from_clinical, initialize_f_eigen

N_FOLDS = 3
CV_RANDOM_STATE = 75

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
data_root = Path(args.data_root)
csv_ppmi = data_root / "data_ppmi_pd.csv"
csv_K = (
    data_root
    / "iit_connectivity_matrix"
    / "streamline_normalized_top_regions"
    / "iit_connectome_top_20.csv"
)
if not csv_ppmi.is_file():
    raise FileNotFoundError(f"Missing PPMI table: {csv_ppmi} (set PPMI_DATA_ROOT or --data-root)")
if not csv_K.is_file():
    raise FileNotFoundError(f"Missing connectome CSV: {csv_K}")

df = pd.read_csv(csv_ppmi)
df_K = pd.read_csv(csv_K)

print("original size:", df.shape)
relevant_cols = [
    col
    for col in df.columns
    if col.startswith(("L_", "R_")) and ("_thickavg" in col or "_thickavg_resid" in col)
]
relevant_cols += ["MCATOT", "TD_score", "PIGD_score"]
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=relevant_cols)
print("after drop na", df.shape)

subj_counts = df["subj_id"].value_counts()
longitudinal_ids = subj_counts[subj_counts > 1].index
df = df[df["subj_id"].isin(longitudinal_ids)].copy()
df = df.drop_duplicates(subset=["subj_id", "time"])
print("after drop dupes", df.shape)

X_obs = df[
    [
        col
        for col in df.columns
        if col.startswith(("L_", "R_"))
        and col.endswith("_thickavg")
        and not col.endswith("_thickavg_resid")
    ]
]
biomarker_names = [
    col
    for col in df.columns
    if col.startswith(("L_", "R_"))
    and col.endswith("_thickavg")
    and not col.endswith("_thickavg_resid")
]
print("n biomarkers:", len(biomarker_names))

X_obs = X_obs.to_numpy()
X_obs = np.max(X_obs, axis=0) - X_obs

K = df_K.drop(df_K.columns[0], axis=1).to_numpy()
np.fill_diagonal(K, 0)
print("K shape:", K.shape)

t_max = 40.0
ids = df["subj_id"].to_numpy()
dt = df["time"].to_numpy() / 12.0
cog = df[["MCATOT", "TD_score", "PIGD_score"]].to_numpy()
nhy = df["NHY"].to_numpy()

df["NSD_STAGE"] = df["NSD_STAGE"].replace({"Not NSD": 0, "2b": 2})
df["NSD_STAGE"] = pd.to_numeric(df["NSD_STAGE"], errors="coerce")

initial_beta, _pid_to_beta, _ = fit_mixedlm_beta_from_clinical(
    df=df,
    ids=ids,
    dt=dt,
    t_max=t_max,
    verbose=True,
    rng=np.random.default_rng(75),
)

f_init_list = initialize_f_eigen(
    K=K,
    jitter_strength=0.05,
    n_eigs=100,
    rng=np.random.RandomState(75),
)
f_init = f_init_list[0]
if isinstance(f_init, list):
    f_init = f_init[0]
f_init = np.ravel(f_init)


def create_patient_list(X_obs, ids, dt, cog, initial_beta=None):
    unique_ids = np.unique(ids)
    id_to_index = {pid: idx for idx, pid in enumerate(unique_ids)}
    patient_list = []
    for pid in unique_ids:
        mask = ids == pid
        row = {
            "id": pid,
            "X_obs": X_obs[mask],
            "dt": dt[mask],
            "cog": cog[mask],
            "nhy": nhy[mask],
        }
        if initial_beta is not None:
            row["initial_beta"] = initial_beta[id_to_index[pid]]
        patient_list.append(row)
    return patient_list


X = create_patient_list(X_obs, ids, dt, cog, initial_beta)
X_train, X_val = train_test_split(X, test_size=0.2, random_state=75)

# ---------------------------------------------------------------------------
# Grid (full vs --quick)
# ---------------------------------------------------------------------------
if args.quick:
    N_SUBTYPES_LIST = [2]
    param_grid_hyper = {
        "lambda_f": [1.0],
        "lambda_cog": [0.0],
        "lambda_scalar": [5.0],
        "lambda_jsd": [0],
        "lambda_beta": [0.0],
    }
else:
    N_SUBTYPES_LIST = [2, 3, 4]
    param_grid_hyper = {
        "lambda_f": [1.0, 1.2, 1.4],
        "lambda_cog": [0.0, 0.01, 0.05],
        "lambda_scalar": [1.0, 2.0, 5.0, 10.0],
        "lambda_jsd": [0, 5, 10, 50],
        "lambda_beta": [0.0],
    }

hyper_names = list(param_grid_hyper.keys())
hyper_values = list(param_grid_hyper.values())
hyper_combinations = list(product(*hyper_values))
n_hyper_per_K = len(hyper_combinations)
total_combinations = len(N_SUBTYPES_LIST) * n_hyper_per_K

print("\n=== Grid Search Setup ===")
print("n_subtypes candidates:", N_SUBTYPES_LIST)
print("Hyperparameter combinations per K:", n_hyper_per_K)
print("Total combinations:", total_combinations)
print("Current candidate:", current_candidate)

if current_candidate >= total_combinations:
    sys.exit(0)

k_idx = current_candidate // n_hyper_per_K
sub_cand = current_candidate % n_hyper_per_K
n_subtypes = N_SUBTYPES_LIST[k_idx]
params = {name: hyper_combinations[sub_cand][i] for i, name in enumerate(hyper_names)}
params["n_subtypes"] = n_subtypes
params["lambda_kappa"] = params["lambda_f"]

print(f"\n=== Candidate {current_candidate} (K={n_subtypes}, hyper idx={sub_cand}) ===")
for name, value in params.items():
    print(f"  {name}: {value}")


def build_estimator(params, K, f_init, t_max, rng_seed=75):
    return SubtypingEM(
        K=K,
        initial_f=f_init,
        n_subtypes=params["n_subtypes"],
        jac_toggle=True,
        max_iter=200,
        t_max=t_max,
        step=0.01,
        epsilon=1e-2,
        lambda_f=params["lambda_f"],
        lambda_cog=params["lambda_cog"],
        lambda_scalar=params["lambda_scalar"],
        lambda_jsd=params["lambda_jsd"],
        lambda_beta=params["lambda_beta"],
        lambda_kappa=params["lambda_kappa"],
        verbose=1,
        rng=np.random.default_rng(rng_seed),
    )


if args.skip_cv:
    fold_lses: list[float] = []
    cv_mean_lse = float("nan")
    cv_std_lse = float("nan")
    print("Skipping K-fold CV (--skip-cv).")
else:
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE)
    fold_lses = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(X_train)))):
        X_train_fold = [X_train[i] for i in train_idx]
        X_val_fold = [X_train[i] for i in val_idx]
        em_fold = build_estimator(params, K, f_init, t_max, rng_seed=75 + current_candidate * 100 + fold_idx)
        em_fold.fit(X_train_fold)
        tr = em_fold.transform(X_val_fold, use_cognitive_prior=True)
        lse_fold = em_fold._compute_val_score(X_val_fold, tr["beta"])
        fold_lses.append(lse_fold)
        if em_fold.verbose >= 1:
            print(f"  Fold {fold_idx + 1}/{N_FOLDS} val LSE: {lse_fold:.6f}")
    cv_mean_lse = float(np.mean(fold_lses))
    cv_std_lse = float(np.std(fold_lses))
    print(f"CV mean LSE: {cv_mean_lse:.6f} ± {cv_std_lse:.6f} ({N_FOLDS} folds)")

subtyping_em = build_estimator(params, K, f_init, t_max, rng_seed=75)
subtyping_em.fit(X_train)

transform_results_with_cog = subtyping_em.transform(X_val, use_cognitive_prior=True)
transform_results_no_cog = subtyping_em.transform(X_val, use_cognitive_prior=False)

beta_val_with_cog = transform_results_with_cog["beta"]
val_assignments_with_cog = transform_results_with_cog["subtype"]
beta_val_no_cog = transform_results_no_cog["beta"]
val_assignments_no_cog = transform_results_no_cog["subtype"]

val_lse_with_cog = subtyping_em._compute_val_score(X_val, beta_val_with_cog)
val_lse_no_cog = subtyping_em._compute_val_score(X_val, beta_val_no_cog)

script_dir = EXPERIMENT_DIR
out_dir = script_dir / "results"
out_dir.mkdir(parents=True, exist_ok=True)

param_str = "_".join(
    f"{name}{val:.3f}".replace(".", "p") if isinstance(val, float) else f"{name}{val}"
    for name, val in params.items()
)
out_path = out_dir / f"PPMI_subtyping_grid_local_{current_candidate:03d}_{param_str}.npz"

bic_value = subtyping_em.bic_
lse_final = float(subtyping_em.lse_final)
n_obs = int(subtyping_em.n_obs_)
bic_k = int(subtyping_em._bic_n_params())
bic_penalty = bic_k * np.log(n_obs)
bic_neg2_log_L = bic_value - bic_penalty

train_ids = [p["id"] for p in X_train]
val_ids_unique = [p["id"] for p in X_val]
all_train_ids = np.array(train_ids)
all_val_ids = np.array(val_ids_unique)
train_assignments_array = subtyping_em.final_assignments

np.savez(
    out_path,
    theta_history=np.array(subtyping_em.theta_history),
    cog_history=np.array(subtyping_em.cog_regression_history),
    beta_history=np.array(subtyping_em.beta_history),
    kappa_history=np.array(subtyping_em.kappa_history),
    lse_history=np.array(subtyping_em.lse_history),
    assignment_history=np.array(subtyping_em.assignment_history),
    beta_val=np.array(beta_val_with_cog),
    beta_val_with_cog=np.array(beta_val_with_cog),
    beta_val_no_cog=np.array(beta_val_no_cog),
    val_assignments_with_cog=np.array(val_assignments_with_cog),
    val_assignments_no_cog=np.array(val_assignments_no_cog),
    val_lse_with_cog=val_lse_with_cog,
    val_lse_no_cog=val_lse_no_cog,
    candidate=current_candidate,
    f_init=f_init,
    train_assignments=train_assignments_array,
    val_assignments=val_assignments_with_cog,
    train_ids=all_train_ids,
    val_ids=all_val_ids,
    final_assignments=subtyping_em.final_assignments,
    cluster_f=np.array(subtyping_em.cluster_f),
    cluster_cog_a=np.array(subtyping_em.cluster_cog_a),
    cluster_cog_b=np.array(subtyping_em.cluster_cog_b),
    final_scalar_K=subtyping_em.final_scalar_K,
    final_s=subtyping_em.final_s,
    final_kappa=subtyping_em.final_kappa,
    n_subtypes=params["n_subtypes"],
    n_subtypes_list=np.array(N_SUBTYPES_LIST),
    lambda_f=params["lambda_f"],
    lambda_cog=params["lambda_cog"],
    lambda_scalar=params["lambda_scalar"],
    lambda_jsd=params["lambda_jsd"],
    lambda_beta=params["lambda_beta"],
    lambda_kappa=params["lambda_kappa"],
    param_grid_size=total_combinations,
    n_hyper_per_K=n_hyper_per_K,
    k_idx=k_idx,
    sub_cand=sub_cand,
    bic=bic_value,
    bic_neg2_log_L=bic_neg2_log_L,
    bic_penalty=bic_penalty,
    n_obs=n_obs,
    bic_n_params=bic_k,
    lse_final=lse_final,
    cv_mean_lse=cv_mean_lse,
    cv_std_lse=cv_std_lse,
    cv_variance=float(cv_std_lse**2) if np.isfinite(cv_std_lse) else np.nan,
    cv_per_fold_lse=np.array(fold_lses, dtype=float),
    n_folds=N_FOLDS if not args.skip_cv else 0,
    skip_cv=args.skip_cv,
    quick=args.quick,
    sse_per_biomarker=np.array(subtyping_em._sse_per_biomarker),
    n_obs_rows=int(subtyping_em._n_obs_rows),
)

print("Saved:", out_path)
print(f"Final Training LSE: {lse_final:.6f}")
print(f"BIC: {bic_value:.4f}")
print(f"Validation LSE (with cog): {val_lse_with_cog:.6f}")
print(f"Validation LSE (no cog): {val_lse_no_cog:.6f}")
