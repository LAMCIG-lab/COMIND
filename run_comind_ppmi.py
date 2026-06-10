"""
PPMI subtyping experiment driver.

Per-experiment defaults live in ``<exp-dir>/experiment_config.py``; CLI flags override.
"""
import argparse
import importlib.util
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from COMIND_transformer.model_selection import count_bic_params
from COMIND_transformer.subtyping_em_transformer import SubtypingEM
from COMIND_transformer.utils import fit_mixedlm_beta_from_clinical, initialize_f_eigen
from COMIND_transformer.warm_start import (
    load_legacy_presubtyping_from_npz,
    load_warm_start_from_npz,
)

N_FOLDS = 3
CV_RANDOM_STATE = 75

COMIND_ROOT = os.path.dirname(os.path.abspath(__file__))


def load_experiment_config(exp_dir: str) -> dict:
    path = os.path.join(exp_dir, "experiment_config.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"experiment_config.py not found in {exp_dir}")
    spec = importlib.util.spec_from_file_location("experiment_config", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.EXPERIMENT_CONFIG


parser = argparse.ArgumentParser()
parser.add_argument(
    "--exp-dir",
    type=str,
    required=True,
    help="Experiment folder containing experiment_config.py, results/, logs/.",
)
parser.add_argument("--candidate", type=int, default=0)
parser.add_argument(
    "--warm-start-npz",
    type=str,
    default=None,
    help="Warm-start checkpoint; empty disables. Defaults to EXPERIMENT_CONFIG.",
)
parser.add_argument(
    "--warm-start-format",
    choices=("subtyping", "legacy"),
    default=None,
    help="subtyping: modern npz; legacy: pre-subtyping miccai-style npz (Z=1).",
)
parser.add_argument(
    "--cold-start",
    action="store_true",
    help="Disable warm-start (ignore config and --warm-start-npz).",
)
parser.add_argument("--max-iter", type=int, default=None)
parser.add_argument(
    "--theta-solver-stages",
    type=str,
    default=None,
    help="Comma-separated stages, e.g. lbfgs_approx or lbfgs_exact",
)
parser.add_argument(
    "--skip-cv",
    action="store_true",
    default=None,
    help="Skip K-fold CV; fit once on full training set.",
)
parser.add_argument(
    "--profile",
    action="store_true",
    default=None,
    help="cProfile final fit; write .prof next to results npz.",
)
parser.add_argument("--jitter", action="store_true", default=None)
parser.add_argument("--no-jitter", action="store_true", help="Disable warm-start jitter.")
parser.add_argument("--jitter-strength", type=float, default=None)
parser.add_argument("--jitter-seed", type=int, default=None)
parser.add_argument(
    "--no-checkpoint",
    action="store_true",
    help="Disable per-iteration checkpoint during the final fit.",
)
parser.add_argument(
    "--checkpoint-path",
    type=str,
    default=None,
    help="Checkpoint npz for final fit (default: results/checkpoint_latest.npz).",
)
parser.add_argument(
    "--no-cv-parallel",
    action="store_true",
    help="Run K-fold CV folds sequentially instead of in parallel.",
)
parser.add_argument(
    "--cv-workers",
    type=int,
    default=None,
    help="Parallel CV worker count (default: PBS_NP or n_folds).",
)
parser.add_argument(
    "--strict-tol",
    action="store_true",
    help="Use tightened L-BFGS-B ftol/gtol (one order of magnitude vs scipy defaults).",
)
parser.add_argument(
    "--ode-method",
    choices=("LSODA", "BDF"),
    default=None,
    help="ODE integrator for forward trajectories.",
)
parser.add_argument("--n-anneal-iters", type=int, default=None)
parser.add_argument("--kappa-anneal-strength", type=float, default=None)
parser.add_argument("--f-anneal-strength", type=float, default=None)
parser.add_argument("--anneal-decay", type=float, default=None)
parser.add_argument("--n-subtypes", type=int, default=None, help="Override n_subtypes.")
parser.add_argument("--lambda-f", type=float, default=None)
parser.add_argument("--lambda-cog", type=float, default=None)
parser.add_argument("--lambda-scalar", type=float, default=None)
parser.add_argument("--lambda-jsd", type=float, default=None)
parser.add_argument("--lambda-beta", type=float, default=None)
parser.add_argument("--lambda-kappa", type=float, default=None)
args = parser.parse_args()

script_dir = os.path.abspath(args.exp_dir)
cfg = load_experiment_config(script_dir)
current_candidate = args.candidate

warm_start_npz = None if args.cold_start else args.warm_start_npz
if warm_start_npz is None and not args.cold_start:
    warm_start_npz = cfg.get("warm_start_npz")
warm_start_npz = warm_start_npz.strip() if warm_start_npz else None

warm_start_format = (
    args.warm_start_format
    if args.warm_start_format is not None
    else cfg.get("warm_start_format", "subtyping")
)
max_iter = args.max_iter if args.max_iter is not None else cfg["max_iter"]
if args.theta_solver_stages:
    theta_solver_stages = tuple(s.strip() for s in args.theta_solver_stages.split(","))
else:
    theta_solver_stages = tuple(cfg["theta_solver_stages"])
skip_cv = cfg["skip_cv"] if args.skip_cv is None else args.skip_cv
cv_parallel = cfg.get("cv_parallel", True) if not args.no_cv_parallel else False
cv_workers = args.cv_workers or cfg.get("cv_workers") or int(
    os.environ.get("PBS_NP", N_FOLDS)
)
ode_method = args.ode_method or cfg.get("ode_method", "LSODA")
if args.profile is None and ode_method == "BDF":
    use_profile = True
else:
    use_profile = cfg["profile_default"] if args.profile is None else args.profile
strict_tol = args.strict_tol
n_anneal_iters = (
    args.n_anneal_iters
    if args.n_anneal_iters is not None
    else cfg.get("n_anneal_iters", 0)
)
kappa_anneal_strength = (
    args.kappa_anneal_strength
    if args.kappa_anneal_strength is not None
    else cfg.get("kappa_anneal_strength", 0.05)
)
f_anneal_strength = (
    args.f_anneal_strength
    if args.f_anneal_strength is not None
    else cfg.get("f_anneal_strength", 0.01)
)
anneal_decay = (
    args.anneal_decay
    if args.anneal_decay is not None
    else cfg.get("anneal_decay", 0.7)
)
if args.no_jitter:
    use_jitter = False
else:
    use_jitter = cfg["jitter_default"] if args.jitter is None else args.jitter
jitter_strength = (
    args.jitter_strength
    if args.jitter_strength is not None
    else cfg.get("jitter_strength", 0.05)
)
jitter_seed = (
    args.jitter_seed if args.jitter_seed is not None else cfg.get("jitter_seed", 42)
)

print(f"\n=== Experiment: {cfg.get('name', script_dir)} ===")
print(f"  exp_dir={script_dir}")
print(f"  max_iter={max_iter}, theta_solver_stages={theta_solver_stages}")
print(
    f"  skip_cv={skip_cv}, cv_parallel={cv_parallel}, cv_workers={cv_workers}, "
    f"profile={use_profile}, jitter={use_jitter}, strict_tol={strict_tol}, "
    f"ode_method={ode_method}, n_anneal_iters={n_anneal_iters}"
)
print(f"  warm_start_npz={warm_start_npz or '(none/cold-start)'}")
print(f"  warm_start_format={warm_start_format}")

df = pd.read_csv("/home/dsemchin/data/data_ppmi_pd.csv")
df_K = pd.read_csv(
    "/home/dsemchin/data/iit_connectivity_matrix/streamline_normalized_top_regions/"
    "iit_connectome_top_20.csv"
)
n_biomarkers = 68

print("original size:", df.shape)
relevant_cols = [
    col
    for col in df.columns
    if col.startswith(("L_", "R_")) and ("_thickavg" in col or "_thickavg_resid" in col)
]
relevant_cols += ["MCATOT", "TD_score", "PIGD_score"]
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=relevant_cols)
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
biomarker_names = list(X_obs.columns)
X_obs = X_obs.to_numpy()
X_obs = np.max(X_obs, axis=0) - X_obs

K = df_K.drop(df_K.columns[0], axis=1).to_numpy()
np.fill_diagonal(K, 0)

t_max = 40
ids = df["subj_id"].to_numpy()
dt = df["time"].to_numpy() / 12
cog = df[["MCATOT", "TD_score", "PIGD_score"]].to_numpy()
nhy = df["NHY"].to_numpy()

initial_beta, pid_to_beta, _ = fit_mixedlm_beta_from_clinical(
    df=df, ids=ids, dt=dt, t_max=t_max, verbose=True, rng=np.random.default_rng(75)
)

f_init_list = initialize_f_eigen(
    K=K, jitter_strength=0.05, n_eigs=100, rng=np.random.RandomState(75)
)
f_init = f_init_list[0] if not isinstance(f_init_list[0], list) else f_init_list[0][0]
f_init = np.ravel(f_init)


def create_patient_list(X_obs_arr, ids_arr, dt_arr, cog_arr, initial_beta_arr=None):
    unique_ids = np.unique(ids_arr)
    id_to_index = {pid: idx for idx, pid in enumerate(unique_ids)}
    patient_list = []
    for pid in unique_ids:
        mask = ids_arr == pid
        patient_data = {
            "id": pid,
            "X_obs": X_obs_arr[mask],
            "dt": dt_arr[mask],
            "cog": cog_arr[mask],
            "nhy": nhy[mask],
        }
        if initial_beta_arr is not None:
            patient_data["initial_beta"] = initial_beta_arr[id_to_index[pid]]
        patient_list.append(patient_data)
    return patient_list


X = create_patient_list(X_obs, ids, dt, cog, initial_beta)
X_train, X_val = train_test_split(X, test_size=0.2, random_state=75)

warm_start_full = None
if warm_start_npz:
    if not os.path.isfile(warm_start_npz):
        raise FileNotFoundError(f"--warm-start-npz not found: {warm_start_npz}")
    train_patient_ids = [p["id"] for p in X_train]
    if warm_start_format == "legacy":
        warm_start_full = load_legacy_presubtyping_from_npz(
            warm_start_npz, train_patient_ids, n_biomarkers=n_biomarkers
        )
    else:
        warm_start_full = load_warm_start_from_npz(warm_start_npz, train_patient_ids)
    print(f"Warm-start loaded from: {warm_start_npz} ({warm_start_format})")
    print(
        f"  n_subtypes={warm_start_full['n_subtypes']}, "
        f"scalar_K={warm_start_full['initial_scalar_K']:.4f}"
    )

if warm_start_full is not None and use_jitter:
    _jrng = np.random.default_rng(jitter_seed)
    warm_start_full["initial_f"] = warm_start_full["initial_f"] * (
        1.0 + _jrng.uniform(0.0, jitter_strength, warm_start_full["initial_f"].shape)
    )
    warm_start_full["initial_kappa"] = _jrng.uniform(
        0.0, jitter_strength, warm_start_full["initial_kappa"].shape
    )
    print(
        f"Jitter applied: strength={jitter_strength}, seed={jitter_seed}  "
        f"(f multiplicative, kappa U(0, {jitter_strength}))"
    )

if args.n_subtypes is not None:
    N_SUBTYPES_LIST = [args.n_subtypes]
else:
    N_SUBTYPES_LIST = list(cfg["n_subtypes_list"])

param_grid_hyper = dict(cfg["param_grid_hyper"])
hyper_names = list(param_grid_hyper.keys())
hyper_values = list(param_grid_hyper.values())
hyper_combinations = list(product(*hyper_values))
n_hyper_per_K = len(hyper_combinations)
total_combinations = cfg.get("candidate_count", len(N_SUBTYPES_LIST) * n_hyper_per_K)

if current_candidate >= total_combinations:
    sys.exit(0)

k_idx = current_candidate // n_hyper_per_K
sub_cand = current_candidate % n_hyper_per_K
n_subtypes = N_SUBTYPES_LIST[k_idx]
params = {name: hyper_combinations[sub_cand][i] for i, name in enumerate(hyper_names)}
params["n_subtypes"] = n_subtypes
params["max_iter"] = max_iter
params["theta_solver_stages"] = theta_solver_stages
params["strict_tol"] = strict_tol
params["ode_method"] = ode_method
params["n_anneal_iters"] = n_anneal_iters
params["kappa_anneal_strength"] = kappa_anneal_strength
params["f_anneal_strength"] = f_anneal_strength
params["anneal_decay"] = anneal_decay

_lambda_cli = {
    "lambda_f": args.lambda_f,
    "lambda_cog": args.lambda_cog,
    "lambda_scalar": args.lambda_scalar,
    "lambda_jsd": args.lambda_jsd,
    "lambda_beta": args.lambda_beta,
    "lambda_kappa": args.lambda_kappa,
}
for name, val in _lambda_cli.items():
    if val is not None:
        params[name] = val

print(f"\n=== Candidate {current_candidate} (K={n_subtypes}) ===")
for name, value in params.items():
    print(f"  {name}: {value}")


def build_estimator(params_dict, K_mat, f_init_vec, t_max_val, rng_seed=75, warm_start=None):
    em_kwargs = dict(
        K=K_mat,
        initial_f=f_init_vec,
        n_subtypes=params_dict["n_subtypes"],
        max_iter=params_dict["max_iter"],
        theta_solver_stages=params_dict["theta_solver_stages"],
        t_max=t_max_val,
        step=0.01,
        epsilon=1e-2,
        lambda_f=params_dict["lambda_f"],
        lambda_cog=params_dict["lambda_cog"],
        lambda_scalar=params_dict["lambda_scalar"],
        lambda_kappa=params_dict.get("lambda_kappa", params_dict["lambda_f"]),
        lambda_jsd=params_dict["lambda_jsd"],
        lambda_beta=params_dict["lambda_beta"],
        strict_tol=params_dict.get("strict_tol", False),
        ode_method=params_dict.get("ode_method", "LSODA"),
        n_anneal_iters=params_dict.get("n_anneal_iters", 0),
        kappa_anneal_strength=params_dict.get("kappa_anneal_strength", 0.05),
        f_anneal_strength=params_dict.get("f_anneal_strength", 0.01),
        anneal_decay=params_dict.get("anneal_decay", 0.7),
        verbose=1,
        rng=np.random.default_rng(rng_seed),
    )
    if warm_start is not None:
        if warm_start["n_subtypes"] != params_dict["n_subtypes"]:
            raise ValueError(
                f"warm-start n_subtypes={warm_start['n_subtypes']} != "
                f"params n_subtypes={params_dict['n_subtypes']}"
            )
        em_kwargs.update(
            initial_f=warm_start["initial_f"],
            initial_s=warm_start["initial_s"],
            initial_scalar_K=warm_start["initial_scalar_K"],
            initial_kappa=warm_start["initial_kappa"],
            initial_assignments=warm_start["initial_assignments"],
            initial_cluster_cog_a=warm_start["initial_cluster_cog_a"],
            initial_cluster_cog_b=warm_start["initial_cluster_cog_b"],
            initial_beta=warm_start["initial_beta"],
        )
    return SubtypingEM(**em_kwargs)


def _subtyping_em_from_params(params_dict, K_mat, f_init_vec, t_max_val, rng_seed):
    return build_estimator(params_dict, K_mat, f_init_vec, t_max_val, rng_seed=rng_seed)


def _run_cv_fold(task):
    fold_idx, train_idx, val_idx, X_train, params, K, f_init, t_max_val, rng_seed = task
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    train_idx = np.asarray(train_idx, dtype=int)
    val_idx = np.asarray(val_idx, dtype=int)
    X_train_fold = [X_train[i] for i in train_idx]
    X_val_fold = [X_train[i] for i in val_idx]

    em_fold = _subtyping_em_from_params(params, K, f_init, t_max_val, rng_seed)
    em_fold.fit(X_train_fold)
    tr = em_fold.transform(X_val_fold, use_cognitive_prior=True)
    lse_fold = em_fold._compute_val_score(X_val_fold, tr["beta"])
    return fold_idx, float(lse_fold)


out_dir = os.path.join(script_dir, "results")
os.makedirs(out_dir, exist_ok=True)
checkpoint_path = None
if not args.no_checkpoint:
    checkpoint_path = args.checkpoint_path or os.path.join(out_dir, "checkpoint_latest.npz")

fold_lses = []
if skip_cv:
    print("Skipping K-fold CV (--skip-cv / experiment default).")
    cv_mean_lse = float("nan")
    cv_std_lse = float("nan")
else:
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE)
    fold_tasks = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(X_train)))):
        fold_tasks.append(
            (
                fold_idx,
                train_idx.tolist(),
                val_idx.tolist(),
                X_train,
                params,
                K,
                f_init,
                t_max,
                75 + current_candidate * 100 + fold_idx,
            )
        )

    if cv_parallel and cv_workers > 1:
        print(
            f"K-fold CV: {N_FOLDS} folds in parallel "
            f"(workers={min(cv_workers, N_FOLDS)}, no checkpoint)"
        )
        fold_lses = [float("nan")] * N_FOLDS
        with ProcessPoolExecutor(max_workers=min(cv_workers, N_FOLDS)) as executor:
            futures = [executor.submit(_run_cv_fold, task) for task in fold_tasks]
            for fut in as_completed(futures):
                fold_idx, lse_fold = fut.result()
                fold_lses[fold_idx] = lse_fold
                print(f"  Fold {fold_idx + 1}/{N_FOLDS} val LSE: {lse_fold:.6f}")
    else:
        print(f"K-fold CV: {N_FOLDS} folds sequential (no checkpoint)")
        for task in fold_tasks:
            fold_idx, lse_fold = _run_cv_fold(task)
            fold_lses.append(lse_fold)
            print(f"  Fold {fold_idx + 1}/{N_FOLDS} val LSE: {lse_fold:.6f}")

    cv_mean_lse = float(np.mean(fold_lses))
    cv_std_lse = float(np.std(fold_lses))
    print(f"CV mean LSE: {cv_mean_lse:.6f} ± {cv_std_lse:.6f}")

subtyping_em = build_estimator(
    params, K, f_init, t_max, rng_seed=75, warm_start=warm_start_full
)
if checkpoint_path:
    print(f"Final fit checkpoint: {checkpoint_path}")

if use_profile:
    import cProfile

    _profiler = cProfile.Profile()
    _profiler.enable()

subtyping_em.fit(X_train, checkpoint_path=checkpoint_path)

if use_profile:
    _profiler.disable()

transform_results_with_cog = subtyping_em.transform(X_val, use_cognitive_prior=True)
transform_results_no_cog = subtyping_em.transform(X_val, use_cognitive_prior=False)
beta_val_with_cog = transform_results_with_cog["beta"]
val_assignments_with_cog = transform_results_with_cog["subtype"]
beta_val_no_cog = transform_results_no_cog["beta"]
val_assignments_no_cog = transform_results_no_cog["subtype"]
val_lse_with_cog = subtyping_em._compute_val_score(X_val, beta_val_with_cog)
val_lse_no_cog = subtyping_em._compute_val_score(X_val, beta_val_no_cog)

param_str = "_".join(
    [
        f"{name}{val:.3f}".replace(".", "p")
        if isinstance(val, float)
        else f"{name}{val}"
        for name, val in params.items()
        if name not in (
            "max_iter",
            "theta_solver_stages",
            "strict_tol",
            "ode_method",
            "n_anneal_iters",
            "kappa_anneal_strength",
            "f_anneal_strength",
            "anneal_decay",
        )
    ]
)
out_path = os.path.join(
    out_dir, f"PPMI_subtyping_grid_betajsd_{current_candidate:03d}_{param_str}.npz"
)

if use_profile:
    prof_path = out_path.replace(".npz", ".prof")
    _profiler.dump_stats(prof_path)
    print(f"Profile written: {prof_path}")

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

train_ids = [p["id"] for p in X_train]
all_train_ids = np.array(train_ids)
all_val_ids = np.array([p["id"] for p in X_val])

np.savez(
    out_path,
    theta_history=np.array(subtyping_em.theta_history),
    cog_history=np.array(subtyping_em.cog_regression_history),
    beta_history=np.array(subtyping_em.beta_history),
    kappa_history=np.array(subtyping_em.kappa_history),
    final_kappa=np.array(subtyping_em.final_kappa),
    lse_history=np.array(subtyping_em.lse_history),
    iter_times=np.array(getattr(subtyping_em, "iter_times", [])),
    accepted_solver_stages=np.array(
        getattr(subtyping_em, "accepted_solver_stages", []), dtype=object
    ),
    assign_changes=np.array(getattr(subtyping_em, "assign_changes", [])),
    assignment_history=np.array(subtyping_em.assignment_history),
    beta_val=np.array(beta_val_with_cog),
    beta_val_with_cog=np.array(beta_val_with_cog),
    beta_val_no_cog=np.array(beta_val_no_cog),
    val_assignments_with_cog=np.array(val_assignments_with_cog),
    val_assignments_no_cog=np.array(val_assignments_no_cog),
    val_lse_with_cog=val_lse_with_cog,
    val_lse_no_cog=val_lse_no_cog,
    candidate=current_candidate,
    experiment_name=cfg.get("name", ""),
    warm_start_npz=warm_start_npz or "",
    warm_start_format=warm_start_format,
    jitter_applied=use_jitter,
    jitter_strength=jitter_strength if use_jitter else 0.0,
    jitter_seed=jitter_seed if use_jitter else -1,
    skip_cv=skip_cv,
    cv_parallel=cv_parallel,
    cv_workers=cv_workers,
    checkpoint_path=checkpoint_path or "",
    max_iter=max_iter,
    theta_solver_stages=np.array(theta_solver_stages, dtype=object),
    f_init=f_init,
    train_assignments=subtyping_em.final_assignments,
    val_assignments=val_assignments_with_cog,
    train_ids=all_train_ids,
    val_ids=all_val_ids,
    final_assignments=subtyping_em.final_assignments,
    cluster_f=np.array(subtyping_em.cluster_f),
    cluster_cog_a=np.array(subtyping_em.cluster_cog_a),
    cluster_cog_b=np.array(subtyping_em.cluster_cog_b),
    final_scalar_K=subtyping_em.final_scalar_K,
    final_s=subtyping_em.final_s,
    biomarker_names=np.array(biomarker_names, dtype=object),
    n_subtypes=params["n_subtypes"],
    n_subtypes_list=np.array(N_SUBTYPES_LIST),
    lambda_f=params["lambda_f"],
    lambda_cog=params["lambda_cog"],
    lambda_scalar=params["lambda_scalar"],
    lambda_kappa=params.get("lambda_kappa", params["lambda_f"]),
    lambda_jsd=params["lambda_jsd"],
    strict_tol=strict_tol,
    ode_method=ode_method,
    n_anneal_iters=n_anneal_iters,
    kappa_anneal_strength=kappa_anneal_strength,
    f_anneal_strength=f_anneal_strength,
    anneal_decay=anneal_decay,
    lambda_beta=params["lambda_beta"],
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
    cv_variance=float(cv_std_lse**2) if np.isfinite(cv_std_lse) else float("nan"),
    cv_per_fold_lse=np.array(fold_lses),
    n_folds=0 if skip_cv else N_FOLDS,
    sse_per_biomarker=np.array(subtyping_em._sse_per_biomarker),
    n_obs_rows=int(subtyping_em._n_obs_rows),
)

print("Saved:", out_path)
print(f"Final LSE: {lse_final:.6f}, BIC: {bic_value:.4f}")
print(f"Final kappa: {subtyping_em.final_kappa}")
