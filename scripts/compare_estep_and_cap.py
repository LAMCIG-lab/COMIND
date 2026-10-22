"""
Compare the per-patient β cap and the joint grid E-step on synthetic data.

Runs the 2 × 2 grid ``cap_beta_by_followup ∈ {False, True}`` ×
``estep_mode ∈ {"current", "joint_grid"}`` on one synthetic cohort with
unequal subtype sizes, using the same random initializations (seeds) in every
condition, and reports subtype recovery, subtype sizes and β recovery.

Usage::

    python scripts/compare_estep_and_cap.py --backend jax --n-seeds 5 --csv out.csv

or from a notebook::

    from scripts.compare_estep_and_cap import run_comparison
    summary, per_iter = run_comparison(backend="jax", n_seeds=5)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from COMIND_transformer.subtyping_em_transformer import SubtypingEM  # noqa: E402
from COMIND_transformer.utils import solve_system  # noqa: E402

CONDITIONS = (
    (False, "current"),
    (False, "joint_grid"),
    (True, "current"),
    (True, "joint_grid"),
)


def make_cohort(
    n_patients: int = 200,
    n_biomarkers: int = 6,
    sizes: tuple = (0.5, 0.3, 0.2),
    t_max: float = 12.0,
    step: float = 0.1,
    scalar_K: float = 0.8,
    kappa: float = 0.1,
    forcing: float = 0.3,
    noise: float = 0.03,
    seed: int = 0,
) -> dict:
    """
    Synthetic cohort with unequal subtype sizes and patients deep in the plateau.

    Each subtype has a single nonzero forcing entry at a different biomarker;
    K, ``scalar_K`` and ``kappa`` are shared. True β is uniform on
    ``[0, t_max - 3]`` (the longest follow-up is 3 years), which covers the
    whole disease course including the plateau. Each patient has 2–4 yearly
    visits with Gaussian noise.

    Parameters
    ----------
    n_patients, n_biomarkers : int
        Cohort size and number of biomarkers.
    sizes : tuple of float
        Subtype proportions; the number of subtypes is ``len(sizes)``.
    t_max, step : float
        Model time window and grid step.
    scalar_K, kappa, forcing : float
        Shared coupling, shared self-coupling and the forcing value of each
        subtype's single forced biomarker.
    noise : float
        Observation noise standard deviation.
    seed : int
        Seed for K, labels, β, visits and noise.

    Returns
    -------
    dict
        ``X`` (list of patient dicts with ``subtype_true``), ``K``,
        ``beta_true`` (n_patients,), ``subtype_true`` (n_patients,),
        ``t_plateau`` (time at which every biomarker of every subtype has
        reached 95 % of its asymptote), ``t_span`` and ``trajectories``
        (n_subtypes, n_biomarkers, len(t_span)).
    """
    rng = np.random.default_rng(seed)
    n_subtypes = len(sizes)
    K = rng.random((n_biomarkers, n_biomarkers))
    np.fill_diagonal(K, 0.0)
    K /= K.sum(axis=1, keepdims=True)
    kappa_vec = np.full(n_biomarkers, kappa)
    forced = np.round(np.linspace(0, n_biomarkers - 1, n_subtypes)).astype(int)
    F_true = np.zeros((n_subtypes, n_biomarkers))
    F_true[np.arange(n_subtypes), forced] = forcing

    t_span = np.linspace(0.0, t_max, int(t_max / step))
    trajectories = np.stack(
        [solve_system(np.zeros(n_biomarkers), f, K, t_span, scalar_K, kappa_vec) for f in F_true]
    )
    t_long = np.arange(0.0, 10.0 * t_max, step)
    t_plateau = 0.0
    for f in F_true:
        x_long = solve_system(np.zeros(n_biomarkers), f, K, t_long, scalar_K, kappa_vec)
        reached = x_long >= 0.95 * x_long[:, -1:]
        t_plateau = max(t_plateau, float(np.max(t_long[np.argmax(reached, axis=1)])))

    counts = np.floor(np.asarray(sizes) / np.sum(sizes) * n_patients).astype(int)
    counts[0] += n_patients - counts.sum()
    subtype_true = rng.permutation(np.repeat(np.arange(n_subtypes), counts))
    max_visits = 4
    beta_true = rng.uniform(0.0, t_max - (max_visits - 1), n_patients)

    X = []
    for i in range(n_patients):
        z = int(subtype_true[i])
        dt = np.arange(int(rng.integers(2, max_visits + 1)), dtype=float)
        x_obs = np.array(
            [np.interp(beta_true[i] + dt, t_span, trajectories[z, b]) for b in range(n_biomarkers)]
        ).T + rng.normal(0.0, noise, size=(dt.size, n_biomarkers))
        X.append(dict(id=i, X_obs=x_obs, dt=dt, cog=np.zeros((dt.size, 1)), subtype_true=z))

    return dict(X=X, K=K, beta_true=beta_true, subtype_true=subtype_true, t_plateau=t_plateau,
                t_span=t_span, trajectories=trajectories)


def _label_permutation(fitted: np.ndarray, true: np.ndarray, n_subtypes: int) -> np.ndarray:
    """``perm[fitted_label] = true_label`` maximizing overlap (Hungarian), defined for empty labels too."""
    overlap = np.zeros((n_subtypes, n_subtypes))
    np.add.at(overlap, (fitted, true), 1)
    rows, cols = linear_sum_assignment(-overlap)
    perm = np.empty(n_subtypes, dtype=int)
    perm[rows] = cols
    return perm


def run_comparison(
    backend: str = "numpy",
    n_seeds: int = 5,
    cohort: dict = None,
    cohort_seed: int = 0,
    n_patients: int = 200,
    max_iter: int = 30,
    conditions: tuple = CONDITIONS,
    verbose: bool = True,
    **em_kwargs,
) -> tuple:
    """
    Fit every (cap, E-step) condition with ``n_seeds`` random initializations.

    Parameters
    ----------
    backend : {"numpy", "jax"}
        Fit backend. The NumPy backend uses ``theta_solver_stages=("lbfgs_approx",)``
        unless overridden in ``em_kwargs``.
    n_seeds : int
        Random initializations per condition; seed ``k`` gives
        ``rng=np.random.default_rng(k)`` in every condition.
    cohort : dict, optional
        Output of ``make_cohort``. Default: ``make_cohort(n_patients, seed=cohort_seed)``.
    cohort_seed, n_patients : int
        Used only when ``cohort`` is None.
    max_iter : int
        Outer EM iterations.
    conditions : tuple of (bool, str)
        ``(cap_beta_by_followup, estep_mode)`` pairs.
    verbose : bool
        Print one line per run and the condition means.
    **em_kwargs
        Extra ``SubtypingEM`` arguments (override the defaults used here).

    Returns
    -------
    summary : pandas.DataFrame
        One row per (condition, seed): ``ari``, ``size_true_k``, ``size_fit_k``,
        ``size_abs_err_k`` (fractions of the cohort, fitted labels matched to
        true labels), ``size_abs_err_mean``, ``beta_mae``, ``beta_mae_plateau``,
        ``n_plateau``, ``beta_at_upper_fraction`` (final iteration),
        ``estep_beta_moves_total``, ``assign_changes_total``, ``lse_final``,
        ``bic``, ``outer_iters``, ``wall_s``; ``fit_error`` if the fit raised.
    per_iter : pandas.DataFrame
        Long format, one row per (condition, seed, iteration, matched subtype):
        ``count`` and ``fraction`` from ``subtype_counts_history_``
        (iteration 0 is the initialization).
    """
    if cohort is None:
        cohort = make_cohort(n_patients=n_patients, seed=cohort_seed)
    X, K = cohort["X"], cohort["K"]
    beta_true, true = cohort["beta_true"], np.asarray(cohort["subtype_true"])
    t_max = float(cohort["t_span"][-1])
    step = float(cohort["t_span"][1] - cohort["t_span"][0])
    n_subtypes = cohort["trajectories"].shape[0]
    plateau = beta_true >= cohort["t_plateau"]
    size_true = np.bincount(true, minlength=n_subtypes) / true.size

    base = dict(K=K, n_subtypes=n_subtypes, t_max=t_max, step=step, backend=backend,
                max_iter=max_iter, verbose=0)
    if backend == "numpy":
        base["theta_solver_stages"] = ("lbfgs_approx",)
    base.update(em_kwargs)

    rows, iter_rows = [], []
    for cap, estep_mode in conditions:
        for seed in range(n_seeds):
            row = dict(backend=backend, cap_beta_by_followup=cap, estep_mode=estep_mode, seed=seed)
            em = SubtypingEM(**base, cap_beta_by_followup=cap, estep_mode=estep_mode,
                             rng=np.random.default_rng(seed))
            t0 = time.perf_counter()
            try:
                em.fit(X)
            except Exception as exc:
                row["fit_error"] = repr(exc)
                rows.append(row)
                continue
            row["wall_s"] = time.perf_counter() - t0

            fitted = np.asarray(em.final_assignments, dtype=int)
            perm = _label_permutation(fitted, true, n_subtypes)
            size_fit = np.bincount(perm[fitted], minlength=n_subtypes) / fitted.size
            beta_err = np.abs(em.beta_history[:, -1] - beta_true)
            row["ari"] = float(adjusted_rand_score(true, fitted))
            for k in range(n_subtypes):
                row[f"size_true_{k}"] = float(size_true[k])
                row[f"size_fit_{k}"] = float(size_fit[k])
                row[f"size_abs_err_{k}"] = float(abs(size_fit[k] - size_true[k]))
            row["size_abs_err_mean"] = float(np.mean(np.abs(size_fit - size_true)))
            row["beta_mae"] = float(np.mean(beta_err))
            row["beta_mae_plateau"] = float(np.mean(beta_err[plateau])) if plateau.any() else np.nan
            row["n_plateau"] = int(plateau.sum())
            row["beta_at_upper_fraction"] = float(em.beta_at_upper_fraction_history_[-1])
            row["estep_beta_moves_total"] = int(np.sum(em.estep_beta_moves_history_))
            row["assign_changes_total"] = int(np.sum(em.assign_changes))
            row["lse_final"] = float(em.lse_final)
            row["bic"] = float(em.bic_)
            row["outer_iters"] = len(em.lse_history) - 1
            rows.append(row)

            counts = np.asarray(em.subtype_counts_history_)
            for it in range(counts.shape[1]):
                for k_fit in range(n_subtypes):
                    iter_rows.append(dict(
                        backend=backend, cap_beta_by_followup=cap, estep_mode=estep_mode,
                        seed=seed, iteration=it, subtype=int(perm[k_fit]),
                        count=int(counts[k_fit, it]),
                        fraction=float(counts[k_fit, it] / fitted.size),
                    ))
            if verbose:
                print(f"cap={cap!s:5} estep={estep_mode:10} seed={seed}  ARI={row['ari']:.3f}  "
                      f"size_err={row['size_abs_err_mean']:.3f}  beta_MAE={row['beta_mae']:.2f} "
                      f"(plateau {row['beta_mae_plateau']:.2f})  at_upper="
                      f"{row['beta_at_upper_fraction']:.2f}  {row['wall_s']:.1f}s", flush=True)

    summary = pd.DataFrame(rows)
    per_iter = pd.DataFrame(iter_rows)
    if verbose and "ari" in summary:
        cols = ["ari", "size_abs_err_mean", "beta_mae", "beta_mae_plateau",
                "beta_at_upper_fraction", "estep_beta_moves_total", "assign_changes_total",
                "lse_final", "bic", "outer_iters", "wall_s"]
        with pd.option_context("display.width", 200, "display.max_columns", None):
            print(summary.groupby(["cap_beta_by_followup", "estep_mode"])[cols].mean().round(3))
    return summary, per_iter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument("--backend", choices=("numpy", "jax"), default="numpy")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--n-patients", type=int, default=200)
    parser.add_argument("--cohort-seed", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--csv", type=str, default=None,
                        help="Write the per-run table here and the per-iteration "
                             "table next to it with suffix _per_iter.csv.")
    args = parser.parse_args()

    summary, per_iter = run_comparison(
        backend=args.backend, n_seeds=args.n_seeds, n_patients=args.n_patients,
        cohort_seed=args.cohort_seed, max_iter=args.max_iter,
    )
    if args.csv:
        summary.to_csv(args.csv, index=False)
        per_iter.to_csv(str(Path(args.csv).with_suffix("")) + "_per_iter.csv", index=False)


if __name__ == "__main__":
    main()
