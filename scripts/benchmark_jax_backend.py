"""
Benchmark the SubtypingEM backends.

Compares NumPy ``lbfgs_approx``, NumPy ``lbfgs_exact`` and the JAX backend:

- first-call compile time (JAX only; ``_traj_jit`` and each value_and_grad)
- mean wall time per loss+grad evaluation for the globals, f and beta steps
- per M-step (from a shared initial state): wall time, nfev, nit, message
- full fit: wall time, outer iterations, ``lse_final``, ``bic_``, ARI
- final reconstruction loss re-evaluated with LSODA for every backend

Usage
-----
From the command line (synthetic data)::

    python scripts/benchmark_jax_backend.py --n-patients 100 --n-biomarkers 8

From a notebook::

    import sys; sys.path.append("/path/to/COMIND/scripts")
    from benchmark_jax_backend import run_benchmark
    df = run_benchmark(X, K, n_subtypes=2, t_max=12, step=0.1, lambda_kappa=0.2)
"""

import argparse
import contextlib
import time
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from COMIND_transformer import optimizer_beta, optimizer_theta_cluster, optimizer_theta_globals
from COMIND_transformer.subtyping_em_transformer import SubtypingEM
from COMIND_transformer.utils import solve_system

BACKENDS = [
    ("numpy:lbfgs_approx", "numpy", ("lbfgs_approx",)),
    ("numpy:lbfgs_exact", "numpy", ("lbfgs_exact",)),
    ("jax", "jax", None),
]


@contextlib.contextmanager
def _capture_minimize(modules):
    """Record every scipy ``minimize`` result produced inside ``modules``."""
    results = []
    originals = {m: m.minimize for m in modules}

    def wrapped(*args, **kwargs):
        res = originals[modules[0]](*args, **kwargs)
        results.append(res)
        return res

    for m in modules:
        m.minimize = wrapped
    try:
        yield results
    finally:
        for m, fn in originals.items():
            m.minimize = fn


def _summarize(results) -> dict:
    return dict(
        nfev=int(sum(r.nfev for r in results)),
        nit=int(sum(r.nit for r in results)),
        message=" | ".join(sorted({str(r.message) for r in results})),
    )


def _mean_time(fn: Callable[[], object], n_repeat: int) -> float:
    fn()
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        fn()
    return (time.perf_counter() - t0) / n_repeat


def _make_em(K, n_subtypes, t_max, step, backend, stages, seed, em_kwargs):
    kw = dict(em_kwargs)
    if stages is not None:
        kw["theta_solver_stages"] = stages
    return SubtypingEM(
        K=K, n_subtypes=n_subtypes, t_max=t_max, step=step, backend=backend,
        rng=np.random.default_rng(seed), verbose=0, **kw,
    )


def _initial_state(em, X):
    em.t_span = np.linspace(0.0, em.t_max, int(em.t_max / em.step))
    flat = em._prepare_data(X)
    state = em._initialize_state(flat, X)
    return flat, state


def _jax_compile_times(em, flat, state) -> dict:
    """First-call (compile + run) and steady-state times of the jitted functions."""
    import jax

    from COMIND_transformer import jax_core as jc
    from COMIND_transformer import optimizer_jax as oj

    jax.clear_caches()
    data, meta = jc.pack_data(
        flat["X_obs"], flat["obs_weight"], flat["dt"], flat["ids"], flat["cog"],
        flat["n_patients"], pad_rows=em.jax_pad_rows, pad_patients=em.jax_pad_patients,
    )
    P = meta["P"]
    _, h, n_steps = jc.grid_from_t_span(em.t_span)
    h = jc.f64(h)
    F = jc.f64(np.stack([np.ravel(f) for f in state["cluster_f"]]))
    s = jc.f64(state["current_s"])
    kappa = jc.f64(state["current_kappa"])
    sK = jc.f64(state["current_scalar_K"])
    K = jc.f64(em.K)
    beta_full = jc.pad_beta(state["current_beta"], P)
    assign_full = jc.pad_assignments(state["assignments"], P)
    hyp = jc.make_hyp(lambda_kappa=em.lambda_kappa, lambda_scalar=em.lambda_scalar,
                      scalar_K_center=em.scalar_K_center, lambda_f=em.lambda_f,
                      lambda_cog=em.lambda_cog, lambda_jsd=em.lambda_jsd)
    A, b = oj._stack_cog(state["cluster_cog_a"], state["cluster_cog_b"], F.shape[0],
                         data.cog.shape[1])
    n_bins = max(20, int(em.t_max * 2))
    s_free = jc.f64(np.ones(len(state["current_s"])))

    def traj_call():
        return jc._traj_jit(F, kappa, sK, K, h, n_steps=n_steps)

    traj = traj_call()
    calls = {
        "traj": traj_call,
        "globals_vg": lambda: jc._globals_vg(
            s, kappa, sK, F, beta_full, assign_full, data, K, hyp, h,
            n_steps=n_steps, s_free=s_free),
        "f_vg": lambda: jc._cluster_vg(
            F, s, kappa, sK, beta_full, assign_full, data, K, hyp, h, n_steps=n_steps),
        "beta_vg": lambda: jc._beta_vg(
            beta_full, traj, s, assign_full, jc.f64(A), jc.f64(b), data, hyp, h,
            jc.f64(0.0), jc.f64(em.t_max), jc.f64(1.0),
            n_bins=n_bins, use_jsd=em.lambda_jsd > 0, auto_bandwidth=True),
    }
    jax.clear_caches()
    out = {}
    for name, fn in calls.items():
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        out[f"first_call_{name}_s"] = time.perf_counter() - t0
    traj = traj_call()
    eval_ms = {
        "eval_globals_ms": 1e3 * _mean_time(lambda: jax.block_until_ready(calls["globals_vg"]()), 20),
        "eval_f_ms": 1e3 * _mean_time(lambda: jax.block_until_ready(calls["f_vg"]()), 20),
        "eval_beta_ms": 1e3 * _mean_time(lambda: jax.block_until_ready(calls["beta_vg"]()), 20),
    }
    out.update(eval_ms)
    return out


def _numpy_eval_times(em, flat, state, method: str, n_repeat: int = 3) -> dict:
    """Mean loss+grad evaluation time for the NumPy objectives."""
    from COMIND_transformer.optimizer_beta import _vectorized_beta_loss_and_grad

    t_span = em.t_span
    ids, dt, X_obs, w = flat["ids"], flat["dt"], flat["X_obs"], flat["obs_weight"]
    beta = state["current_beta"]
    assignments = state["assignments"]
    s, kappa, sK = state["current_s"], state["current_kappa"], state["current_scalar_K"]
    cluster_f = state["cluster_f"]
    t_obs = dt + beta[ids]

    globals_fn = {
        "lbfgs_approx": optimizer_theta_globals.theta_s_loss_jac_approx_multi,
        "lbfgs_exact": optimizer_theta_globals.theta_s_loss_jac_exact_multi,
    }[method]
    cluster_fn = {
        "lbfgs_approx": optimizer_theta_cluster.theta_cluster_loss_jac,
        "lbfgs_exact": optimizer_theta_cluster.theta_cluster_loss_jac_exact,
    }[method]
    params = np.concatenate([s, kappa, [sK]])

    def globals_eval():
        return globals_fn(params, t_obs, X_obs, K_, t_span, cluster_f, assignments[ids],
                          0.0, em.lambda_scalar, em.lambda_kappa, em.scalar_K_center,
                          em.fit_s, em.ode_method, w, None)

    def f_eval():
        for z in range(em.n_subtypes):
            rows = np.isin(ids, np.where(assignments == z)[0])
            if np.any(rows):
                cluster_fn(np.ravel(cluster_f[z]), t_obs[rows], X_obs[rows], K_, t_span, s,
                           sK, em.lambda_f, kappa, em.ode_method, w[rows])

    K_ = em.K
    X_pred = {}
    dxdt = {}
    K_eff = sK * K_ + np.diag(kappa)
    for z in range(em.n_subtypes):
        f = np.ravel(cluster_f[z])
        X_pred[z] = solve_system(np.zeros(len(f)), f, K_, t_span, sK, kappa, em.ode_method)
        dxdt[z] = (1.0 - X_pred[z]) * (K_eff @ X_pred[z] + f[:, None])

    def beta_eval():
        return _vectorized_beta_loss_and_grad(
            beta, X_obs, dt, ids, flat["cog"], t_span, cluster_f, sK, s, assignments, K_,
            state["cluster_cog_a"], state["cluster_cog_b"], em.lambda_cog, em.lambda_jsd,
            em.t_max, X_pred, dxdt, lambda_beta=em.lambda_beta, beta_mean=state["beta_mean"],
            beta_var=state["beta_var"], obs_weight=w,
        )

    return {
        "eval_globals_ms": 1e3 * _mean_time(globals_eval, n_repeat),
        "eval_f_ms": 1e3 * _mean_time(f_eval, n_repeat),
        "eval_beta_ms": 1e3 * _mean_time(beta_eval, n_repeat),
    }


def _m_steps(em, flat, state) -> dict:
    """Run each M-step once from the shared initial state; time and count evaluations."""
    out = {}
    st = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in state.items()}
    st["cluster_f"] = [np.ravel(f).copy() for f in state["cluster_f"]]
    st["cluster_cog_a"] = [np.copy(a) for a in state["cluster_cog_a"]]
    st["cluster_cog_b"] = list(state["cluster_cog_b"])
    kw = dict(X_obs=flat["X_obs"], dt_obs=flat["dt"], ids=flat["ids"], K=em.K,
              t_span=em.t_span, obs_weight=flat["obs_weight"])

    if em.backend == "jax":
        from COMIND_transformer import jax_core as jc
        from COMIND_transformer import optimizer_jax as oj

        packed = jc.pack_data(
            flat["X_obs"], flat["obs_weight"], flat["dt"], flat["ids"], flat["cog"],
            flat["n_patients"], pad_rows=em.jax_pad_rows, pad_patients=em.jax_pad_patients,
        )
        info: Dict[str, object] = {}
        t0 = time.perf_counter()
        s, kappa, sK, _ = oj.fit_theta_globals_jax(
            **kw, cluster_f=st["cluster_f"], assignments=st["assignments"],
            beta_pred=st["current_beta"], s_guess=st["current_s"],
            kappa_guess=st["current_kappa"], scalar_K_guess=st["current_scalar_K"],
            lambda_scalar=em.lambda_scalar, lambda_kappa=em.lambda_kappa,
            scalar_K_center=em.scalar_K_center, fit_s=em.fit_s, s_mask=em.s_mask_,
            strict_tol=em.strict_tol, packed=packed, info=info,
        )
        out.update(_step_row("globals", time.perf_counter() - t0, info))
        info = {}
        t0 = time.perf_counter()
        F = oj.fit_theta_cluster_all_jax(
            flat["X_obs"], flat["dt"], flat["ids"], em.K, em.t_span,
            cluster_f=st["cluster_f"], assignments=st["assignments"], s=s, scalar_K=sK,
            lambda_f=em.lambda_f, beta_pred=st["current_beta"], kappa=kappa,
            strict_tol=em.strict_tol, obs_weight=flat["obs_weight"], packed=packed, info=info,
        )
        out.update(_step_row("f", time.perf_counter() - t0, info))
        info = {}
        t0 = time.perf_counter()
        traj = oj.solve_trajectories_jax(F, kappa, sK, em.K, em.t_span)
        oj.estimate_beta_jax(
            st["current_beta"], flat["X_obs"], flat["dt"], flat["ids"], flat["cog"],
            em.t_span, traj, s, st["assignments"], st["cluster_cog_a"], st["cluster_cog_b"],
            lambda_cog=em.lambda_cog, lambda_jsd=em.lambda_jsd, lambda_beta=em.lambda_beta,
            beta_mean=st["beta_mean"], beta_var=st["beta_var"], t_max=em.t_max,
            strict_tol=em.strict_tol, obs_weight=flat["obs_weight"], packed=packed, info=info,
        )
        out.update(_step_row("beta", time.perf_counter() - t0, info))
        return out

    method = em.theta_solver_stages[0]
    with _capture_minimize([optimizer_theta_globals]) as res:
        t0 = time.perf_counter()
        s, kappa, sK, _ = optimizer_theta_globals.fit_theta_globals(
            **kw, cluster_f=st["cluster_f"], assignments=st["assignments"],
            beta_pred=st["current_beta"], s_guess=st["current_s"],
            kappa_guess=st["current_kappa"], scalar_K_guess=st["current_scalar_K"],
            lambda_scalar=em.lambda_scalar, lambda_kappa=em.lambda_kappa,
            scalar_K_center=em.scalar_K_center, fit_s=em.fit_s, s_mask=em.s_mask_,
            method=method, strict_tol=em.strict_tol, ode_method=em.ode_method,
        )
        out.update(_step_row("globals", time.perf_counter() - t0, _summarize(res)))
    with _capture_minimize([optimizer_theta_cluster]) as res:
        t0 = time.perf_counter()
        F = []
        ids = flat["ids"]
        for z in range(em.n_subtypes):
            cl_idx = np.where(st["assignments"] == z)[0]
            if len(cl_idx) == 0:
                F.append(st["cluster_f"][z])
                continue
            rows = np.isin(ids, cl_idx)
            F.append(optimizer_theta_cluster.fit_theta_cluster(
                X_obs=flat["X_obs"][rows], dt_obs=flat["dt"][rows],
                ids=SubtypingEM._reindex_ids(ids[rows]), K=em.K, t_span=em.t_span, s=s,
                scalar_K=sK, lambda_f=em.lambda_f, method=method,
                beta_pred=st["current_beta"][cl_idx], f_guess=st["cluster_f"][z],
                kappa=kappa, strict_tol=em.strict_tol, ode_method=em.ode_method,
                obs_weight=flat["obs_weight"][rows],
            ))
        out.update(_step_row("f", time.perf_counter() - t0, _summarize(res)))
    with _capture_minimize([optimizer_beta]) as res:
        t0 = time.perf_counter()
        optimizer_beta.estimate_beta(
            st["current_beta"], flat["X_obs"], flat["dt"], flat["ids"], flat["cog"],
            em.t_span, F, sK, s, st["assignments"], em.K, st["cluster_cog_a"],
            st["cluster_cog_b"], lambda_cog=em.lambda_cog, lambda_jsd=em.lambda_jsd,
            lambda_beta=em.lambda_beta, beta_mean=st["beta_mean"], beta_var=st["beta_var"],
            t_max=em.t_max, kappa=kappa, strict_tol=em.strict_tol, ode_method=em.ode_method,
            obs_weight=flat["obs_weight"],
        )
        out.update(_step_row("beta", time.perf_counter() - t0, _summarize(res)))
    return out


def _step_row(name: str, wall: float, info: dict) -> dict:
    return {
        f"{name}_step_s": wall,
        f"{name}_nfev": info.get("nfev"),
        f"{name}_nit": info.get("nit"),
        f"{name}_message": info.get("message"),
    }


def run_benchmark(
    X: List[dict],
    K: np.ndarray,
    n_subtypes: int,
    t_max: float,
    step: float,
    seed: int = 0,
    backends: Optional[list] = None,
    verbose: bool = True,
    **em_kwargs,
) -> pd.DataFrame:
    """
    Benchmark NumPy ``lbfgs_approx``, NumPy ``lbfgs_exact`` and the JAX backend.

    Parameters
    ----------
    X : list of dict
        Patient list as accepted by ``SubtypingEM.fit``.
    K : np.ndarray
        Connectivity, shape (B, B).
    n_subtypes : int
    t_max, step : float
        Time grid, as in ``SubtypingEM``.
    seed : int
        Seed for each fit's ``rng`` (identical across backends).
    backends : list, optional
        Subset of ``BACKENDS`` entries ``(label, backend, stages)``.
    verbose : bool
        Print the table.
    **em_kwargs
        Extra ``SubtypingEM`` arguments (lambdas, ``max_iter``, initial values,
        ``jax_pad_rows`` ...). ``rng`` is ignored in favor of ``seed``.

    Returns
    -------
    pandas.DataFrame
        One row per backend.
    """
    from sklearn.metrics import adjusted_rand_score

    em_kwargs.pop("rng", None)
    has_true = all("subtype_true" in p for p in X)
    true = np.array([p["subtype_true"] for p in X]) if has_true else None
    rows = []
    for label, backend, stages in backends or BACKENDS:
        row = {"backend": label}
        em = _make_em(K, n_subtypes, t_max, step, backend, stages, seed, em_kwargs)
        flat, state = _initial_state(em, X)

        if backend == "jax":
            row.update(_jax_compile_times(em, flat, state))
        else:
            row.update(_numpy_eval_times(em, flat, state, stages[0]))
        try:
            row.update(_m_steps(em, flat, state))
        except Exception as exc:  # e.g. LSODA sensitivity failures on stiff params
            row["m_step_error"] = repr(exc)

        em = _make_em(K, n_subtypes, t_max, step, backend, stages, seed, em_kwargs)
        t0 = time.perf_counter()
        try:
            em.fit(X)
        except Exception as exc:
            row["fit_error"] = repr(exc)
            rows.append(row)
            continue
        row["fit_s"] = time.perf_counter() - t0
        row["outer_iters"] = len(em.lse_history) - 1
        row["lse_final"] = float(em.lse_final)
        row["bic"] = float(em.bic_)
        row["lse_lsoda"] = float(
            em._compute_val_score(X, em.beta_history[:, -1], em.final_assignments)
        )
        if has_true:
            row["ari"] = float(adjusted_rand_score(true, em.final_assignments))
        if backend == "jax":
            row["nonfinite_events"] = em.jax_nonfinite_events_
        rows.append(row)

    df = pd.DataFrame(rows).set_index("backend")
    if verbose:
        with pd.option_context("display.max_columns", None, "display.width", 200,
                               "display.max_colwidth", 60):
            print(df.T)
    return df


def make_synthetic(
    n_patients: int = 100,
    n_biomarkers: int = 8,
    t_max: float = 12.0,
    step: float = 0.1,
    noise: float = 0.02,
    seed: int = 0,
):
    """
    Synthetic two-subtype cohort with informed initial betas.

    Parameters
    ----------
    n_patients, n_biomarkers : int
    t_max, step : float
    noise : float
        Observation noise standard deviation.
    seed : int

    Returns
    -------
    X : list of dict
        Patients with ``subtype_true``.
    K : np.ndarray
        Row-normalized connectivity with zero diagonal.
    initial_beta : np.ndarray
        True beta + N(0, 1), clipped to ``[0, t_max]``.
    """
    rng = np.random.default_rng(seed)
    K = rng.random((n_biomarkers, n_biomarkers))
    np.fill_diagonal(K, 0.0)
    K /= K.sum(axis=1, keepdims=True)
    t_span = np.linspace(0.0, t_max, int(t_max / step))
    F_true = np.zeros((2, n_biomarkers))
    F_true[0, :2] = [0.4, 0.2]
    F_true[1, -2:] = [0.2, 0.4]
    kappa = np.full(n_biomarkers, 0.1)
    trajs = [solve_system(np.zeros(n_biomarkers), f, K, t_span, 0.3, kappa) for f in F_true]
    X, beta_true = [], []
    for i in range(n_patients):
        z = i % 2
        beta = rng.uniform(1.0, t_max - 5.0)
        n_visits = int(rng.integers(2, 5))
        dt = np.arange(n_visits, dtype=float)
        x_obs = np.array(
            [[np.interp(beta + d, t_span, trajs[z][b]) for b in range(n_biomarkers)] for d in dt]
        ) + rng.normal(0.0, noise, size=(n_visits, n_biomarkers))
        X.append(dict(id=i, X_obs=x_obs, dt=dt, cog=np.zeros((n_visits, 1)), subtype_true=z))
        beta_true.append(beta)
    initial_beta = np.clip(np.array(beta_true) + rng.normal(0.0, 1.0, n_patients), 0.0, t_max)
    return X, K, initial_beta


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n-patients", type=int, default=100)
    parser.add_argument("--n-biomarkers", type=int, default=8)
    parser.add_argument("--n-subtypes", type=int, default=2)
    parser.add_argument("--t-max", type=float, default=12.0)
    parser.add_argument("--step", type=float, default=0.1)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-exact", action="store_true",
                        help="Skip the (slow) NumPy lbfgs_exact backend.")
    parser.add_argument("--csv", type=str, default=None, help="Write the table to this CSV.")
    args = parser.parse_args()

    X, K, initial_beta = make_synthetic(
        args.n_patients, args.n_biomarkers, args.t_max, args.step, seed=args.seed
    )
    backends = [b for b in BACKENDS if not (args.skip_exact and b[0] == "numpy:lbfgs_exact")]
    df = run_benchmark(
        X, K, args.n_subtypes, args.t_max, args.step, seed=args.seed, backends=backends,
        max_iter=args.max_iter, lambda_kappa=0.2, lambda_scalar=0.2,
        initial_beta=initial_beta,
    )
    if args.csv:
        df.to_csv(args.csv)


if __name__ == "__main__":
    main()
