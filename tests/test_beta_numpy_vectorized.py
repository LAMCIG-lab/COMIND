"""Vectorized NumPy beta step and ``KernelJSDMulti`` are bit-identical to the original loops."""

import time

import numpy as np
import pytest

from _reference_beta import (
    _ReferenceKernelJSDMulti,
    _reference_estimate_beta,
    _reference_vectorized_beta_loss_and_grad,
)
from test_jax_core import STEP, T_MAX, grid, lsoda_trajectories, make_patients, make_problem

from COMIND_transformer import optimizer_beta as ob
from COMIND_transformer import subtyping_em_transformer as sem
from COMIND_transformer.kernel_jsd_multi import KernelJSDMulti
from COMIND_transformer.subtyping_em_transformer import SubtypingEM


def _beta_args(seed, nan_frac=0.1, empty_subtype=False, lambda_cog=0.0, lambda_jsd=0.0,
               n_patients=60):
    prob = make_problem(seed=seed, nan_frac=nan_frac, n_patients=n_patients)
    if empty_subtype:
        prob["assignments"][prob["assignments"] == 1] = 2
    t_span, _, _ = grid()
    Xp = lsoda_trajectories(prob["F"], prob["K"], t_span, prob["scalar_K"], prob["kappa"])
    K_eff = prob["scalar_K"] * prob["K"] + np.diag(prob["kappa"])
    Z = prob["n_subtypes"]
    dxdt = {z: (1.0 - Xp[z]) * (K_eff @ Xp[z] + prob["F"][z][:, None]) for z in range(Z)}
    args = (
        prob["beta"], prob["X_obs"], prob["dt"], prob["ids"], prob["cog"], t_span,
        list(prob["F"]), prob["scalar_K"], prob["s"], prob["assignments"], prob["K"],
        list(prob["cog_a"]), list(prob["cog_b"]), lambda_cog, lambda_jsd, T_MAX,
        {z: Xp[z] for z in range(Z)}, dxdt,
    )
    return prob, t_span, Xp, args


@pytest.mark.parametrize("lambda_jsd", [0.0, 2.0])
@pytest.mark.parametrize("lambda_cog", [0.0, 0.3])
@pytest.mark.parametrize("lambda_beta", [0.0, 0.5])
@pytest.mark.parametrize("nan_frac,empty", [(0.0, False), (0.15, False), (0.15, True)])
def test_loss_and_grad_match_reference(lambda_jsd, lambda_cog, lambda_beta, nan_frac, empty):
    for seed in (60, 61):
        prob, _, _, args = _beta_args(seed, nan_frac, empty, lambda_cog, lambda_jsd)
        kw = dict(lambda_beta=lambda_beta, beta_mean=4.0, beta_var=3.0,
                  obs_weight=prob["obs_weight"] if nan_frac > 0 else None)
        l_ref, g_ref = _reference_vectorized_beta_loss_and_grad(*args, **kw)
        l_new, g_new = ob._vectorized_beta_loss_and_grad(*args, **kw)
        assert l_new == l_ref
        np.testing.assert_array_equal(g_new, g_ref)


def test_unsorted_ids_fallback():
    prob, _, _, args = _beta_args(62, lambda_cog=0.2, lambda_jsd=1.0)
    perm = np.random.default_rng(62).permutation(len(prob["ids"]))
    args = list(args)
    for i in (1, 2, 3, 4):  # X_obs, dt, ids, cog
        args[i] = args[i][perm]
    w = prob["obs_weight"][perm]
    l_ref, g_ref = _reference_vectorized_beta_loss_and_grad(*args, obs_weight=w)
    l_new, g_new = ob._vectorized_beta_loss_and_grad(*args, obs_weight=w)
    assert l_new == l_ref
    np.testing.assert_array_equal(g_new, g_ref)


@pytest.mark.parametrize("bandwidth", [None, 0.8])
@pytest.mark.parametrize("sizes", [(30, 45), (12, 50, 33), (5, 1, 70, 20)])
def test_kernel_jsd_matches_reference(bandwidth, sizes):
    rng = np.random.default_rng(sum(sizes))
    dists = [rng.uniform(0.0, T_MAX, n) for n in sizes]
    ref = _ReferenceKernelJSDMulti(dists, value_range=(0, T_MAX), bandwidth=bandwidth)
    new = KernelJSDMulti(dists, value_range=(0, T_MAX), bandwidth=bandwidth)
    assert new.jsd() == ref.jsd()
    for g_new, g_ref in zip(new.jsd_derivatives(), ref.jsd_derivatives()):
        np.testing.assert_array_equal(g_new, g_ref)
    # Densities are cached and shared, so a second call returns the same numbers.
    assert new.jsd() == new.jsd()


def test_estimate_beta_matches_reference():
    prob, t_span, _, _ = _beta_args(63, lambda_cog=0.2, lambda_jsd=1.0)
    common = dict(
        beta_all=prob["beta"], X_obs=prob["X_obs"], dt=prob["dt"], ids=prob["ids"],
        cog=prob["cog"], t_span=t_span, cluster_f=list(prob["F"]),
        scalar_K=prob["scalar_K"], s=prob["s"], assignments=prob["assignments"],
        K=prob["K"], cog_a=list(prob["cog_a"]), cog_b=list(prob["cog_b"]),
        lambda_cog=0.2, lambda_jsd=1.0, lambda_beta=0.3, beta_mean=4.0, beta_var=3.0,
        t_max=T_MAX, kappa=prob["kappa"], obs_weight=prob["obs_weight"],
    )
    b_ref, l_ref = _reference_estimate_beta(**common)
    b_new, l_new = ob.estimate_beta(**common)
    np.testing.assert_array_equal(b_new, b_ref)
    assert l_new == l_ref


def test_full_fit_matches_reference(monkeypatch, capsys):
    X, K, initial_beta = make_patients(seed=64, n_patients=60)
    common = dict(K=K, n_subtypes=2, t_max=T_MAX, step=STEP, max_iter=6, verbose=0,
                  theta_solver_stages=("lbfgs_approx",), lambda_kappa=0.2, lambda_scalar=0.2,
                  lambda_jsd=0.3, lambda_cog=0.1, lambda_beta=0.1, initial_beta=initial_beta)

    t0 = time.perf_counter()
    em_new = SubtypingEM(**common, rng=np.random.default_rng(6)).fit(X)
    t_new = time.perf_counter() - t0

    monkeypatch.setattr(sem, "estimate_beta", _reference_estimate_beta)
    t0 = time.perf_counter()
    em_ref = SubtypingEM(**common, rng=np.random.default_rng(6)).fit(X)
    t_ref = time.perf_counter() - t0

    with capsys.disabled():
        print(f"\n[full fit] lse_final new={em_new.lse_final!r} ref={em_ref.lse_final!r} "
              f"(identical={em_new.lse_final == em_ref.lse_final}); "
              f"fit time new {t_new:.2f}s vs ref {t_ref:.2f}s")
    np.testing.assert_array_equal(em_new.final_assignments, em_ref.final_assignments)
    assert em_new.lse_final == em_ref.lse_final
    np.testing.assert_array_equal(em_new.beta_history, em_ref.beta_history)


def test_timing_report(capsys):
    prob, _, _, args = _beta_args(65, lambda_cog=0.2, lambda_jsd=1.0, n_patients=400)
    kw = dict(obs_weight=prob["obs_weight"])

    def bench(fn, reps=5):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn(*args, **kw)
        return (time.perf_counter() - t0) / reps

    t_ref = bench(_reference_vectorized_beta_loss_and_grad)
    t_new = bench(ob._vectorized_beta_loss_and_grad)

    rng = np.random.default_rng(0)
    dists = [rng.uniform(0, T_MAX, 150) for _ in range(3)]
    t0 = time.perf_counter()
    _ReferenceKernelJSDMulti(dists, (0, T_MAX)).jsd_derivatives()
    t_kref = time.perf_counter() - t0
    t0 = time.perf_counter()
    KernelJSDMulti(dists, (0, T_MAX)).jsd_derivatives()
    t_knew = time.perf_counter() - t0
    with capsys.disabled():
        print(f"\n[timing] beta loss+grad (400 patients): ref {t_ref * 1e3:.1f} ms, "
              f"new {t_new * 1e3:.1f} ms ({t_ref / t_new:.1f}x); "
              f"jsd_derivatives (3x150): ref {t_kref * 1e3:.1f} ms, new {t_knew * 1e3:.2f} ms")
