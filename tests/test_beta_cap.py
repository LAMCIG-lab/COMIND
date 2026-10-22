"""Tests for the per-patient beta cap (``cap_beta_by_followup``)."""

import numpy as np
import pytest

pytest.importorskip("jax")

from test_jax_core import T_MAX, STEP, grid, lsoda_trajectories, make_patients, make_problem  # noqa: E402

from COMIND_transformer import optimizer_jax as oj  # noqa: E402
from COMIND_transformer import subtyping_em_transformer as sem  # noqa: E402
from COMIND_transformer.optimizer_beta import estimate_beta  # noqa: E402
from COMIND_transformer.subtyping_em_transformer import SubtypingEM  # noqa: E402


def _late_patients(seed: int = 0, n_patients: int = 40):
    """Patients plus informed initial betas, with some initial betas above the cap."""
    X, K, initial_beta = make_patients(seed=seed, n_patients=n_patients)
    caps = np.array([T_MAX - np.max(p["dt"]) for p in X])
    initial_beta = initial_beta.copy()
    initial_beta[:5] = caps[:5] + 1.0
    return X, K, initial_beta, caps


def test_cap_equals_followup():
    X, K, _ = make_patients(seed=1, n_patients=20)
    expected = np.array([T_MAX - np.max(p["dt"]) for p in X])

    flat = SubtypingEM(K=K, t_max=T_MAX, step=STEP, cap_beta_by_followup=True,
                       verbose=0)._prepare_data(X)
    np.testing.assert_array_equal(flat["beta_upper"], expected)
    assert np.all(flat["initial_beta"] >= 0) and np.all(flat["initial_beta"] <= expected)

    flat_off = SubtypingEM(K=K, t_max=T_MAX, step=STEP, verbose=0)._prepare_data(X)
    np.testing.assert_array_equal(flat_off["beta_upper"], np.full(len(X), T_MAX))


def test_cap_rejects_followup_beyond_t_max():
    X, K, _ = make_patients(seed=2, n_patients=10)
    X[3] = dict(X[3], dt=np.array([0.0, T_MAX]), X_obs=X[3]["X_obs"][:2],
                cog=X[3]["cog"][:2])
    em = SubtypingEM(K=K, t_max=T_MAX, step=STEP, cap_beta_by_followup=True, verbose=0)
    with pytest.raises(ValueError, match=r"\[3\]"):
        em._prepare_data(X)
    # Without the cap the same data is accepted.
    SubtypingEM(K=K, t_max=T_MAX, step=STEP, verbose=0)._prepare_data(X)


def test_initial_beta_clipped_with_warning():
    X, K, initial_beta, caps = _late_patients(seed=3)
    em = SubtypingEM(K=K, t_max=T_MAX, step=STEP, cap_beta_by_followup=True,
                     initial_beta=initial_beta, verbose=0)
    with pytest.warns(RuntimeWarning, match="clipped the initial beta of 5"):
        flat = em._prepare_data(X)
    assert np.all(flat["initial_beta"] <= caps)


def _beta_problem(seed: int = 5):
    prob = make_problem(seed=seed)
    t_span, _, _ = grid()
    X_preds = lsoda_trajectories(prob["F"], prob["K"], t_span, prob["scalar_K"], prob["kappa"])
    return prob, t_span, X_preds


def test_estimate_beta_none_equals_full_t_max_numpy():
    prob, t_span, _ = _beta_problem()
    n = prob["n_patients"]
    common = dict(
        beta_all=prob["beta"], X_obs=prob["X_obs"], dt=prob["dt"], ids=prob["ids"],
        cog=prob["cog"], t_span=t_span, cluster_f=list(prob["F"]),
        scalar_K=prob["scalar_K"], s=prob["s"], assignments=prob["assignments"],
        K=prob["K"], cog_a=list(prob["cog_a"]), cog_b=list(prob["cog_b"]),
        lambda_cog=0.1, lambda_jsd=0.5, t_max=T_MAX, kappa=prob["kappa"],
        obs_weight=prob["obs_weight"],
    )
    b0, l0 = estimate_beta(**common)
    b1, l1 = estimate_beta(**common, beta_upper=np.full(n, T_MAX))
    np.testing.assert_array_equal(b0, b1)
    assert l0 == l1


def test_estimate_beta_none_equals_full_t_max_jax():
    prob, t_span, X_preds = _beta_problem()
    n = prob["n_patients"]
    common = dict(
        beta_all=prob["beta"], X_obs=prob["X_obs"], dt=prob["dt"], ids=prob["ids"],
        cog=prob["cog"], t_span=t_span, trajectories=X_preds, s=prob["s"],
        assignments=prob["assignments"], cog_a=list(prob["cog_a"]),
        cog_b=list(prob["cog_b"]), lambda_cog=0.1, lambda_jsd=0.5, t_max=T_MAX,
        obs_weight=prob["obs_weight"],
    )
    b0, l0 = oj.estimate_beta_jax(**common)
    b1, l1 = oj.estimate_beta_jax(**common, beta_upper=np.full(n, T_MAX))
    np.testing.assert_array_equal(b0, b1)
    assert l0 == l1


def test_estimate_beta_respects_per_patient_upper():
    prob, t_span, X_preds = _beta_problem(seed=6)
    upper = np.full(prob["n_patients"], 2.0)
    b_np, _ = estimate_beta(
        prob["beta"].clip(0, 2.0), prob["X_obs"], prob["dt"], prob["ids"], prob["cog"],
        t_span, list(prob["F"]), prob["scalar_K"], prob["s"], prob["assignments"],
        prob["K"], list(prob["cog_a"]), list(prob["cog_b"]), t_max=T_MAX,
        kappa=prob["kappa"], obs_weight=prob["obs_weight"], beta_upper=upper,
    )
    b_jax, _ = oj.estimate_beta_jax(
        prob["beta"].clip(0, 2.0), prob["X_obs"], prob["dt"], prob["ids"], prob["cog"],
        t_span, X_preds, prob["s"], prob["assignments"], list(prob["cog_a"]),
        list(prob["cog_b"]), t_max=T_MAX, obs_weight=prob["obs_weight"], beta_upper=upper,
    )
    assert np.all(b_np <= 2.0) and np.all(b_jax <= 2.0)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_fit_respects_cap(backend):
    X, K, initial_beta, caps = _late_patients(seed=7)
    em = SubtypingEM(K=K, n_subtypes=2, t_max=T_MAX, step=STEP, max_iter=4, verbose=0,
                     backend=backend, theta_solver_stages=("lbfgs_approx",),
                     lambda_kappa=0.2, lambda_scalar=0.2, initial_beta=initial_beta,
                     cap_beta_by_followup=True, rng=np.random.default_rng(1))
    with pytest.warns(RuntimeWarning, match="clipped"):
        em.fit(X)
    np.testing.assert_array_equal(em.beta_upper_, caps)
    assert np.all(em.beta_history >= 0)
    assert np.all(em.beta_history <= caps[:, None] + 1e-12)

    out = em.transform(X[:10], use_cognitive_prior=False)
    assert np.all(out["beta"] >= 0)
    assert np.all(out["beta"] <= caps[:10] + 1e-12)
    soft = em.transform_soft(X[:10], use_cognitive_prior=False)
    for z in range(em.n_subtypes):
        assert np.all(soft[f"beta_{z + 1}"] <= caps[:10] + 1e-12)


def test_transform_rejects_grid_above_cap():
    X, K, _ = make_patients(seed=8, n_patients=20)
    em = SubtypingEM(K=K, n_subtypes=2, t_max=T_MAX, step=STEP, max_iter=1, verbose=0,
                     theta_solver_stages=("lbfgs_approx",), cap_beta_by_followup=True)
    em.fit(X)
    new = dict(X[0], dt=np.array([0.0, 0.95 * T_MAX]), X_obs=X[0]["X_obs"][:2],
               cog=X[0]["cog"][:2])
    with pytest.raises(ValueError, match="every beta grid point"):
        em.transform([X[1], new])


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_flag_off_matches_hardwired_none(backend, monkeypatch):
    X, K, initial_beta = make_patients(seed=9, n_patients=40)
    common = dict(K=K, n_subtypes=2, t_max=T_MAX, step=STEP, max_iter=4, verbose=0,
                  backend=backend, theta_solver_stages=("lbfgs_approx",),
                  lambda_kappa=0.2, lambda_scalar=0.2, lambda_jsd=0.3,
                  initial_beta=initial_beta)

    em_a = SubtypingEM(**common, rng=np.random.default_rng(4)).fit(X)

    orig_np, orig_jax = sem.estimate_beta, oj.estimate_beta_jax

    def np_none(*args, **kwargs):
        kwargs["beta_upper"] = None
        return orig_np(*args, **kwargs)

    def jax_none(*args, **kwargs):
        kwargs["beta_upper"] = None
        return orig_jax(*args, **kwargs)

    monkeypatch.setattr(sem, "estimate_beta", np_none)
    monkeypatch.setattr(oj, "estimate_beta_jax", jax_none)
    em_b = SubtypingEM(**common, rng=np.random.default_rng(4)).fit(X)

    assert em_a.lse_final == em_b.lse_final
    np.testing.assert_array_equal(em_a.final_assignments, em_b.final_assignments)
    np.testing.assert_array_equal(em_a.beta_history, em_b.beta_history)
