"""Tests for the JAX ``transform`` / ``transform_soft`` (``transform_backend="jax"``)."""

import time

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from test_jax_core import STEP, T_MAX, lsoda_trajectories, make_K  # noqa: E402

from COMIND_transformer import jax_core as jc  # noqa: E402
from COMIND_transformer.subtyping_em_transformer import SubtypingEM  # noqa: E402


def fitted_model(seed: int = 0, n_subtypes: int = 3, n_biomarkers: int = 6,
                 n_patients: int = 80, cap: bool = False, noise: float = 0.02):
    """A SubtypingEM with known "fitted" parameters, plus patients drawn from it."""
    rng = np.random.default_rng(seed)
    K = make_K(n_biomarkers, seed)
    t_span = np.linspace(0.0, T_MAX, int(T_MAX / STEP))
    F = np.zeros((n_subtypes, n_biomarkers))
    for z in range(n_subtypes):
        F[z, 2 * z % n_biomarkers] = 0.5
    kappa = np.full(n_biomarkers, 0.1)
    scalar_K = 0.3
    s = rng.uniform(0.8, 1.2, n_biomarkers)
    cog_a = [np.array([2.0]) for _ in range(n_subtypes)]
    cog_b = [1.0 for _ in range(n_subtypes)]

    em = SubtypingEM(K=K, n_subtypes=n_subtypes, t_max=T_MAX, step=STEP, lambda_cog=0.05,
                     cap_beta_by_followup=cap, verbose=0)
    em.t_span = t_span
    em.cluster_f = [f.copy() for f in F]
    em.final_s = s
    em.final_kappa = kappa
    em.final_scalar_K = scalar_K
    em.cluster_cog_a = cog_a
    em.cluster_cog_b = cog_b

    trajs = lsoda_trajectories(F, K, t_span, scalar_K, kappa)
    X = []
    for i in range(n_patients):
        z = int(rng.integers(n_subtypes))
        n_visits = int(rng.integers(2, 5))
        dt = np.arange(n_visits, dtype=float)
        beta = rng.uniform(0.5, T_MAX - n_visits)
        t = beta + dt
        x = np.array([[s[b] * np.interp(tt, t_span, trajs[z][b]) for b in range(n_biomarkers)]
                      for tt in t]) + rng.normal(0.0, noise, (n_visits, n_biomarkers))
        if i % 5 == 0:
            x[-1, 0] = np.nan
        cog = ((t - cog_b[z]) / cog_a[z][0] + rng.normal(0.0, 0.3, n_visits)).reshape(-1, 1)
        X.append(dict(id=f"p{i}", X_obs=x, dt=dt, cog=cog, subtype_true=z, beta_true=beta))
    return em, X, trajs


@pytest.mark.parametrize("cap", [False, True])
@pytest.mark.parametrize("use_cog", [False, True])
def test_identical_trajectories_agree(cap, use_cog):
    em, X, trajs = fitted_model(seed=50, cap=cap)
    t0 = time.perf_counter()
    ref = em._transform_numpy(X, use_cog, _trajectories=trajs)
    t_np = time.perf_counter() - t0
    t0 = time.perf_counter()
    out = em._transform_jax(X, use_cog, _trajectories=trajs)
    t_jax = time.perf_counter() - t0

    agree = np.mean(ref["subtype"] == out["subtype"])
    dbeta = np.abs(ref["beta"] - out["beta"])
    print(f"\n[same traj, cap={cap}, cog={use_cog}] subtype agreement={agree:.3f}, "
          f"median|dbeta|={np.median(dbeta):.2e}, max|dbeta|={dbeta.max():.2e}, "
          f"numpy {t_np:.2f}s vs jax {t_jax:.2f}s")
    assert agree >= 0.98
    assert np.median(dbeta) < 0.02 * T_MAX
    if cap:
        caps = np.array([T_MAX - p["dt"].max() for p in X])
        assert np.all(out["beta"] <= caps + 1e-12)
    np.testing.assert_array_equal(em.beta_val, out["beta"])
    np.testing.assert_array_equal(em.transform_assignments, out["subtype"])


def test_own_solvers_agree(capsys):
    em, X, _ = fitted_model(seed=51)
    ref = em.transform(X, use_cognitive_prior=False)
    em.transform_backend = "jax"
    out = em.transform(X, use_cognitive_prior=False)
    agree = np.mean(ref["subtype"] == out["subtype"])
    dbeta = np.abs(ref["beta"] - out["beta"])
    with capsys.disabled():
        print(f"\n[own solvers] subtype agreement={agree:.3f}, "
              f"median|dbeta|={np.median(dbeta):.2e}, max|dbeta|={dbeta.max():.2e}")
    assert agree >= 0.9


@pytest.mark.parametrize("per_timepoint", [False, True])
def test_transform_soft_matches_numpy_layout(per_timepoint):
    em, X, trajs = fitted_model(seed=52, n_patients=40)
    ref = em._transform_soft_numpy(X, True, 0.5, per_timepoint, _trajectories=trajs)
    probs_np = em.assignment_probabilities_
    out = em._transform_soft_jax(X, True, 0.5, per_timepoint, _trajectories=trajs)

    assert out.dtype == ref.dtype
    assert len(out) == len(ref)
    if per_timepoint:
        assert len(out) == sum(len(p["dt"]) for p in X)
        for f in ("patient_idx", "visit", "id"):
            np.testing.assert_array_equal(out[f], ref[f])
    proba = np.column_stack([out[f"proba_{z + 1}"] for z in range(em.n_subtypes)])
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(em.assignment_probabilities_, proba)
    assert em.assignment_probabilities_.shape == probs_np.shape

    assert np.mean(out["subtype"] == ref["subtype"]) >= 0.95
    assert np.mean(out["subtype_best"] == ref["subtype_best"]) >= 0.9
    for z in range(em.n_subtypes):
        assert np.median(np.abs(out[f"beta_{z + 1}"] - ref[f"beta_{z + 1}"])) < 0.02 * T_MAX
    if per_timepoint:
        np.testing.assert_allclose(out["t_obs"] - out["beta"],
                                   np.concatenate([p["dt"] for p in X]), atol=1e-12)

    hard = em._transform_jax(X, True, _trajectories=trajs)
    if not per_timepoint:
        np.testing.assert_array_equal(out["beta"], hard["beta"])
        np.testing.assert_array_equal(out["subtype"], hard["subtype"])


def test_no_recompiles_across_transform_calls():
    em, X, _ = fitted_model(seed=53, n_patients=60)
    em.transform_backend = "jax"
    em.transform_pad_rows = 256
    em.transform_pad_patients = 64

    def sizes():
        return {name: fn._cache_size() for name, fn in jc.JITTED_FUNCTIONS.items()}

    em.transform(X[:30])
    em.transform_soft(X[:30], per_timepoint=True)
    first = sizes()
    em.transform(X[30:], use_cognitive_prior=False)
    em.transform_soft(X[30:], use_cognitive_prior=False, temperature=2.0, per_timepoint=True)
    assert sizes() == first


def test_invalid_transform_backend():
    with pytest.raises(ValueError):
        SubtypingEM(transform_backend="torch")
