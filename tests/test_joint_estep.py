"""Tests for the joint (beta, subtype) grid E-step (``estep_mode="joint_grid"``)."""

import numpy as np
import pytest

pytest.importorskip("jax")

from test_jax_core import STEP, T_MAX, grid, lsoda_trajectories, make_patients, make_problem  # noqa: E402

from COMIND_transformer import assignments as asg  # noqa: E402
from COMIND_transformer import optimizer_jax as oj  # noqa: E402
from COMIND_transformer import subtyping_em_transformer as sem  # noqa: E402
from COMIND_transformer.assignments import sse_matrix, update_assignments_joint_grid  # noqa: E402
from COMIND_transformer.subtyping_em_transformer import SubtypingEM  # noqa: E402
from COMIND_transformer.utils import make_beta_grid  # noqa: E402

BACKENDS = ["numpy", "jax"]


def _estep(backend, prob, X_preds, t_span, beta, beta_grid, beta_upper=None, lambda_cog=0.0):
    args = (
        prob["X_obs"], prob["dt"], prob["ids"], prob["cog"], beta, list(prob["F"]),
        prob["scalar_K"], prob["kappa"], prob["s"], prob["K"], t_span,
        list(prob["cog_a"]), list(prob["cog_b"]), lambda_cog, beta_grid,
    )
    kw = dict(beta_upper=beta_upper, obs_weight=prob["obs_weight"], X_preds=X_preds)
    if backend == "numpy":
        return update_assignments_joint_grid(*args, **kw)
    return oj.update_assignments_joint_grid_jax(*args, **kw)


def _losses(prob, X_preds, t_span, beta, lambda_cog):
    return sse_matrix(
        prob["X_obs"], prob["dt"], prob["ids"], beta, X_preds, prob["s"], t_span,
        cog=prob["cog"], cluster_cog_a=list(prob["cog_a"]),
        cluster_cog_b=list(prob["cog_b"]), lambda_cog=lambda_cog,
        obs_weight=prob["obs_weight"],
    )


def _setup(seed):
    prob = make_problem(seed=seed)
    t_span, _, _ = grid()
    X_preds = lsoda_trajectories(prob["F"], prob["K"], t_span, prob["scalar_K"], prob["kappa"])
    return prob, t_span, X_preds


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("lambda_cog", [0.0, 0.4])
def test_per_patient_non_increase(backend, lambda_cog):
    for seed in (30, 31, 32):
        prob, t_span, X_preds = _setup(seed)
        rng = np.random.default_rng(seed)
        beta = prob["beta"]
        upper = beta + rng.uniform(0.0, 4.0, size=len(beta))
        z_cur = prob["assignments"]
        z_new, beta_start = _estep(backend, prob, X_preds, t_span, beta,
                                   make_beta_grid(T_MAX, 15), upper, lambda_cog)
        n = len(beta)
        cur = _losses(prob, X_preds, t_span, beta, lambda_cog)[np.arange(n), z_cur]
        new = _losses(prob, X_preds, t_span, beta_start, lambda_cog)[np.arange(n), z_new]
        assert np.all(new <= cur * (1 + 1e-12) + 1e-12)
        assert np.all(beta_start <= upper)


@pytest.mark.parametrize("backend", BACKENDS)
def test_ties_keep_current_state(backend):
    """With beta-independent (constant) trajectories every candidate ties."""
    prob, t_span, X_preds = _setup(33)
    rng = np.random.default_rng(33)
    const = [np.repeat(rng.uniform(0.1, 0.9, size=(x.shape[0], 1)), x.shape[1], axis=1)
             for x in X_preds]
    beta = prob["beta"]
    z_best = np.argmin(_losses(prob, const, t_span, beta, 0.0), axis=1)
    z_new, beta_start = _estep(backend, prob, const, t_span, beta, make_beta_grid(T_MAX, 10))
    np.testing.assert_array_equal(z_new, z_best)
    np.testing.assert_array_equal(beta_start, beta)


@pytest.mark.parametrize("backend", BACKENDS)
def test_masked_candidates_never_chosen(backend):
    prob, t_span, X_preds = _setup(34)
    rng = np.random.default_rng(34)
    # Observations generated at beta = 9, so the uncapped search prefers late betas.
    z_row = prob["assignments"][prob["ids"]]
    t_row = prob["dt"] + 9.0
    prob["X_obs"] = np.stack([
        prob["s"] * np.array([np.interp(t_row[r], t_span, X_preds[z_row[r]][b])
                              for b in range(len(prob["s"]))])
        for r in range(len(t_row))
    ])
    upper = rng.uniform(1.5, 4.0, size=prob["n_patients"])
    beta = np.minimum(prob["beta"], upper) * rng.uniform(0.5, 1.0, size=prob["n_patients"])
    beta_grid = make_beta_grid(T_MAX, 20)

    _, free = _estep(backend, prob, X_preds, t_span, beta, beta_grid)
    assert np.any(free > upper), "test needs some patients that prefer a beta above the cap"

    _, capped = _estep(backend, prob, X_preds, t_span, beta, beta_grid, upper)
    assert np.all(capped <= upper)
    on_grid_or_current = np.isclose(capped[:, None], beta_grid[None, :]).any(axis=1) | (
        capped == beta
    )
    assert np.all(on_grid_or_current)


@pytest.mark.parametrize("lambda_cog", [0.0, 0.3])
def test_numpy_and_jax_agree(lambda_cog):
    for seed in (35, 36):
        prob, t_span, X_preds = _setup(seed)
        rng = np.random.default_rng(seed)
        upper = prob["beta"] + rng.uniform(0.0, 5.0, size=prob["n_patients"])
        beta_grid = make_beta_grid(T_MAX, 25)
        z_np, b_np = _estep("numpy", prob, X_preds, t_span, prob["beta"], beta_grid,
                            upper, lambda_cog)
        z_jx, b_jx = _estep("jax", prob, X_preds, t_span, prob["beta"], beta_grid,
                            upper, lambda_cog)
        np.testing.assert_array_equal(z_np, z_jx)
        np.testing.assert_allclose(b_np, b_jx, rtol=0, atol=1e-10)


def test_joint_grid_losses_match_sse_matrix():
    prob, t_span, X_preds = _setup(37)
    rng = np.random.default_rng(37)
    cands = rng.uniform(-1.0, T_MAX + 2.0, size=(4, prob["n_patients"]))
    out = asg.joint_grid_losses(
        prob["X_obs"], prob["dt"], prob["ids"], cands, X_preds, prob["s"], t_span,
        cog=prob["cog"], cluster_cog_a=list(prob["cog_a"]),
        cluster_cog_b=list(prob["cog_b"]), lambda_cog=0.2, obs_weight=prob["obs_weight"],
    )
    for c in range(cands.shape[0]):
        ref = _losses(prob, X_preds, t_span, cands[c], 0.2)
        np.testing.assert_allclose(out[c], ref, rtol=1e-12, atol=1e-12)


def _fit(backend, X, K, initial_beta, **kw):
    params = dict(K=K, n_subtypes=2, t_max=T_MAX, step=STEP, max_iter=4, verbose=0,
                  backend=backend, theta_solver_stages=("lbfgs_approx",),
                  lambda_kappa=0.2, lambda_scalar=0.2, initial_beta=initial_beta,
                  rng=np.random.default_rng(2))
    params.update(kw)
    return SubtypingEM(**params).fit(X)


@pytest.mark.parametrize("backend", BACKENDS)
def test_current_mode_reproduces_default(backend, monkeypatch):
    X, K, initial_beta = make_patients(seed=40, n_patients=40)
    em_default = _fit(backend, X, K, initial_beta)

    def boom(*args, **kwargs):
        raise AssertionError("joint-grid E-step must not run in 'current' mode")

    monkeypatch.setattr(sem, "update_assignments_joint_grid", boom)
    monkeypatch.setattr(oj, "update_assignments_joint_grid_jax", boom)
    em_current = _fit(backend, X, K, initial_beta, estep_mode="current")

    assert em_default.lse_final == em_current.lse_final
    np.testing.assert_array_equal(em_default.final_assignments, em_current.final_assignments)
    np.testing.assert_array_equal(em_default.beta_history, em_current.beta_history)
    assert np.all(em_current.estep_beta_moves_history_ == 0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_joint_grid_fit_and_diagnostics(backend, tmp_path):
    X, K, initial_beta = make_patients(seed=41, n_patients=40)
    ck = str(tmp_path / "ck.npz")
    em = SubtypingEM(K=K, n_subtypes=2, t_max=T_MAX, step=STEP, max_iter=4, verbose=0,
                     backend=backend, theta_solver_stages=("lbfgs_approx",),
                     lambda_kappa=0.2, lambda_scalar=0.2, initial_beta=initial_beta,
                     estep_mode="joint_grid", estep_grid_size=12, cap_beta_by_followup=True,
                     rng=np.random.default_rng(3))
    em.fit(X, checkpoint_path=ck)
    n_hist = len(em.lse_history)
    assert em.subtype_counts_history_.shape == (2, n_hist)
    assert np.all(em.subtype_counts_history_.sum(axis=0) == len(X))
    assert em.beta_at_upper_fraction_history_.shape == (n_hist,)
    assert em.estep_beta_moves_history_.shape == (n_hist,)
    assert em.estep_beta_moves_history_[0] == 0
    assert em.estep_beta_moves_history_.sum() > 0
    assert np.all((em.beta_at_upper_fraction_history_ >= 0)
                  & (em.beta_at_upper_fraction_history_ <= 1))
    assert np.all(em.beta_history <= em.beta_upper_[:, None] + 1e-12)

    saved = np.load(ck, allow_pickle=True)
    for key in ("beta_upper", "subtype_counts_history", "beta_at_upper_fraction_history",
                "estep_beta_moves_history"):
        assert key in saved.files
    np.testing.assert_array_equal(saved["subtype_counts_history"], em.subtype_counts_history_)


def test_jitter_iterations_ignore_estep_mode(monkeypatch):
    X, K, initial_beta = make_patients(seed=42, n_patients=30)
    calls = []
    orig = sem.update_assignments_joint_grid

    def counting(*args, **kwargs):
        calls.append(1)
        return orig(*args, **kwargs)

    monkeypatch.setattr(sem, "update_assignments_joint_grid", counting)
    em = _fit("numpy", X, K, initial_beta, max_iter=3, estep_mode="joint_grid",
              assignments_jitter=True, jitter_iter=1)
    # Jitter runs on every loop_iter > 0, so only iteration 0 uses the joint grid.
    assert len(calls) == 1
    assert np.all(em.estep_beta_moves_history_[2:] == 0)


def test_invalid_estep_mode():
    with pytest.raises(ValueError):
        SubtypingEM(estep_mode="soft")
    with pytest.raises(ValueError):
        SubtypingEM(estep_grid_size=0)
