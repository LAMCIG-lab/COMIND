"""Tests for obs_model="zsym" (symmetric s-clipping) and biomarker_weights."""
import os
import sys

import jax
import numpy as np
import pytest

from COMIND_transformer import jax_core as jc
from COMIND_transformer.subtyping_em_transformer import SubtypingEM

sys.path.insert(0, os.path.dirname(__file__))
from make_golden_scale import OUT as GOLDEN_PATH  # noqa: E402
from make_golden_scale import golden_em, golden_outputs, golden_toy  # noqa: E402


def _zsym_toy(s_true, seed=0, n_pat=80, B=4, t_max=30.0, step=0.05, noise=0.05):
    """Two subtypes, x saturating at 1 inside t_max, observed as s (2x - 1) + noise."""
    rng = np.random.default_rng(seed)
    K = np.zeros((B, B))
    for b in range(B - 1):
        K[b, b + 1] = K[b + 1, b] = 1.0
    F = np.zeros((2, B)); F[0, 0] = 0.3; F[1, B - 1] = 0.3
    t_span = np.linspace(0, t_max, int(t_max / step))
    _, h, n = jc.grid_from_t_span(t_span)
    tr = np.asarray(jc._traj_jit(jc.f64(F), jc.f64(np.zeros(B)), jc.f64(0.5), jc.f64(K),
                                 jc.f64(h), n_steps=n))
    s_true = np.asarray(s_true, float)
    pats = []
    for i in range(n_pat):
        z = i % 2; b0 = rng.uniform(0, t_max - 3); dt = np.array([0.0, 1.0, 2.0])
        x = np.array([[np.interp(b0 + d, t_span, tr[z, :, b]) for b in range(B)] for d in dt])
        pats.append({"id": i, "X_obs": s_true * (2 * x - 1) + rng.normal(0, noise, x.shape),
                     "dt": dt, "cog": np.zeros((3, 1)), "initial_beta": b0})
    return dict(K=K, F=F, traj=tr, t_span=t_span, h=h, n_steps=n, pats=pats,
                t_max=t_max, step=step)


def _zsym_em(toy, **kw):
    B = toy["K"].shape[0]
    args = dict(K=toy["K"], initial_f=toy["F"] + 0.01, initial_scalar_K=0.4,
                initial_kappa=np.zeros(B), n_subtypes=2, max_iter=50, t_max=toy["t_max"],
                step=toy["step"], fit_s=True, s_lower_value=1.0, s_upper=3.0, verbose=0,
                backend="jax", obs_model="zsym",
                initial_assignments=np.arange(len(toy["pats"])) % 2,
                rng=np.random.default_rng(0))
    args.update(kw)
    return SubtypingEM(**args)


def _packed(pats, obs_weight=None):
    X = np.vstack([p["X_obs"] for p in pats])
    dt = np.concatenate([p["dt"] for p in pats])
    ids = np.concatenate([np.full(len(p["dt"]), i) for i, p in enumerate(pats)])
    data, meta = jc.pack_data(X, obs_weight, dt, ids, np.zeros((len(dt), 1)), len(pats))
    return X, dt, ids, data, meta


@pytest.fixture(scope="module")
def recovered():
    toy = _zsym_toy([1.5, 2.5, 1.5, 2.5])
    return toy, _zsym_em(toy).fit(toy["pats"])


# 1. Default behaviour unchanged ------------------------------------------------

def test_scale_matches_golden():
    g = np.load(GOLDEN_PATH)
    for k, v in golden_outputs().items():
        assert np.array_equal(np.asarray(v), g[k]), k


# 2. Loss identity -------------------------------------------------------------

def test_zsym_loss_equals_transformed_space_lse():
    toy = _zsym_toy([2.0, 2.5, 1.5, 2.0], noise=0.02)
    pats = toy["pats"]
    B = toy["K"].shape[0]
    s = np.array([2.0, 2.5, 1.5, 2.0]) + 0.5
    X, dt, ids, data, meta = _packed(pats)
    assert np.all(np.abs(X) < s)  # no clipping anywhere
    beta = np.array([p["initial_beta"] for p in pats])
    z = np.arange(len(pats)) % 2
    loss = float(jc.globals_loss_zsym(
        jc.f64(s), jc.f64(np.zeros(B)), jc.f64(0.5), jc.f64(toy["F"]),
        jc.pad_beta(beta, meta["P"]), jc.pad_assignments(z, meta["P"]), data,
        jc.f64(toy["K"]), jc.make_hyp(), jc.f64(toy["h"]), jc.f64(np.zeros(B)),
        n_steps=toy["n_steps"],
    ))
    t = beta[ids] + dt
    x = np.array([[np.interp(tt, toy["t_span"], toy["traj"][zz, :, b]) for b in range(B)]
                  for tt, zz in zip(t, z[ids])])
    z_t = (s + np.clip(X, -s, s)) / (2 * s)
    by_hand = float(np.sum((2 * s) ** 2 * (z_t - x) ** 2))
    direct = float(np.sum((X - s * (2 * x - 1)) ** 2))
    assert np.isclose(loss, by_hand, rtol=1e-10)
    assert np.isclose(loss, direct, rtol=1e-10)


# 3. Gradient check ------------------------------------------------------------

def test_zsym_grad_s_matches_finite_difference():
    toy = _zsym_toy([2.0, 2.5, 1.5, 2.0], n_pat=20)
    pats = toy["pats"]
    B = toy["K"].shape[0]
    _, _, _, data, meta = _packed(pats)
    beta = jc.pad_beta(np.array([p["initial_beta"] for p in pats]), meta["P"])
    z = jc.pad_assignments(np.arange(len(pats)) % 2, meta["P"])
    rest = (jc.f64(np.full(B, 0.05)), jc.f64(0.5), jc.f64(toy["F"] * 0.9), beta, z, data,
            jc.f64(toy["K"]), jc.make_hyp(), jc.f64(toy["h"]), jc.f64(np.zeros(B)))

    def f(s):
        return jc.globals_loss_zsym(s, *rest, n_steps=toy["n_steps"])

    s0 = np.array([1.7, 2.2, 1.3, 2.9])
    g = np.asarray(jax.grad(f)(jc.f64(s0)))
    eps = 1e-5
    fd = np.array([
        (float(f(jc.f64(s0 + eps * e))) - float(f(jc.f64(s0 - eps * e)))) / (2 * eps)
        for e in np.eye(B)
    ])
    assert np.allclose(g, fd, rtol=1e-6)


# 4-5. Recovery and bounds -------------------------------------------------------

def test_zsym_recovers_s(recovered):
    _, em = recovered
    assert np.all(np.abs(em.final_s_clip - np.array([1.5, 2.5, 1.5, 2.5])) < 0.2)


def test_zsym_bounds_and_tied_parameters(recovered):
    _, em = recovered
    assert np.all((em.final_s_clip >= 1.0) & (em.final_s_clip <= 3.0))
    assert np.array_equal(em.final_s, 2 * em.final_s_clip)
    assert np.array_equal(em.final_c, -em.final_s_clip)
    assert em._n_free_c(4) == 0


def test_zsym_s_clipped_at_upper_bound():
    toy = _zsym_toy([3.6, 3.6, 3.6, 3.6], n_pat=40)
    em = _zsym_em(toy, max_iter=5).fit(toy["pats"])
    assert np.all(em.final_s_clip <= 3.0 + 1e-12)
    assert np.all(em.final_s_clip > 2.9)


def test_zsym_default_start_is_two():
    toy = _zsym_toy([2.0] * 4, n_pat=20)
    em = _zsym_em(toy, max_iter=1).fit(toy["pats"])
    s0 = em.theta_history[4:8, 0]  # theta = [f_0 (B), current_s (B), scalar_K]
    assert np.allclose(s0, 2 * 2.0)
    assert np.allclose(em.c_history_[0], -2.0)


def test_zsym_validation():
    K = np.eye(3)
    with pytest.raises(ValueError):
        SubtypingEM(K=K, obs_model="zsym", backend="numpy")
    with pytest.raises(ValueError):
        SubtypingEM(K=K, obs_model="zsym", backend="jax", fit_intercept=True)
    with pytest.raises(ValueError):
        SubtypingEM(K=K, obs_model="bogus", backend="jax")
    with pytest.raises(ValueError):
        SubtypingEM(K=K, biomarker_weights=np.ones(3), backend="numpy")


# 6. Weights -----------------------------------------------------------------------

def test_unit_weights_reproduce_unweighted():
    K, pats, t_max, step = golden_toy()
    train, val = pats[:30], pats[30:]
    a = golden_em(K, t_max, step).fit(train)
    b = golden_em(K, t_max, step, biomarker_weights=np.ones(K.shape[0])).fit(train)
    for name in ("final_s", "final_kappa", "lse_history", "beta_history", "final_assignments"):
        assert np.array_equal(np.asarray(getattr(a, name)), np.asarray(getattr(b, name))), name
    assert a.bic_ == b.bic_
    assert a.score(val) == b.score(val)


def test_weighted_sse_is_sum_of_weighted_per_biomarker_sse(recovered):
    toy, em = recovered
    pats = toy["pats"]
    B = toy["K"].shape[0]
    w = np.array([4.0, 0.25, 1.0, 2.0])
    em_w = _zsym_em(toy, biomarker_weights=w)
    # Fixed parameters: the packed weighted mask turns recon_sse into sum_b w_b SSE_b.
    flat = em_w._prepare_data(pats)
    X, dt, ids, data, meta = _packed(pats, flat["obs_weight"])
    _, _, _, data_u, _ = _packed(pats)
    beta = jc.pad_beta(np.asarray(em.beta_history)[:, -1], meta["P"])
    z = jc.pad_assignments(em.final_assignments, meta["P"])
    traj = jc._traj_jit(jc.f64(np.array([np.ravel(f) for f in em.cluster_f])),
                        jc.f64(em.final_kappa), jc.f64(em.final_scalar_K), jc.f64(toy["K"]),
                        jc.f64(toy["h"]), n_steps=toy["n_steps"])
    c = jc.f64(em.final_c)
    r_w = np.asarray(jc.recon_residuals(traj, jc.f64(em.final_s), beta, z,
                                        jc.shift_data(data, c), jc.f64(toy["h"])))
    r_u = np.asarray(jc.recon_residuals(traj, jc.f64(em.final_s), beta, z,
                                        jc.shift_data(data_u, c), jc.f64(toy["h"])))
    sse_b = np.sum(r_u ** 2, axis=0)
    assert np.isclose(np.sum(r_w ** 2), np.sum(w * sse_b), rtol=1e-12)

    # The held-out score uses the same weights; per-value denominator = observed cells.
    em.biomarker_weights = w
    tr = em.transform(pats)
    sse_ho, n_ho = em.sse_per_biomarker(pats, transformed=tr)
    weighted = em._compute_val_score(em._shift_patients(pats), tr["beta"], tr["subtype"],
                                     weighted=True)
    assert np.isclose(weighted, np.sum(w * sse_ho), rtol=1e-10)
    assert np.isclose(-em.score(pats), np.sum(w * sse_ho) / n_ho.sum(), rtol=1e-10)
    em.biomarker_weights = None


def test_score_blocks_recombine_to_unweighted_total(recovered):
    toy, em = recovered
    pats = toy["pats"][:30]
    groups = {"imaging": np.array([0, 1, 2]), "clinical": np.array([3])}
    tr = em.transform(pats)
    blocks = em.score_blocks(pats, groups, transformed=tr)
    _, n_obs = em.sse_per_biomarker(pats, transformed=tr)
    n_img, n_clin = n_obs[:3].sum(), n_obs[3]
    total = (blocks["imaging"] * n_img + blocks["clinical"] * n_clin) / (n_img + n_clin)
    assert np.isclose(total, -em.score(pats, weighted=False), rtol=1e-10)


def test_bic_ignores_weights():
    toy = _zsym_toy([2.0] * 4, n_pat=30)
    w = np.array([5.0, 1.0, 1.0, 0.2])
    em = _zsym_em(toy, max_iter=3, biomarker_weights=w).fit(toy["pats"])
    flat = em._prepare_data(toy["pats"])
    assert np.array_equal(flat["obs_mask"], np.isfinite(np.vstack(
        [p["X_obs"] for p in toy["pats"]])).astype(float))
    # _sse_per_biomarker is unweighted: recompute it from the fit at unit weights.
    sse_u, _ = em.sse_per_biomarker(toy["pats"], transformed=np.array(
        list(zip(np.asarray(em.beta_history)[:, -1], em.final_assignments)),
        dtype=[("beta", "f8"), ("subtype", "i4")]))
    assert np.allclose(em._sse_per_biomarker, sse_u, rtol=1e-4)


# 7. Transform consistency ----------------------------------------------------------

@pytest.mark.parametrize("weights", [None, np.array([3.0, 1.0, 1.0, 0.5])])
def test_transform_on_training_matches_fit(weights):
    toy = _zsym_toy([1.5, 2.5, 1.5, 2.5])
    em = _zsym_em(toy, biomarker_weights=weights, transform_backend="jax").fit(toy["pats"])
    tr = em.transform(toy["pats"])
    beta_fit = np.asarray(em.beta_history)[:, -1]
    assert np.mean(tr["subtype"] == em.final_assignments) >= 0.95
    same = tr["subtype"] == em.final_assignments
    assert np.median(np.abs(tr["beta"][same] - beta_fit[same])) < 0.1
