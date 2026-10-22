"""Tests for the per-biomarker intercept (prediction = c + s * x)."""
import numpy as np
import pytest

from COMIND_transformer import jax_core as jc
from COMIND_transformer.subtyping_em_transformer import SubtypingEM


def _toy(seed=0, offset=None, n_pat=60, B=4, t_max=20.0, step=0.05):
    rng = np.random.default_rng(seed)
    K = np.zeros((B, B)); K[1, 0] = K[0, 1] = 1.0; K[2, 1] = K[1, 2] = 1.0
    F = np.zeros((2, B)); F[0, 0] = 0.15; F[1, 3] = 0.15
    t_span = np.linspace(0, t_max, int(t_max / step))
    _, h, n = jc.grid_from_t_span(t_span)
    tr = np.asarray(jc._traj_jit(jc.f64(F), jc.f64(np.zeros(B)), jc.f64(0.2), jc.f64(K), jc.f64(h), n_steps=n))
    off = np.zeros(B) if offset is None else np.asarray(offset, float)
    pats = []
    for i in range(n_pat):
        z = i % 2; b0 = rng.uniform(1, 12); dt = np.array([0.0, 1.0, 2.0])
        mean = np.array([[np.interp(b0 + d, t_span, tr[z, :, b]) for b in range(B)] for d in dt])
        pats.append({"X_obs": mean + off + rng.normal(0, 0.05, mean.shape), "dt": dt,
                     "cog": np.zeros((3, 1)), "initial_beta": b0})
    return K, pats, t_max, step


def _em(K, t_max, step, **kw):
    B = K.shape[0]
    args = dict(K=K, initial_f=np.full(B, 0.02), initial_s=np.ones(B), initial_scalar_K=0.1,
                initial_kappa=np.zeros(B), n_subtypes=2, max_iter=3, t_max=t_max, step=step,
                fit_s=False, s_mask=np.zeros(B, bool), verbose=0, backend="jax",
                rng=np.random.default_rng(0))
    args.update(kw)
    return SubtypingEM(**args)


def test_shift_data_matches_intercept_prediction():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(7, 3)); X[2, 1] = np.nan
    data, meta = jc.pack_data(X, None, np.zeros(7), np.arange(7) % 2, np.zeros((7, 1)), 2)
    c = np.array([0.5, -1.0, 2.0])
    shifted = jc._shift_data_jit(data, jc.f64(c))
    got = np.asarray(shifted.X)[:7]
    want = np.where(np.isfinite(X), X, 0.0) - c
    w = np.asarray(data.W)[:7]
    assert np.allclose(got * w, want * w)


def test_globals_loss_intercept_zero_c_equals_globals_loss():
    K, pats, t_max, step = _toy()
    X = np.vstack([p["X_obs"] for p in pats]); dt = np.concatenate([p["dt"] for p in pats])
    ids = np.repeat(np.arange(len(pats)), 3)
    data, meta = jc.pack_data(X, None, dt, ids, np.zeros((len(dt), 1)), len(pats))
    t_span = np.linspace(0, t_max, int(t_max / step)); _, h, n = jc.grid_from_t_span(t_span)
    B = K.shape[0]; F = jc.f64(np.full((2, B), 0.05)); hyp = jc.make_hyp()
    beta = jc.pad_beta(np.full(len(pats), 3.0), meta["P"])
    z = jc.pad_assignments(np.arange(len(pats)) % 2, meta["P"])
    args = (jc.f64(np.ones(B)), jc.f64(np.zeros(B)), jc.f64(0.1))
    base = float(jc.globals_loss(*args, F, beta, z, data, jc.f64(K), hyp, jc.f64(h), n_steps=n))
    with_c = float(jc.globals_loss_intercept(*args, jc.f64(np.zeros(B)), F, beta, z, data,
                                             jc.f64(K), hyp, jc.f64(h), jc.f64(np.zeros(B)), n_steps=n))
    assert np.isclose(base, with_c)


def test_intercept_recovers_offset_and_bic_counts_it():
    offset = np.array([0.8, 0.8, 0.8, 0.8])
    K, pats, t_max, step = _toy(offset=offset)
    # Start c near the reference level (in the driver: the CN mean). Starting
    # at 0 lets f absorb the offset instead (a local optimum).
    em = _em(K, t_max, step, fit_intercept=True, c_bounds=(0.0, np.inf), max_iter=20,
             initial_c=np.full(4, 0.6))
    em.fit(pats)
    assert np.allclose(em.final_c, offset, atol=0.1)
    assert em._n_free_c(K.shape[0]) == K.shape[0]


def test_intercept_bounds_and_mask_hold():
    K, pats, t_max, step = _toy(offset=[0.8] * 4)
    mask = np.array([True, True, True, False])
    em = _em(K, t_max, step, fit_intercept=True, c_bounds=(0.0, 0.5), c_mask=mask,
             initial_c=np.array([0.1, 0.1, 0.1, 0.3]))
    em.fit(pats)
    assert np.all(em.final_c[:3] <= 0.5 + 1e-9) and np.all(em.final_c[:3] >= -1e-9)
    assert em.final_c[3] == pytest.approx(0.3)


def test_no_intercept_is_unchanged():
    K, pats, t_max, step = _toy()
    a = _em(K, t_max, step).fit(pats)
    b = _em(K, t_max, step, fit_intercept=False).fit(pats)
    assert np.allclose(a.lse_history, b.lse_history)
    assert np.allclose(a.final_c, 0.0)


def test_transform_applies_intercept():
    offset = np.full(4, 0.8)
    K, pats, t_max, step = _toy(offset=offset)
    em = _em(K, t_max, step, fit_intercept=True, c_bounds=(0.0, np.inf), max_iter=20,
             initial_c=np.full(4, 0.6)).fit(pats)
    out = em.transform(pats)
    assert np.isfinite(em.score(pats))
    beta_fit = np.asarray(em.beta_history)[:, -1]
    assert np.corrcoef(out["beta"], beta_fit)[0, 1] > 0.8


def test_numpy_backend_rejected():
    K, _, t_max, step = _toy()
    with pytest.raises(ValueError):
        _em(K, t_max, step, fit_intercept=True, backend="numpy")
