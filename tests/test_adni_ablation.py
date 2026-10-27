"""Unit tests for the 2026-10-26 ADNI ablation flags (A2, A4, A5)."""
import importlib.util
import os
import sys

import numpy as np
import pytest

pytest.importorskip("jax")

from COMIND_transformer.subtyping_em_transformer import SubtypingEM

EXP_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "experiments",
    "2026_10_20_ADNI_optuna_connectome_experiment",
)
sys.path.insert(0, EXP_DIR)
from comind_init import initial_values  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "noshift_driver", os.path.join(EXP_DIR, "run_comind_optuna_bic_noshift.py")
)
drv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(drv)


def _toy(seed=0, offset=None, n_pat=24, B=4, t_max=20.0, step=0.05):
    from COMIND_transformer import jax_core as jc

    rng = np.random.default_rng(seed)
    K = np.zeros((B, B))
    K[1, 0] = K[0, 1] = 1.0
    K[2, 1] = K[1, 2] = 1.0
    F = np.zeros((2, B))
    F[0, 0] = 0.15
    F[1, 3] = 0.15
    t_span = np.linspace(0, t_max, int(t_max / step))
    _, h, n = jc.grid_from_t_span(t_span)
    tr = np.asarray(
        jc._traj_jit(jc.f64(F), jc.f64(np.zeros(B)), jc.f64(0.2), jc.f64(K), jc.f64(h), n_steps=n)
    )
    off = np.zeros(B) if offset is None else np.asarray(offset, float)
    pats = []
    for i in range(n_pat):
        z = i % 2
        b0 = rng.uniform(1, 12)
        dt = np.array([0.0, 1.0, 2.0])
        mean = np.array([[np.interp(b0 + d, t_span, tr[z, :, b]) for b in range(B)] for d in dt])
        pats.append({
            "X_obs": mean + off + rng.normal(0, 0.05, mean.shape),
            "dt": dt,
            "cog": np.zeros((3, 1)),
            "initial_beta": b0,
        })
    return K, pats, t_max, step


def test_a2_init_scalar_k_cap():
    K, pats, _, _ = _toy()
    B = K.shape[0]
    c0 = np.zeros(B)
    capped = initial_values(
        0, pats, 2, c0, range(B), 0.654, 0, init_scalar_k_cap=0.02
    )
    assert capped["initial_scalar_K"] == pytest.approx(0.02)
    uncapped = initial_values(0, pats, 2, c0, range(B), 0.654, 0)
    assert uncapped["initial_scalar_K"] == pytest.approx(0.654)
    below = initial_values(
        0, pats, 2, c0, range(B), 0.01, 0, init_scalar_k_cap=0.02
    )
    assert below["initial_scalar_K"] == pytest.approx(0.01)


def test_a4_s_upper_bound():
    K, pats, t_max, step = _toy()
    B = K.shape[0]
    em = SubtypingEM(
        K=K,
        initial_f=np.full(B, 0.02),
        initial_s=np.ones(B),
        initial_scalar_K=0.1,
        initial_kappa=np.zeros(B),
        n_subtypes=2,
        max_iter=4,
        t_max=t_max,
        step=step,
        fit_s=True,
        s_mask=np.ones(B, dtype=bool),
        s_upper=1.0,
        verbose=0,
        backend="jax",
        rng=np.random.default_rng(0),
    )
    em.fit(pats)
    assert np.all(em.final_s <= 1.0 + 1e-9)


def test_a5_forecast_mse_matches_hand_computation_when_c_is_zero():
    K, pats, t_max, step = _toy()
    B = K.shape[0]
    em = SubtypingEM(
        K=K,
        initial_f=np.full(B, 0.02),
        initial_s=np.ones(B),
        initial_scalar_K=0.1,
        initial_kappa=np.zeros(B),
        n_subtypes=2,
        max_iter=4,
        t_max=t_max,
        step=step,
        fit_s=False,
        s_mask=np.zeros(B, dtype=bool),
        fit_intercept=False,
        verbose=0,
        backend="jax",
        rng=np.random.default_rng(0),
    )
    em.fit(pats)
    assert np.allclose(em.final_c, 0.0)

    hist, last = [], []
    for p in pats:
        hist.append({**p, "X_obs": np.asarray(p["X_obs"], float)[:-1],
                     "dt": np.asarray(p["dt"], float)[:-1],
                     "cog": np.asarray(p["cog"])[:-1]})
        last.append({**p, "X_obs": np.asarray(p["X_obs"], float)[-1:],
                     "dt": np.asarray(p["dt"], float)[-1:]})
    tr = em.transform(em._shift_patients(hist), _shifted=True)
    yhat = drv.predict_visits(em, tr["beta"], tr["subtype"], last)
    se, n = 0.0, 0
    for p, y in zip(last, yhat):
        r = p["X_obs"] - y
        ok = np.isfinite(r)
        se += float(np.sum(r[ok] ** 2))
        n += int(ok.sum())
    hand = se / max(n, 1)
    got = drv.forecast_mse(em, pats)
    assert np.isfinite(got)
    assert got == pytest.approx(hand, rel=1e-10, abs=1e-12)


def test_study_paths_exp_dir():
    csv = os.path.join(drv.CONNECTOME_DIR, "K_82_log_knn8_row.csv")
    p = drv.study_paths("/tmp/script", csv, 4, exp_dir="/tmp/ablation_root")
    assert p["exp_dir"] == "/tmp/ablation_root"
    assert p["out_dir"].startswith("/tmp/ablation_root/")
    assert p["journal_path"].startswith("/tmp/ablation_root/")
    default = drv.study_paths("/tmp/script", csv, 4)
    assert default["out_dir"].startswith("/tmp/script/")
