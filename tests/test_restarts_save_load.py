"""SubtypingEM(init="kmeans") restarts, sklearn clone / cross_validate, and save / load."""
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_validate

from COMIND_transformer.initialization import initial_values
from COMIND_transformer.subtyping_em_transformer import SubtypingEM

from test_zsym_weights import _zsym_toy


def _em(toy, **kw):
    B = toy["K"].shape[0]
    args = dict(K=toy["K"], n_subtypes=2, max_iter=3, t_max=toy["t_max"], step=toy["step"],
                fit_s=True, s_lower_value=1.0, s_upper=3.0, verbose=0, backend="jax",
                obs_model="zsym", init="kmeans", n_init=2, random_state=7,
                lambda_f=1e-3, scalar_K_center=0.3, init_pattern_cols=np.arange(B))
    args.update(kw)
    return SubtypingEM(**args)


@pytest.fixture(scope="module")
def toy():
    return _zsym_toy([2.0] * 4, n_pat=30)


@pytest.fixture(scope="module")
def fitted(toy):
    return _em(toy).fit(toy["pats"])


def test_restarts_recorded_and_best_kept(toy, fitted):
    assert len(fitted.restarts_) == 2
    ok = [r for r in fitted.restarts_ if not r["failed"]] or fitted.restarts_
    best = min(ok, key=lambda r: r["objective"])
    assert fitted.chosen_restart_ == best["restart"]
    assert fitted.lse_final == pytest.approx(best["lse_final"])


def test_restart_matches_explicit_start(toy, fitted):
    r = fitted.chosen_restart_
    B = toy["K"].shape[0]
    start = initial_values(r, toy["pats"], 2, np.zeros(B), np.arange(B), 0.3, 7, zsym_s0=2.0)
    em = _em(toy, init="given", n_init=1, rng=start["rng"],
             **{k: start[k] for k in ("initial_f", "initial_s", "initial_kappa",
                                      "initial_scalar_K", "initial_assignments")})
    em.fit(toy["pats"])
    np.testing.assert_allclose(np.asarray(em.cluster_f), np.asarray(fitted.cluster_f))


def test_clone_is_reproducible(toy, fitted):
    again = clone(_em(toy)).fit(toy["pats"])
    np.testing.assert_allclose(np.asarray(again.cluster_f), np.asarray(fitted.cluster_f))
    np.testing.assert_array_equal(again.final_assignments, fitted.final_assignments)


def test_cross_validate_runs(toy):
    res = cross_validate(_em(toy, n_init=1), toy["pats"], cv=KFold(3), error_score="raise")
    assert res["test_score"].shape == (3,)
    assert np.all(np.isfinite(res["test_score"]))


def test_save_load_round_trip(toy, fitted, tmp_path):
    path = fitted.save(str(tmp_path / "fit.npz"), val_ids=np.arange(5))
    est = SubtypingEM.load(path)
    held_out = toy["pats"][:8]
    a, b = fitted.transform(held_out), est.transform(held_out)
    np.testing.assert_allclose(a["beta"], b["beta"])
    np.testing.assert_array_equal(a["subtype"], b["subtype"])
    assert est.score(held_out) == pytest.approx(fitted.score(held_out))
    assert est.score_cvic(held_out, n_gh=6) == pytest.approx(fitted.score_cvic(held_out, n_gh=6))
    np.testing.assert_array_equal(est.final_beta_, fitted.final_beta_)
    assert np.load(path)["val_ids"].tolist() == list(range(5))



def test_zsym_restarts_not_failed_by_init_ratio(toy, monkeypatch):
    # A zsym start over-predicts by construction (real ADNI: ratio 1.7-2.4). That alone
    # must not fail a restart, or selection silently drops the empty-subtype check.
    from COMIND_transformer import initialization
    monkeypatch.setattr(initialization, "init_ratio", lambda *a, **k: 5.0)
    em = _em(toy).fit(toy["pats"])
    assert all(r["init_ratio"] == 5.0 for r in em.restarts_)
    counts = np.asarray(em.subtype_counts_history_)[:, 2]
    assert counts.min() >= 0.02 * len(toy["pats"])  # so no restart should fail
    assert not any(r["failed"] for r in em.restarts_) and not em.failed_all_restarts_


def test_inner_ftol_reaches_globals_and_f_steps(toy, monkeypatch):
    import COMIND_transformer.optimizer_jax as oj
    seen = []
    real = oj.minimize
    monkeypatch.setattr(oj, "minimize", lambda *a, **k: seen.append((k.get("options") or {}).get("ftol")) or real(*a, **k))
    _em(toy, init="given", n_init=1, max_iter=1, inner_ftol=1e-7).fit(toy["pats"])
    assert seen.count(1e-7) >= 2  # globals and cluster-f; the beta step keeps its own options
    seen.clear()
    _em(toy, init="given", n_init=1, max_iter=1).fit(toy["pats"])
    assert 1e-7 not in seen
