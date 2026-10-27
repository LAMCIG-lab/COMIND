"""Integral CVIC scoring and ordered Nadeau–Bengio Tukey selection of k."""

import numpy as np
import pytest
from sklearn.base import BaseEstimator

from COMIND_transformer.cvic import (
    gauss_hermite_log_integral,
    log_kde_prior,
    ordered_tukey_nb,
    silverman_bandwidth,
    _hermite_nodes,
)
from COMIND_transformer.select_n_subtypes import (
    SelectNSubtypes,
    aggregate_cvic_scores,
    select_n_subtypes,
)
from COMIND_transformer.subtyping_em_transformer import SubtypingEM
from COMIND_transformer.utils import split_observed


def test_gauss_hermite_unit_gaussian_log_integral_is_zero():
    u, log_w = _hermite_nodes(15)
    tau = 1.0
    betas = np.sqrt(2.0) * tau * u
    g = -0.5 * betas ** 2 - 0.5 * np.log(2.0 * np.pi)
    log_z = gauss_hermite_log_integral(g, tau, log_w)
    assert abs(log_z) < 1e-12


def test_log_kde_matches_manual_logsumexp():
    rng = np.random.default_rng(0)
    atoms = rng.normal(3.0, 1.0, size=40)
    bw = silverman_bandwidth(atoms)
    x = np.linspace(-2.0, 8.0, 25)
    got = log_kde_prior(x, atoms, bw)
    log_comp = (
        -0.5 * np.log(2.0 * np.pi * bw ** 2)
        - 0.5 * ((x[:, None] - atoms) / bw) ** 2
    )
    want = np.log(np.exp(log_comp).sum(axis=1) / atoms.size)
    np.testing.assert_allclose(got, want, atol=1e-12)


def test_ordered_tukey_selects_smallest_credible_k():
    rng = np.random.default_rng(0)
    n = 120
    subject = rng.normal(0.0, 1.0, size=n)
    # k=1 is clearly worse; k=2,3,4 agree within noise.
    means = np.array([-0.8, 0.0, 0.01, -0.01])
    X = subject[:, None] + means + rng.normal(0.0, 0.04, size=(n, 4))
    k, info = ordered_tukey_nb(X, n_folds=5, alpha=0.05, n_draw=20_000, seed=0)
    assert k == 2
    assert info["bonferroni"]
    assert info["alpha"] == pytest.approx(0.05 / 3)
    assert info["rejected"][0]
    assert not info["rejected"][1]
    np.testing.assert_allclose(info["cvic"], -2.0 * X.sum(axis=0))


def test_ordered_tukey_all_rejected_returns_kmax():
    rng = np.random.default_rng(1)
    n = 80
    subject = rng.normal(0.0, 0.2, size=n)
    means = np.array([0.0, 1.0, 2.0, 3.0])
    X = subject[:, None] + means + rng.normal(0.0, 0.02, size=(n, 4))
    k, info = ordered_tukey_nb(X, n_folds=5, alpha=0.05, n_draw=20_000, seed=0)
    assert k == 4
    assert np.all(info["rejected"])


class _FakeEM(BaseEstimator):
    def __init__(self, n_subtypes=1, lambda_f=0.0, table=None):
        self.n_subtypes = n_subtypes
        self.lambda_f = lambda_f
        self.table = table

    def fit(self, X):
        return self

    def subject_loglik_cvic(self, X, n_gh=15):
        idx = np.array([p["idx"] for p in X], dtype=int)
        return np.asarray(self.table, dtype=np.float64)[idx, int(self.n_subtypes) - 1]


def test_select_n_subtypes_fake_estimator_and_alias():
    rng = np.random.default_rng(2)
    n = 60
    subject = rng.normal(0.0, 1.0, size=n)
    means = np.array([-0.7, 0.0, 0.02])
    table = subject[:, None] + means + rng.normal(0.0, 0.05, size=(n, 3))
    X = [{"idx": i, "id": i} for i in range(n)]
    sel = select_n_subtypes(
        estimator=_FakeEM(table=table),
        max_k=3,
        n_folds=5,
        fold_seed=0,
        n_draw=20_000,
        verbose=0,
        refit=True,
        hyperparams={},
    )
    sel.fit(X)
    assert sel.n_subtypes_ == 2
    assert sel.log_scores_.shape == (n, 3)
    np.testing.assert_allclose(sel.log_scores_, table)
    assert isinstance(sel.best_estimator_, _FakeEM)
    assert sel.best_estimator_.n_subtypes == 2
    assert SelectNSubtypes is select_n_subtypes


def test_preset_hyperparams_dict_and_per_k_list():
    rng = np.random.default_rng(3)
    n = 40
    subject = rng.normal(size=n)
    table = subject[:, None] + np.array([-1.0, 0.0, 0.0])
    X = [{"idx": i} for i in range(n)]
    common = dict(
        estimator=_FakeEM(table=table), max_k=3, n_folds=5, n_draw=5_000,
        verbose=0, refit=True,
    )
    shared = select_n_subtypes(**common, hyperparams={"lambda_f": 0.3})
    shared.fit(X)
    assert all(p["lambda_f"] == 0.3 for row in shared.fold_params_ for p in row)
    assert shared.best_params_ == {"lambda_f": 0.3}
    assert shared.best_estimator_.lambda_f == 0.3

    per_k = [{"lambda_f": 0.1}, {"lambda_f": 0.2}, {"lambda_f": 0.4}]
    listed = select_n_subtypes(**common, hyperparams=per_k)
    listed.fit(X)
    for k, row in enumerate(listed.fold_params_):
        assert all(p["lambda_f"] == per_k[k]["lambda_f"] for p in row)
    assert listed.best_params_["lambda_f"] == per_k[listed.n_subtypes_ - 1]["lambda_f"]

    with pytest.raises(ValueError):
        select_n_subtypes(**common, hyperparams=[{"lambda_f": 1.0}]).fit(X)

    parallel = select_n_subtypes(**common, hyperparams=per_k, n_jobs=2)
    parallel.fit(X)
    np.testing.assert_allclose(parallel.log_scores_, listed.log_scores_)
    assert parallel.n_subtypes_ == listed.n_subtypes_
    assert parallel.fold_params_ == listed.fold_params_

    one = select_n_subtypes(**common, hyperparams=per_k, only_k=2, n_jobs=1)
    one.fit(X)
    assert one.n_subtypes_ is None
    np.testing.assert_allclose(one.log_scores_[:, 1], listed.log_scores_[:, 1])
    assert np.isnan(one.log_scores_[:, 0]).all()
    assert np.isnan(one.log_scores_[:, 2]).all()
    stacked = listed.log_scores_.copy()
    chosen, info = aggregate_cvic_scores(stacked, n_folds=5, n_draw=5_000, seed=0)
    assert chosen == listed.n_subtypes_
    np.testing.assert_allclose(info["cvic"], listed.cvic_)


class _ScoreFake(BaseEstimator):
    def __init__(self, n_subtypes=1, lambda_f=0.5, lambda_kappa=0.5,
                 lambda_scalar=0.5, lambda_jsd=0.5, table=None):
        self.n_subtypes = n_subtypes
        self.lambda_f = lambda_f
        self.lambda_kappa = lambda_kappa
        self.lambda_scalar = lambda_scalar
        self.lambda_jsd = lambda_jsd
        self.table = table

    def fit(self, X):
        return self

    def score(self, X, y=None):
        return -abs(float(self.lambda_f) - 1.0)

    def subject_loglik_cvic(self, X, n_gh=15):
        idx = np.array([p["idx"] for p in X], dtype=int)
        return np.asarray(self.table, dtype=np.float64)[idx, int(self.n_subtypes) - 1]


def test_default_fit_runs_optuna_inside_the_training_fold():
    pytest.importorskip("optuna")
    n = 8
    table = np.zeros((n, 1))
    X = [{"idx": i} for i in range(n)]
    sel = SelectNSubtypes(
        estimator=_ScoreFake(table=table),
        max_k=1,
        n_folds=2,
        hyperparams=None,
        n_optuna_trials=2,
        n_inner_folds=2,
        n_startup_trials=1,
        n_draw=100,
        verbose=0,
        refit=True,
    )
    sel.fit(X)
    assert sel.n_subtypes_ == 1
    for key in ("lambda_f", "lambda_kappa", "lambda_scalar", "lambda_jsd"):
        assert key in sel.fold_params_[0][0]
        assert key in sel.best_params_
    assert sel.best_estimator_.lambda_f == sel.best_params_["lambda_f"]


def _toy_patients(n=16, B=3, seed=0):
    rng = np.random.default_rng(seed)
    K = np.zeros((B, B))
    K[0, 1] = K[1, 0] = 1.0
    pats = []
    for i in range(n):
        z = i % 2
        dt = np.array([0.0, 1.0, 2.0])
        x = (0.04 + 0.03 * z) * (dt[:, None] + 4.0) + rng.normal(0.0, 0.02, (3, B))
        pats.append({
            "id": i,
            "X_obs": np.clip(x, 0.0, None),
            "dt": dt,
            "cog": np.zeros((3, 1)),
            "subtype_true": z,
        })
    return pats, K


def test_fit_stores_empirical_cvic_quantities_and_scores_held_out():
    pats, K = _toy_patients()
    B = K.shape[0]
    em = SubtypingEM(
        K=K,
        n_subtypes=2,
        t_max=12.0,
        step=0.1,
        max_iter=2,
        verbose=0,
        backend="numpy",
        theta_solver_stages=("lbfgs_approx",),
        rng=np.random.default_rng(1),
        initial_s=np.ones(B),
        initial_scalar_K=0.2,
        initial_kappa=np.zeros(B),
        initial_f=np.full(B, 0.05),
    )
    em.fit(pats)
    assert em.sigma_sq_.shape == (B,)
    assert np.all(em.sigma_sq_ > 0)
    np.testing.assert_allclose(em.sigma_, np.sqrt(em.sigma_sq_))
    np.testing.assert_allclose(em.subtype_prior_.sum(), 1.0)
    assert em.beta_train_.shape == (len(pats),)
    assert em.beta_kde_bandwidth_ > 0
    n_obs_b = np.vstack([split_observed(p["X_obs"])[1] for p in pats]).sum(axis=0)
    np.testing.assert_allclose(
        em.sigma_sq_,
        em._sse_per_biomarker / n_obs_b,
        rtol=1e-12,
    )

    ll = em.subject_loglik_cvic(pats[:4], n_gh=8)
    assert ll.shape == (4,)
    assert np.all(np.isfinite(ll))
    cvic = em.score_cvic(pats[:4], n_gh=8)
    np.testing.assert_allclose(cvic, -2.0 * ll.sum())

    missing = [{**pats[0], "X_obs": np.full_like(pats[0]["X_obs"], np.nan)}]
    np.testing.assert_allclose(em.subject_loglik_cvic(missing, n_gh=8), [0.0])
    # caller arrays are not written
    raw = pats[0]["X_obs"].copy()
    em.subject_loglik_cvic([pats[0]], n_gh=8)
    np.testing.assert_array_equal(pats[0]["X_obs"], raw)
