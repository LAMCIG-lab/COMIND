"""
Select the number of subtypes by integral CVIC and an ordered Tukey test.

Same protocol as DPMoSt ``select_k`` in CVIC mode: one set of folds for every
k, each subject scored only when held out, then the Nadeau–Bengio-corrected
one-sided Tukey sequence. Bonferroni divides the level by ``K_max - 1``.

Hyperparameters inside each training fold follow the synthetic Optuna search:
TPE over ``lambda_f``, ``lambda_kappa``, ``lambda_scalar``, and ``lambda_jsd``,
minimizing mean held-out reconstruction error (``-estimator.score``). Pass
``hyperparams`` to skip the search.
"""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import KFold

from .cvic import ordered_tukey_nb


def suggest_hyperparams(trial):
    """Search ranges from the 2026-10-10 synthetic Optuna notebook.

    ``n_subtypes``, ``lambda_cog``, and ``epsilon`` stay on the estimator.
    """
    return {
        "lambda_f": trial.suggest_float("lambda_f", 1e-1, 2.0, log=True),
        "lambda_kappa": trial.suggest_float("lambda_kappa", 1e-1, 2.0, log=True),
        "lambda_scalar": trial.suggest_float("lambda_scalar", 0.0, 1.0),
        "lambda_jsd": trial.suggest_float("lambda_jsd", 0.0, 1.0),
    }


def _preset_per_k(hyperparams, max_k):
    """``None`` means Optuna. A dict is reused for every k. A list is one dict per k."""
    if hyperparams is None:
        return None
    if isinstance(hyperparams, dict):
        return [dict(hyperparams) for _ in range(max_k)]
    if isinstance(hyperparams, (list, tuple)):
        if len(hyperparams) != max_k:
            raise ValueError(
                f"hyperparams list must have one dict per k (length {max_k}), "
                f"got {len(hyperparams)}"
            )
        out = []
        for i, item in enumerate(hyperparams):
            if not isinstance(item, dict):
                raise TypeError(
                    f"hyperparams[{i}] must be a dict, got {type(item).__name__}"
                )
            out.append(dict(item))
        return out
    raise TypeError(
        "hyperparams must be None (Optuna), a dict, or a list of dicts, "
        f"got {type(hyperparams).__name__}"
    )


def _optuna_best_params(
    estimator,
    X_train,
    k,
    *,
    n_optuna_trials,
    n_inner_folds,
    inner_seed,
    optuna_seed,
    n_startup_trials,
    verbose,
):
    """Minimize mean inner-fold ``-score`` on ``X_train`` only."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    n_train = len(X_train)
    n_inner = int(n_inner_folds)
    if n_inner < 2:
        raise ValueError(f"n_inner_folds must be >= 2, got {n_inner_folds!r}")
    if n_train < n_inner:
        raise ValueError(
            f"training fold has {n_train} subjects, fewer than "
            f"n_inner_folds={n_inner}"
        )
    inner = KFold(n_splits=n_inner, shuffle=True, random_state=int(inner_seed))

    def objective(trial):
        hyper = suggest_hyperparams(trial)
        fold_lse = []
        try:
            for tr_idx, va_idx in inner.split(np.arange(n_train)):
                est = clone(estimator)
                est.set_params(n_subtypes=k, **hyper)
                est.fit([X_train[i] for i in tr_idx])
                fold_lse.append(-float(est.score([X_train[i] for i in va_idx])))
        except Exception as exc:
            if verbose >= 1:
                print(f"  trial {trial.number} failed: {exc}", flush=True)
            return float("inf")
        mean_lse = float(np.mean(fold_lse))
        if verbose >= 2:
            print(
                f"  trial {trial.number}: mean CV LSE = {mean_lse:.6f}  "
                f"params = {hyper}",
                flush=True,
            )
        return mean_lse

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=int(optuna_seed),
            n_startup_trials=int(n_startup_trials),
        ),
    )
    study.optimize(
        objective,
        n_trials=int(n_optuna_trials),
        show_progress_bar=False,
    )
    if not np.isfinite(study.best_value):
        raise RuntimeError(
            f"every Optuna trial failed for n_subtypes={k} "
            f"on {n_train} training subjects"
        )
    return dict(study.best_params), float(study.best_value)


def _cvic_one_fold(
    estimator,
    X,
    train_idx,
    test_idx,
    k,
    preset,
    n_gh,
    fold_i,
    max_k,
    n_folds,
    verbose,
    n_optuna_trials,
    n_inner_folds,
    inner_seed,
    optuna_seed,
    n_startup_trials,
):
    """One training fit and the held-out integral scores for a single fold."""
    train_idx = np.asarray(train_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)
    X_train = [X[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    if preset is None:
        if verbose >= 1:
            print(
                f"Optuna  n_subtypes={k}/{max_k}  "
                f"fold={fold_i + 1}/{n_folds}  n_train={len(X_train)}",
                flush=True,
            )
        params, best_lse = _optuna_best_params(
            estimator,
            X_train,
            k,
            n_optuna_trials=n_optuna_trials,
            n_inner_folds=n_inner_folds,
            inner_seed=inner_seed,
            optuna_seed=optuna_seed,
            n_startup_trials=n_startup_trials,
            verbose=verbose,
        )
        if verbose >= 1:
            print(f"  best inner CV LSE {best_lse:.6f}  params {params}", flush=True)
    else:
        params = dict(preset)
    if verbose >= 1:
        print(
            f"CVIC  n_subtypes={k}/{max_k}  fold={fold_i + 1}/{n_folds}  "
            f"n_train={len(X_train)} n_test={len(X_test)}",
            flush=True,
        )
    est = clone(estimator)
    est.set_params(n_subtypes=int(k), **params)
    est.fit(X_train)
    ll = np.asarray(est.subject_loglik_cvic(X_test, n_gh=n_gh), dtype=np.float64)
    if ll.shape != (len(test_idx),):
        raise ValueError(
            f"subject_loglik_cvic returned shape {ll.shape}, "
            f"expected ({len(test_idx)},)"
        )
    return int(k), int(fold_i), dict(params), test_idx, ll


class SelectNSubtypes(BaseEstimator):
    """
    Cross-validated integral CVIC selection of ``n_subtypes``.

    Parameters
    ----------
    estimator : SubtypingEM
        Template. Cloned per fit. Restarts that live on the estimator are
        left as they are; this class does not add more.
    max_k : int
        Fit k = 1, ..., ``max_k``. With ``only_k`` set, only that one candidate
        is fitted and the Tukey selection is left for a later aggregate.
    only_k : int or None
        ``None`` scores every k up to ``max_k`` and selects. An int scores
        only that candidate, with ``n_jobs`` applying to its folds.
    n_folds : int
        Outer CVIC folds. Default 5. The same splits are reused for every k.
    fold_seed : int
        Outer KFold shuffle seed.
    tukeys_alpha : float
        Nominal level. With ``bonferroni=True`` each test uses
        ``tukeys_alpha / (max_k - 1)``.
    bonferroni : bool
        Divide the Tukey level by ``max_k - 1``. Default True.
    hyperparams : None, dict, or list of dict
        ``None`` (default): Optuna search inside each training fold, then
        again on all of X when ``refit`` is True.
        A dict: those parameters for every k.
        A list of ``max_k`` dicts: ``hyperparams[k - 1]`` for that k.
    n_optuna_trials, n_inner_folds, inner_seed, optuna_seed, n_startup_trials
        Inner search. Defaults match the synthetic notebook: 5 TPE trials,
        3-fold inner CV, seeds 75, 5 startup trials.
    n_gh : int
        Gauss–Hermite nodes for the β integral (default 15).
    n_jobs : int
        joblib workers for the CVIC tasks. Every pair ``(k, fold)`` is one
        task, submitted together. ``1`` runs that batch in this process.
        ``-1`` uses every core.
    n_draw, mc_seed
        Monte Carlo null for the studentized range.
    verbose : int
    refit : bool
        Fit ``best_estimator_`` on all of X at the selected k, with the
        hyperparameters chosen for that k.

    Attributes
    ----------
    n_subtypes_ : int
    log_scores_ : ndarray, shape (n_subjects, max_k)
    cvic_ : ndarray, shape (max_k,)
    tukey_ : dict
    cv_splits_ : list of (train_idx, test_idx)
    fold_params_ : list of lists of dict
        ``fold_params_[k - 1][fold]`` used to fit that training fold.
    best_params_ : dict or None
        Parameters of ``best_estimator_`` when ``refit`` is True.
    best_estimator_ : fitted estimator or None
    """

    def __init__(
        self,
        estimator,
        max_k,
        only_k=None,
        n_folds=5,
        fold_seed=0,
        tukeys_alpha=0.05,
        bonferroni=True,
        hyperparams=None,
        n_optuna_trials=5,
        n_inner_folds=3,
        inner_seed=75,
        optuna_seed=75,
        n_startup_trials=5,
        n_gh=15,
        n_jobs=1,
        n_draw=200_000,
        mc_seed=0,
        verbose=1,
        refit=True,
    ):
        self.estimator = estimator
        self.max_k = max_k
        self.only_k = only_k
        self.n_folds = n_folds
        self.fold_seed = fold_seed
        self.tukeys_alpha = tukeys_alpha
        self.bonferroni = bonferroni
        self.hyperparams = hyperparams
        self.n_optuna_trials = n_optuna_trials
        self.n_inner_folds = n_inner_folds
        self.inner_seed = inner_seed
        self.optuna_seed = optuna_seed
        self.n_startup_trials = n_startup_trials
        self.n_gh = n_gh
        self.n_jobs = n_jobs
        self.n_draw = n_draw
        self.mc_seed = mc_seed
        self.verbose = verbose
        self.refit = refit

    def _optuna_best_params(self, X_train, k):
        """Minimize mean inner-fold ``-score`` on ``X_train`` only."""
        return _optuna_best_params(
            self.estimator,
            X_train,
            k,
            n_optuna_trials=self.n_optuna_trials,
            n_inner_folds=self.n_inner_folds,
            inner_seed=self.inner_seed,
            optuna_seed=self.optuna_seed,
            n_startup_trials=self.n_startup_trials,
            verbose=self.verbose,
        )

    def _fit_at(self, X_train, k, params):
        est = clone(self.estimator)
        est.set_params(n_subtypes=k, **params)
        est.fit(X_train)
        return est

    def fit(self, X, y=None):
        max_k = int(self.max_k)
        n_folds = int(self.n_folds)
        if max_k < 1:
            raise ValueError(f"max_k must be >= 1, got {self.max_k!r}")
        if n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {self.n_folds!r}")
        n = len(X)
        if n < n_folds:
            raise ValueError(f"n_folds={n_folds} > n_subjects={n}")
        if n < 2:
            raise ValueError("need at least 2 subjects to select n_subtypes")

        presets = _preset_per_k(self.hyperparams, max_k)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=int(self.fold_seed))
        splits = list(kf.split(np.arange(n)))
        self.cv_splits_ = splits
        self.fold_params_ = [[None] * n_folds for _ in range(max_k)]

        if self.only_k is None:
            ks = range(1, max_k + 1)
        else:
            only_k = int(self.only_k)
            if not 1 <= only_k <= max_k:
                raise ValueError(f"only_k must be in 1..{max_k}, got {self.only_k!r}")
            ks = (only_k,)
        log_scores = np.full((n, max_k), np.nan, dtype=np.float64)
        n_jobs = int(self.n_jobs)
        tasks = (
            delayed(_cvic_one_fold)(
                self.estimator,
                X,
                train_idx,
                test_idx,
                k,
                None if presets is None else presets[k - 1],
                self.n_gh,
                fold_i,
                max_k,
                n_folds,
                self.verbose,
                self.n_optuna_trials,
                self.n_inner_folds,
                self.inner_seed,
                self.optuna_seed,
                self.n_startup_trials,
            )
            for k in ks
            for fold_i, (train_idx, test_idx) in enumerate(splits)
        )
        for k, fold_i, params, test_idx, ll in Parallel(n_jobs=n_jobs)(tasks):
            self.fold_params_[k - 1][fold_i] = params
            log_scores[test_idx, k - 1] = ll

        filled = np.ones(max_k, dtype=bool)
        if self.only_k is not None:
            filled[:] = False
            filled[int(self.only_k) - 1] = True
        if np.isnan(log_scores[:, filled]).any():
            raise RuntimeError("internal error: some held-out CVIC scores were not filled")

        self.log_scores_ = log_scores
        if self.only_k is not None:
            self.cvic_ = None
            self.tukey_ = None
            self.n_subtypes_ = None
            self.best_params_ = None
            self.best_estimator_ = None
            if self.verbose >= 1:
                col = int(self.only_k) - 1
                score = float(-2.0 * np.sum(log_scores[:, col]))
                print(
                    f"CVIC  k={int(self.only_k)}:{score:.3f}  "
                    f"(only this candidate; selection waits for the aggregate)"
                )
            return self
        selected, info = ordered_tukey_nb(
            log_scores,
            n_folds=n_folds,
            alpha=self.tukeys_alpha,
            n_draw=self.n_draw,
            seed=self.mc_seed,
            bonferroni=self.bonferroni,
        )
        self.cvic_ = info["cvic"]
        self.tukey_ = info
        self.n_subtypes_ = int(selected)

        if self.verbose >= 1:
            cvic_str = ", ".join(
                f"k={k}:{self.cvic_[k - 1]:.3f}" for k in range(1, max_k + 1)
            )
            level = (
                f"alpha={self.tukeys_alpha}/(K_max-1)={info['alpha']}"
                if self.bonferroni and max_k > 1
                else f"alpha={info['alpha']}"
            )
            print(f"CVIC  {cvic_str}")
            print(f"selected n_subtypes={self.n_subtypes_}  (ordered Tukey, {level})")

        if self.refit:
            if presets is None:
                if self.verbose >= 1:
                    print(f"Optuna on all {n} subjects at n_subtypes={self.n_subtypes_}")
                params, best_lse = self._optuna_best_params(list(X), self.n_subtypes_)
                if self.verbose >= 1:
                    print(f"  best inner CV LSE {best_lse:.6f}  params {params}")
            else:
                params = presets[self.n_subtypes_ - 1]
            self.best_params_ = dict(params)
            if self.verbose >= 1:
                print(
                    f"refitting on all {n} subjects at n_subtypes={self.n_subtypes_} "
                    f"params={self.best_params_}"
                )
            self.best_estimator_ = self._fit_at(list(X), self.n_subtypes_, self.best_params_)
        else:
            self.best_params_ = None
            self.best_estimator_ = None
        return self

    def transform(self, X):
        if getattr(self, "best_estimator_", None) is None:
            raise RuntimeError("fit() with refit=True is required before transform()")
        return self.best_estimator_.transform(X)


select_n_subtypes = SelectNSubtypes


def aggregate_cvic_scores(log_scores, n_folds, alpha=0.05, n_draw=200_000, seed=0, bonferroni=True):
    """Run the ordered Tukey selection on held-out scores stacked over k.

    ``log_scores`` has shape ``(n_subjects, max_k)`` and one column per
    candidate, in order k = 1, 2, .... Each column is produced by a fit with
    ``only_k`` set.
    """
    scores = np.asarray(log_scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] < 1:
        raise ValueError(f"log_scores must have shape (n_subjects, max_k), got {scores.shape}")
    if np.isnan(scores).any():
        raise ValueError("log_scores still has missing entries; every candidate k is required")
    return ordered_tukey_nb(
        scores,
        n_folds=n_folds,
        alpha=alpha,
        n_draw=n_draw,
        seed=seed,
        bonferroni=bonferroni,
    )
