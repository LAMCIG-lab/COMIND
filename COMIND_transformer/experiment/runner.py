"""
Optuna worker for one (experiment, arm, n_subtypes) study.

Layout under the experiment folder (all of it gitignored except config.yaml
and NOTES.md):

    <arm>/k<Z>/journal.log        Optuna journal; every scalar result lives in
                                  trial user_attrs
    <arm>/k<Z>/trials/NNNN.npz    SubtypingEM.save(): fitted model, val placement,
                                  subject IDs and per-subject held-out CVIC
                                  log densities (subject-level data)
    <arm>/k<Z>/manifest.json      git commit, resolved config, versions
    <arm>/k<Z>/lambda_f_max.json  lambda_f calibration, written once
"""
from __future__ import annotations

import datetime as dt
import json
import os
import time
from typing import Any, Dict, Optional

import numpy as np
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.trial import TrialState
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_validate

from ..subtyping_em_transformer import SubtypingEM
from . import adni, ppmi
from .config import ExperimentConfig
from .data import Dataset, biomarker_weights
from .diagnostics import (
    fit_summary, forecast_mse, held_out_scores, json_safe, lambda_f_max_report, make_cv_scorer,
)
from .manifest import write_manifest

LOADERS = {"adni": adni, "ppmi": ppmi}
STALE_SECONDS = 26 * 3600
JOURNAL_RETRIES = 30


def load_dataset(cfg: ExperimentConfig) -> Dataset:
    return LOADERS[cfg.dataset].load(cfg.data_csv, cfg.connectome, t_max=cfg.estimator["t_max"],
                                     seed=cfg.seed, **cfg.data)


def build_estimator(cfg: ExperimentConfig, data: Dataset) -> SubtypingEM:
    """Base estimator; trials set the searched penalties on a clone."""
    kw = dict(cfg.estimator)
    kw.setdefault("s_mask", np.ones(len(data.x_cols), dtype=bool))
    return SubtypingEM(
        K=data.K, n_subtypes=cfg.n_subtypes, init="kmeans", n_init=cfg.n_init,
        random_state=cfg.seed, init_pattern_cols=data.groups["imaging"],
        biomarker_weights=biomarker_weights(data.groups, cfg.clinical_weight_share),
        jax_pad_rows=data.n_rows_train, jax_pad_patients=len(data.x_train), verbose=0, **kw,
    )


def study_dir(exp_dir: str, cfg: ExperimentConfig) -> str:
    return os.path.join(exp_dir, cfg.arm, f"k{cfg.n_subtypes}")


def _is_lock_error(exc: BaseException) -> bool:
    """Optuna's NFS journal lock: another worker held or broke the lock file."""
    msg = str(exc).lower()
    return isinstance(exc, FileNotFoundError) or "lock" in msg or "possess" in msg


def _with_lock_retry(fn):
    for attempt in range(1, JOURNAL_RETRIES + 1):
        try:
            return fn()
        except (OSError, RuntimeError) as exc:
            if attempt == JOURNAL_RETRIES or not _is_lock_error(exc):
                raise
            time.sleep(attempt)


def open_study(journal: str, name: str, n_startup: int) -> optuna.Study:
    """Load the study if it exists, else create it; retry while another worker holds the lock."""
    sampler = optuna.samplers.TPESampler(multivariate=True, group=True, constant_liar=True,
                                         n_startup_trials=n_startup)
    for attempt in range(1, JOURNAL_RETRIES + 1):
        try:
            storage = JournalStorage(JournalFileBackend(journal, lock_obj=JournalFileOpenLock(journal)))
            # create_study(load_if_exists=True) appends a create record on every call;
            # many workers doing that at once tore a journal line on NFS.
            if name in optuna.get_all_study_names(storage=storage):
                return optuna.load_study(study_name=name, storage=storage, sampler=sampler)
            return optuna.create_study(study_name=name, storage=storage, direction="minimize",
                                       sampler=sampler, load_if_exists=True)
        except (OSError, RuntimeError) as exc:
            if attempt == JOURNAL_RETRIES or not _is_lock_error(exc):
                raise
            time.sleep(attempt)
    raise AssertionError("unreachable")


def make_objective(cfg: ExperimentConfig, data: Dataset, base: SubtypingEM, trials_dir: str):
    scorer = make_cv_scorer(data.groups, forecast=True)
    cvic = cfg.cvic or cfg.objective == "cvic"

    def objective(trial: optuna.Trial) -> float:
        t0 = time.perf_counter()
        params = {n: trial.suggest_float(n, lo, hi, log=log) for n, (lo, hi, log) in cfg.search_space.items()}
        est = clone(base).set_params(**params)
        attrs: Dict[str, Any] = {}
        cv_extra: Dict[str, np.ndarray] = {}
        if cfg.uses_cv:
            # Same seed and same x_train for every Z, so every Z study sees the same folds,
            # which the ordered Tukey test (scripts/select_z.py) needs.
            splits = list(KFold(cfg.cv_folds, shuffle=True, random_state=cfg.seed).split(data.x_train))
            cv = cross_validate(clone(est).set_params(n_init=1), data.x_train, cv=splits,
                                scoring=scorer, error_score="raise", return_estimator=cvic)
            for key, folds in cv.items():
                if key.startswith("test_"):
                    attrs[f"cv_{key[5:]}"] = float(np.mean(folds))
                    attrs[f"cv_{key[5:]}_folds"] = folds.tolist()
            if cvic:
                # Held-out log density of every training subject, from the fold that left it out.
                loglik, fold = np.empty(len(data.x_train)), np.empty(len(data.x_train), dtype=int)
                for i, (fold_est, (_, test)) in enumerate(zip(cv["estimator"], splits)):
                    loglik[test] = fold_est.subject_loglik_cvic([data.x_train[j] for j in test])
                    fold[test] = i
                attrs["cv_cvic"] = float(-2.0 * loglik.sum())
                attrs["cv_cvic_folds"] = [float(-2.0 * loglik[fold == i].sum()) for i in range(len(splits))]
                cv_extra = {"cv_loglik": loglik, "cv_fold": fold}

        est.fit(data.x_train)
        if cfg.polish_f:
            est.polish_cluster_f(data.x_train)
        attrs.update(fit_summary(est, data.x_train, data.groups))
        val = held_out_scores(est, data.x_val, data.groups)
        attrs.update({f"val_{k}": v for k, v in val.items() if k not in ("beta", "subtype")})
        attrs["val_forecast"] = forecast_mse(est, data.x_val)
        for name, X in data.extra_val.items():
            attrs[f"val_{name}_lse"] = held_out_scores(est, X, data.groups)["lse"]
        attrs.update(LOADERS[cfg.dataset].external_checks(est, data, val["beta"], val["subtype"]))

        value = {"cv_lse": attrs.get("cv_lse"), "cv_forecast": attrs.get("cv_forecast"),
                 "cvic": attrs.get("cv_cvic"), "bic": attrs["bic"], "bic_gauss": attrs["bic_gauss"]}[cfg.objective]
        attrs["fit_seconds"] = time.perf_counter() - t0
        path = os.path.join(trials_dir, f"{trial.number:04d}.npz")
        est.save(path, histories=cfg.save_histories,
                 train_ids=[str(p["id"]) for p in data.x_train],
                 val_ids=[str(p["id"]) for p in data.x_val],
                 val_beta=val["beta"], val_subtype=val["subtype"], **cv_extra)
        for key, v in json_safe(attrs).items():
            # ~45 locked appends per trial; with hundreds of workers on NFS one occasionally
            # finds its lock broken ("did not possess lock"). Retry rather than lose the fit.
            _with_lock_retry(lambda: trial.set_user_attr(key, v))
        print(f"trial {trial.number}: {cfg.objective}={value:.6g} val_lse={attrs['val_lse']:.6g} "
              f"bic={attrs['bic']:.6g} [{attrs['fit_seconds']:.0f}s]", flush=True)
        if not np.isfinite(value):
            raise optuna.TrialPruned(f"non-finite {cfg.objective}")
        return float(value)

    return objective


def reap_stale(study: optuna.Study) -> int:
    """Mark RUNNING trials older than STALE_SECONDS (dead workers) as FAIL."""
    now, n = dt.datetime.now(), 0
    for t in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
        if t.datetime_start is not None and (now - t.datetime_start).total_seconds() > STALE_SECONDS:
            study.tell(t, state=TrialState.FAIL)
            n += 1
    return n


def run_worker(cfg: ExperimentConfig, exp_dir: str, *, config_path: str,
               n_trials: Optional[int] = None) -> optuna.Study:
    """Load data, then run trials until the study has ``n_trials`` or the worker time is up."""
    out = study_dir(exp_dir, cfg)
    trials_dir = os.path.join(out, "trials")
    os.makedirs(trials_dir, exist_ok=True)
    data = load_dataset(cfg)
    base = build_estimator(cfg, data)
    write_manifest(os.path.join(out, "manifest.json"), cfg, config_path, data.info)

    lam_path = os.path.join(out, "lambda_f_max.json")
    if not os.path.exists(lam_path):
        report = lambda_f_max_report(base, data.x_train, cfg.search_space["lambda_f"][:2], cfg.seed)
        tmp = f"{lam_path}.tmp{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(report, f, indent=2)
        os.replace(tmp, lam_path)

    name = f"{cfg.dataset}_{os.path.basename(os.path.abspath(exp_dir))}_{cfg.arm}_k{cfg.n_subtypes}"
    study = open_study(os.path.join(out, "journal.log"), name, cfg.n_startup_trials)
    reap_stale(study)
    objective = make_objective(cfg, data, base, trials_dir)
    budget = cfg.n_trials if n_trials is None else n_trials
    t_end = time.time() + cfg.worker_hours * 3600
    while time.time() < t_end - 1:
        live = sum(t.state in (TrialState.COMPLETE, TrialState.RUNNING) for t in study.trials)
        if live >= budget:
            break
        try:
            study.optimize(objective, n_trials=1, timeout=t_end - time.time())
        except (OSError, RuntimeError) as exc:
            if not _is_lock_error(exc):
                raise
            # Optuna's own journal write (e.g. the final tell) lost the lock: keep the worker.
            print(f"journal lock error, continuing: {exc}", flush=True)
    return study
