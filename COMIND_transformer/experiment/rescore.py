"""
Re-score each Z's chosen trial on fresh CV folds before the Tukey test.

A study's best trial is the lowest CVIC out of ~60 trials scored on the same
folds, so its CVIC is optimistic. Refitting that trial's parameters on a new
split (seed ``cfg.seed + SEED_OFFSET``, same number of folds, same settings as
the in-trial CV: restart 0 only) gives held-out log densities the selection
never saw. One fold per call, so each runs as its own cluster job:

    <arm>/k<Z>/rescore/fold<F>.npz   trial, test indices, held-out log density, IDs

``select_z(..., rescored=True)`` then reads ``collect`` instead of the trial npz.
"""
from __future__ import annotations

import json
import os
from typing import Optional, Tuple

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold

from .config import ExperimentConfig
from .selection import best_trial

SEED_OFFSET = 1000


def _cfg(study_dir: str) -> ExperimentConfig:
    with open(os.path.join(study_dir, "manifest.json")) as f:
        return ExperimentConfig(**json.load(f)["resolved_config"])


def rescore_fold(study_dir: str, fold: int, by: Optional[str] = "cv_cvic") -> str:
    """Fit fold ``fold`` of the fresh split at the study's best trial; write its held-out scores."""
    from .runner import build_estimator, load_dataset

    cfg = _cfg(study_dir)
    trial = best_trial(study_dir, by=by)
    data = load_dataset(cfg)
    seed = cfg.seed + SEED_OFFSET
    splits = list(KFold(cfg.cv_folds, shuffle=True, random_state=seed).split(data.x_train))
    train, test = splits[int(fold)]
    est = clone(build_estimator(cfg, data)).set_params(n_init=1, **trial.params)
    est.fit([data.x_train[i] for i in train])
    loglik = est.subject_loglik_cvic([data.x_train[i] for i in test])

    out_dir = os.path.join(study_dir, "rescore")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"fold{int(fold)}.npz")
    tmp = f"{path}.tmp{os.getpid()}.npz"
    np.savez(tmp, trial=trial.number, seed=seed, n_folds=cfg.cv_folds, test_idx=test, loglik=loglik,
             train_ids=[str(p["id"]) for p in data.x_train])
    os.replace(tmp, path)
    return path


def collect(study_dir: str) -> Tuple[np.ndarray, np.ndarray, list, int]:
    """``(loglik, fold, train_ids, trial)`` over every training subject, from all fold files."""
    cfg = _cfg(study_dir)
    parts = []
    for f in range(cfg.cv_folds):
        path = os.path.join(study_dir, "rescore", f"fold{f}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} missing; run rescore_fold for fold {f}")
        with np.load(path) as d:
            parts.append({k: d[k] for k in d.files})
    trials = {int(p["trial"]) for p in parts}
    if len(trials) != 1:
        raise ValueError(f"{study_dir}: rescore folds come from different trials {sorted(trials)}")
    ids = parts[0]["train_ids"].tolist()
    loglik, fold = np.full(len(ids), np.nan), np.full(len(ids), -1)
    for i, p in enumerate(parts):
        loglik[p["test_idx"]], fold[p["test_idx"]] = p["loglik"], i
    if np.isnan(loglik).any():
        raise ValueError(f"{study_dir}: rescore folds do not cover every training subject")
    return loglik, fold, ids, trials.pop()
