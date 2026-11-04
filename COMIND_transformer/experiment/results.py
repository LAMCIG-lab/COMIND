"""
Notebook access to a study's trials. Read-only; works while the study runs.

    from COMIND_transformer.experiment import results as R
    df = R.trials_table("experiments/<dated>/W0/k3")   # one row per complete trial
    t = R.load_trial("experiments/<dated>/W0/k3", df["bic"].idxmin())
    t.est                      # fitted SubtypingEM (transform, score, score_cvic, ...)
    t.subjects                 # id, split, beta, subtype, baseline metadata (NSD, PTAU_bl, ...)
    t_span, traj = R.trajectories(t.est)       # (Z, T, B) on the z scale
    R.visits(t)                # long table of every observation for spaghetti plots
    R.visit_metadata(t)        # one row per visit: disease time + NSD / PTAU / DX ...

The trial npz holds the model and each subject's placement, not the data:
``load_trial`` rebuilds the Dataset from the study's manifest (same loader, same
seeded split) and checks the subject IDs match.
"""
from __future__ import annotations

import functools
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import optuna
import pandas as pd
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.trial import TrialState

from ..optimizer_jax import solve_trajectories_jax
from ..subtyping_em_transformer import SubtypingEM
from .config import ExperimentConfig
from .data import Dataset


def trials_table(study_dir: str) -> pd.DataFrame:
    """Params and scalar user_attrs (bic, cv_lse, cv_cvic, val_*, ...) of every complete trial."""
    journal = os.path.join(study_dir, "journal.log")
    storage = JournalStorage(JournalFileBackend(journal, lock_obj=JournalFileOpenLock(journal)))
    (name,) = optuna.get_all_study_names(storage=storage)
    rows = []
    for t in optuna.load_study(study_name=name, storage=storage).trials:
        if t.state != TrialState.COMPLETE:
            continue
        scalars = {k: v for k, v in t.user_attrs.items() if isinstance(v, (int, float, str)) or v is None}
        rows.append({"trial": t.number, "value": t.value, **t.params, **scalars})
    return pd.DataFrame(rows).set_index("trial").sort_index() if rows else pd.DataFrame()


@functools.lru_cache(maxsize=4)
def _dataset(config_json: str) -> Dataset:
    from .runner import load_dataset  # runner imports optuna storage helpers; keep it lazy
    return load_dataset(ExperimentConfig(**json.loads(config_json)))


@dataclass
class Trial:
    number: int
    est: SubtypingEM
    cfg: ExperimentConfig
    data: Dataset
    npz: Dict[str, np.ndarray]
    subjects: pd.DataFrame


def load_trial(study_dir: str, number: int) -> Trial:
    with open(os.path.join(study_dir, "manifest.json")) as f:
        resolved = json.load(f)["resolved_config"]
    path = os.path.join(study_dir, "trials", f"{int(number):04d}.npz")
    est = SubtypingEM.load(path)
    with np.load(path) as d:
        npz = {k: d[k] for k in d.files if not k.startswith(("param__", "fit__"))}
    data = _dataset(json.dumps(resolved, sort_keys=True))
    train_ids = [str(p["id"]) for p in data.x_train]
    val_ids = [str(p["id"]) for p in data.x_val]
    if train_ids != npz["train_ids"].tolist() or val_ids != npz["val_ids"].tolist():
        raise ValueError(f"{path}: rebuilt dataset does not match the trial's subject IDs")
    sets = {
        "train": (data.x_train, est.final_beta_, est.final_assignments),
        "val": (data.x_val, npz["val_beta"], npz["val_subtype"]),
    }
    skip = {"X_obs", "dt", "cog", "id", "initial_beta"}
    rows = []
    for split, (patients, beta, subtype) in sets.items():
        for p, b, z in zip(patients, np.asarray(beta, float), np.asarray(subtype, int)):
            meta = {k: np.asarray(v)[0] for k, v in p.items()
                    if k not in skip and np.ndim(v) == 1 and len(v) == len(p["dt"])}
            rows.append({"id": str(p["id"]), "split": split, "beta": b, "subtype": z,
                         "n_visits": len(p["dt"]), **meta})
    subjects = pd.DataFrame(rows)
    if "cv_loglik" in npz:
        subjects.loc[subjects["split"] == "train", "cv_loglik"] = npz["cv_loglik"]
    return Trial(int(number), est, ExperimentConfig(**resolved), data, npz, subjects)


def trajectories(est: SubtypingEM, z_scale: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """``(t_span, traj)`` with traj (Z, T, B): model x, or the prediction c + s x on the z scale."""
    x = np.asarray(solve_trajectories_jax(est.cluster_f, est.final_kappa, est.final_scalar_K,
                                          est.K, est.t_span))
    if not z_scale:
        return est.t_span, x
    B = x.shape[2]
    return est.t_span, est._final_c_or_zero(B) + np.ravel(est.final_s)[None, None, :] * x


def visits(trial: Trial) -> pd.DataFrame:
    """Every observed value: id, split, subtype, disease time (beta + dt), biomarker, value."""
    names = trial.data.x_cols
    place = trial.subjects.set_index(["split", "id"])
    rows = []
    for split, patients in (("train", trial.data.x_train), ("val", trial.data.x_val)):
        for p in patients:
            s = place.loc[(split, str(p["id"]))]
            X = np.asarray(p["X_obs"], float)
            for j, dt in enumerate(np.asarray(p["dt"], float)):
                ok = np.isfinite(X[j])
                rows.extend({"id": str(p["id"]), "split": split, "subtype": int(s["subtype"]),
                             "time": float(s["beta"]) + dt, "biomarker": names[b], "value": X[j, b]}
                            for b in np.flatnonzero(ok))
    return pd.DataFrame(rows)


def visit_metadata(trial: Trial) -> pd.DataFrame:
    """One row per visit: id, split, subtype, beta, disease time and that visit's metadata
    (PPMI NSD_STAGE / NHY, ADNI PTAU / TAU / DX, ...) for validation against disease time."""
    skip = {"X_obs", "dt", "cog", "id", "initial_beta"}
    place = trial.subjects.set_index(["split", "id"])
    rows = []
    for split, patients in (("train", trial.data.x_train), ("val", trial.data.x_val)):
        for p in patients:
            s = place.loc[(split, str(p["id"]))]
            dt = np.asarray(p["dt"], float)
            meta = {k: np.asarray(v) for k, v in p.items()
                    if k not in skip and np.ndim(v) == 1 and len(v) == len(dt)}
            for j in range(len(dt)):
                rows.append({"id": str(p["id"]), "split": split, "subtype": int(s["subtype"]),
                             "beta": float(s["beta"]), "time": float(s["beta"]) + dt[j],
                             **{k: v[j] for k, v in meta.items()}})
    return pd.DataFrame(rows)
