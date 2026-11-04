"""
Choose n_subtypes for one arm: the ordered Tukey test on held-out CVIC.

Each ``<arm>/k<Z>`` study contributes its best trial: lowest ``cv_cvic`` by
default (``by=None`` uses the study objective instead).
That trial's npz holds ``cv_loglik``, the held-out log density of every
training subject from the CV fold that left it out. Stacking those columns
over Z gives the (n_subjects, n_Z) matrix that ``cvic.ordered_tukey_nb``
tests: starting from the smallest Z, keep going to a larger Z only while some
larger Z has credibly lower CVIC (Nadeau-Bengio-corrected SE, Bonferroni over
the n_Z - 1 tests). This is the protocol of ``select_n_subtypes``, except the
hyperparameters per Z come from the existing Optuna study instead of an inner
search inside every fold.
"""
from __future__ import annotations

import glob
import os
import re
from typing import Any, Dict, List, Optional

import numpy as np
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.trial import FrozenTrial, TrialState

from ..cvic import ordered_tukey_nb


def best_trial(study_dir: str, by: Optional[str] = "cv_cvic") -> FrozenTrial:
    """COMPLETE trial with the lowest user attr ``by`` (None = study objective). Read-only."""
    journal = os.path.join(study_dir, "journal.log")
    storage = JournalStorage(JournalFileBackend(journal, lock_obj=JournalFileOpenLock(journal)))
    (name,) = optuna.get_all_study_names(storage=storage)
    done = [t for t in optuna.load_study(study_name=name, storage=storage).trials
            if t.state == TrialState.COMPLETE]
    if by is not None:
        done = [t for t in done if t.user_attrs.get(by) is not None]
    if not done:
        raise ValueError(f"{study_dir}: no complete trials" + (f" with {by}" if by else ""))
    return min(done, key=lambda t: t.value if by is None else t.user_attrs[by])


def select_z(exp_dir: str, arm: str, *, by: Optional[str] = "cv_cvic", rescored: bool = False,
             alpha: float = 0.05, bonferroni: bool = True, n_draw: int = 200_000,
             seed: int = 0) -> Dict[str, Any]:
    """Run the ordered Tukey test over every ``<exp_dir>/<arm>/k<Z>`` study.

    ``rescored=True`` tests the fresh-fold scores of ``rescore.py`` (see there) instead of
    the in-study CV scores the best trial was chosen on.
    """
    from .rescore import collect
    dirs = glob.glob(os.path.join(exp_dir, arm, "k*", "journal.log"))
    zs = sorted(int(re.search(r"k(\d+)$", os.path.dirname(d)).group(1)) for d in dirs)
    if len(zs) < 2:
        raise ValueError(f"{exp_dir}/{arm}: need studies for at least 2 values of Z, found {zs}")

    cols: List[np.ndarray] = []
    trials: Dict[int, Dict[str, Any]] = {}
    ref = None
    for z in zs:
        out = os.path.join(exp_dir, arm, f"k{z}")
        t = best_trial(out, by=by)
        if rescored:
            loglik, fold, ids, number = collect(out)
            if number != t.number:
                raise ValueError(f"{out}: rescored trial {number}, but the best trial is now "
                                 f"{t.number}; rescore again")
            cols.append(loglik)
        else:
            with np.load(os.path.join(out, "trials", f"{t.number:04d}.npz")) as npz:
                if "cv_loglik" not in npz:
                    raise ValueError(f"{out} trial {t.number} has no cv_loglik (run with cvic: false?)")
                ids, fold = npz["train_ids"].tolist(), npz["cv_fold"]
                cols.append(np.asarray(npz["cv_loglik"], float))
        # Tukey compares Z subject by subject, so every Z must hold out the same subjects together.
        if ref is None:
            ref = (ids, fold)
        elif ids != ref[0] or not np.array_equal(fold, ref[1]):
            raise ValueError(f"{out}: training subjects or CV folds differ from k{zs[0]}")
        trials[z] = {"trial": t.number, "objective": t.value, "params": t.params,
                     "cv_cvic": t.user_attrs.get("cv_cvic"),
                     **({"rescored_cvic": float(-2.0 * cols[-1].sum())} if rescored else {})}

    n_folds = int(ref[1].max()) + 1
    idx, info = ordered_tukey_nb(np.column_stack(cols), n_folds=n_folds, alpha=alpha,
                                 n_draw=n_draw, seed=seed, bonferroni=bonferroni)
    # Test i compares Z = zs[i] with every larger candidate.
    tests = [{"z": zs[i], "gap": float(info["gaps"][i]), "se": float(info["se"][i]),
              "p": float(info["pvalues"][i]), "rejected": bool(info["rejected"][i])}
             for i in range(len(zs) - 1) if np.isfinite(info["pvalues"][i])]
    return {"arm": arm, "selected_n_subtypes": zs[idx - 1], "candidates": zs, "best_by": by or "objective", "rescored": rescored,
            "cvic": dict(zip(zs, info["cvic"].tolist())), "tests": tests,
            "alpha_per_test": info["alpha"], "bonferroni": bonferroni,
            "n_subjects": len(ref[0]), "n_folds": n_folds, "best_trials": trials}
