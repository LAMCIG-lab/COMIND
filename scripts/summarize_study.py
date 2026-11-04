"""
Best trial (lowest CVIC by default) per (arm, n_subtypes) of an experiment, read-only.

    python scripts/summarize_study.py experiments/<dated> [--keys val_lse cv_cvic ...]

Writes <experiment>/summary.json (aggregate numbers only) and prints a table.
"""
import argparse
import glob
import json
import os

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.trial import TrialState

DEFAULT_KEYS = ("cv_cvic", "cv_lse", "val_lse", "val_lse_unweighted", "val_mse_imaging",
                "val_mse_clinical", "bic", "bic_gauss", "n_f_active", "s_median_all")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp_dir")
    ap.add_argument("--keys", nargs="+", default=list(DEFAULT_KEYS))
    ap.add_argument("--by", default="cv_cvic", help="trial attr defining 'best' ('objective' = study objective)")
    args = ap.parse_args()

    rows = []
    for journal in sorted(glob.glob(os.path.join(args.exp_dir, "*", "k*", "journal.log"))):
        storage = JournalStorage(JournalFileBackend(journal, lock_obj=JournalFileOpenLock(journal)))
        for name in optuna.get_all_study_names(storage=storage):
            study = optuna.load_study(study_name=name, storage=storage)
            done = [t for t in study.trials if t.state == TrialState.COMPLETE]
            arm_dir = os.path.relpath(os.path.dirname(journal), args.exp_dir)
            row = {"study": arm_dir, "n_complete": len(done),
                   "n_failed": sum(t.state == TrialState.FAIL for t in study.trials)}
            if args.by != "objective":
                done = [t for t in done if t.user_attrs.get(args.by) is not None]
            if done:
                best = min(done, key=lambda t: t.value if args.by == "objective" else t.user_attrs[args.by])
                row.update(best_trial=best.number, value=best.value, params=best.params,
                           **{k: best.user_attrs.get(k) for k in args.keys})
            rows.append(row)
    with open(os.path.join(args.exp_dir, "summary.json"), "w") as f:
        json.dump(rows, f, indent=2)
    cols = ["study", "n_complete", "n_failed", "best_trial", "value"] + list(args.keys)
    print("\t".join(cols))
    for r in rows:
        print("\t".join("" if r.get(c) is None else (f"{r[c]:.5g}" if isinstance(r[c], float) else str(r[c]))
                        for c in cols))


if __name__ == "__main__":
    main()
