"""
One Optuna worker for an experiment config.

    python scripts/run_experiment.py experiments/<dated>/config.yaml --arm S1 --n-subtypes 4
    python scripts/run_experiment.py <config> --arm S1 --dry-run       # load data, print counts
    python scripts/run_experiment.py <config> --arm S1 --smoke         # 1 short trial in <dir>/_smoke
"""
import argparse
import json
import os

from COMIND_transformer.experiment.config import load_config
from COMIND_transformer.experiment.runner import build_estimator, load_dataset, run_worker


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--arm")
    ap.add_argument("--n-subtypes", type=int)
    ap.add_argument("--n-trials", type=int, help="override the config's study budget (e.g. to extend a study)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="one trial, max_iter 3, n_init 1, in _smoke/")
    args = ap.parse_args()

    overrides = {"n_subtypes": args.n_subtypes, "n_trials": args.n_trials}
    exp_dir = os.path.dirname(os.path.abspath(args.config))
    cfg = load_config(args.config, arm=args.arm, **overrides)
    if args.smoke:
        cfg.estimator.update(max_iter=3, min_iter=1)
        cfg.n_init, cfg.n_startup_trials = 1, 1
        exp_dir = os.path.join(exp_dir, "_smoke")
    print(json.dumps(cfg.to_dict(), indent=1, default=str), flush=True)

    if args.dry_run:
        data = load_dataset(cfg)
        build_estimator(cfg, data)
        print(json.dumps({"n_biomarkers": len(data.x_cols), "n_rows_train": data.n_rows_train,
                          **{k: v for k, v in data.info.items() if k != "clinical_z"}}, indent=1))
        return
    study = run_worker(cfg, exp_dir, config_path=args.config, n_trials=1 if args.smoke else None)
    done = [t for t in study.trials if t.value is not None]
    if done:
        best = min(done, key=lambda t: t.value)
        print(f"best trial {best.number}: {cfg.objective}={best.value:.6g}", flush=True)


if __name__ == "__main__":
    main()
