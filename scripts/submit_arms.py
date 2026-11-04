"""
qsub one worker array per (arm, n_subtypes) of an experiment config.

    python scripts/submit_arms.py experiments/<dated>/config.yaml --n-subtypes 3 4 --workers 16
    python scripts/submit_arms.py <config> --n-subtypes 1 2 3 4 5 --workers 6 12 16 20 24
    python scripts/submit_arms.py <config> --n-subtypes 4 --arms S1 S2 --dry-run   # print qsub lines

Fit time grows about linearly with Z (Z=5 is ~8x Z=1), so one worker count per Z
lets every study finish at about the same time. Arms are submitted in the order
given; with more jobs than free cores, Torque starts them first-come first-served
and a worker exits once its study has n_trials, freeing its core for the next.

Job IDs are appended to <experiment>/jobs.txt. Run a smoke trial first
(run_experiment.py --smoke) so a broken config fails in minutes, not on the queue.
"""
import argparse
import datetime as dt
import os
import subprocess

from COMIND_transformer.experiment.config import arm_names, load_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--n-subtypes", type=int, nargs="+", required=True)
    ap.add_argument("--arms", nargs="+", help="default: every arm in the config")
    ap.add_argument("--workers", type=int, nargs="+", default=[16],
                    help="array size per (arm, Z): one number, or one per --n-subtypes value")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    config = os.path.abspath(args.config)
    exp_dir = os.path.dirname(config)
    arms = args.arms or arm_names(config)
    workers = args.workers * len(args.n_subtypes) if len(args.workers) == 1 else args.workers
    if len(workers) != len(args.n_subtypes):
        ap.error("--workers needs one value, or one per --n-subtypes value")
    # Jobs do not source the submitting shell, so forward the data root the configs expand.
    data_root = f",COMIND_DATA={os.environ['COMIND_DATA']}" if os.environ.get("COMIND_DATA") else ""
    for arm in arms:
        for z, n_workers in zip(args.n_subtypes, workers):
            env = f"CONFIG={config},ARM={arm},NSUB={z},COMIND_ROOT={ROOT}{data_root}"
            load_config(config, arm=None if arm == "main" else arm, n_subtypes=z)  # validate now
            logs = os.path.join(exp_dir, arm, f"k{z}", "logs")
            cmd = ["qsub", "-N", f"{os.path.basename(exp_dir)[:8]}_{arm}_k{z}", "-o", logs + "/",
                   "-t", f"0-{n_workers - 1}",
                   "-v", env,
                   os.path.join(ROOT, "scripts", "submit.pbs")]
            if args.dry_run:
                print(" ".join(cmd))
                continue
            os.makedirs(logs, exist_ok=True)
            job = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
            line = f"{dt.datetime.now():%Y-%m-%d %H:%M} {arm} k{z} {job}"
            print(line)
            with open(os.path.join(exp_dir, "jobs.txt"), "a") as f:
                f.write(line + "\n")


if __name__ == "__main__":
    main()
