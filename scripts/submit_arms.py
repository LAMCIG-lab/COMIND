"""
qsub one worker array per (arm, n_subtypes) of an experiment config.

    python scripts/submit_arms.py experiments/<dated>/config.yaml --n-subtypes 3 4 --workers 16
    python scripts/submit_arms.py <config> --n-subtypes 4 --arms S1 S2 --dry-run   # print qsub lines

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
    ap.add_argument("--workers", type=int, default=16, help="array size per (arm, Z)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    config = os.path.abspath(args.config)
    exp_dir = os.path.dirname(config)
    arms = args.arms or arm_names(config)
    for arm in arms:
        for z in args.n_subtypes:
            load_config(config, arm=None if arm == "main" else arm, n_subtypes=z)  # validate now
            logs = os.path.join(exp_dir, arm, f"k{z}", "logs")
            os.makedirs(logs, exist_ok=True)
            cmd = ["qsub", "-N", f"{os.path.basename(exp_dir)[:8]}_{arm}_k{z}", "-o", logs + "/",
                   "-t", f"0-{args.workers - 1}",
                   "-v", f"CONFIG={config},ARM={arm},NSUB={z},COMIND_ROOT={ROOT}",
                   os.path.join(ROOT, "scripts", "submit.pbs")]
            if args.dry_run:
                print(" ".join(cmd))
                continue
            job = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
            line = f"{dt.datetime.now():%Y-%m-%d %H:%M} {arm} k{z} {job}"
            print(line)
            with open(os.path.join(exp_dir, "jobs.txt"), "a") as f:
                f.write(line + "\n")


if __name__ == "__main__":
    main()
