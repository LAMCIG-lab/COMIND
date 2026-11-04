"""
Fresh-fold CVIC for each Z's best trial (see COMIND_transformer/experiment/rescore.py).

    python scripts/rescore_cvic.py experiments/<dated> --arm W0 --n-subtypes 1 2 3 4 5   # qsub one array per Z
    python scripts/rescore_cvic.py experiments/<dated> --arm W0 --n-subtypes 3 --fold 0  # one fold, here
    python scripts/select_z.py experiments/<dated> --arms W0 --rescored                  # when all are done

Run it after the studies finish: the best trial must not change afterwards.
"""
import argparse
import os
import subprocess

from COMIND_transformer.experiment.config import load_config
from COMIND_transformer.experiment.rescore import rescore_fold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp_dir")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--n-subtypes", type=int, nargs="+", required=True)
    ap.add_argument("--fold", type=int, help="run this fold here instead of submitting")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    exp_dir = os.path.abspath(args.exp_dir)
    if args.fold is not None:
        for z in args.n_subtypes:
            print(rescore_fold(os.path.join(exp_dir, args.arm, f"k{z}"), args.fold), flush=True)
        return
    n_folds = load_config(os.path.join(exp_dir, "config.yaml"), arm=args.arm).cv_folds
    data_root = f",COMIND_DATA={os.environ['COMIND_DATA']}" if os.environ.get("COMIND_DATA") else ""
    for z in args.n_subtypes:
        logs = os.path.join(exp_dir, args.arm, f"k{z}", "rescore", "logs")
        cmd = ["qsub", "-N", f"rescore_{args.arm}_k{z}", "-o", logs + "/", "-t", f"0-{n_folds - 1}",
               "-v", f"EXP_DIR={exp_dir},ARM={args.arm},NSUB={z},COMIND_ROOT={ROOT}{data_root}",
               os.path.join(ROOT, "scripts", "rescore.pbs")]
        if args.dry_run:
            print(" ".join(cmd))
            continue
        os.makedirs(logs, exist_ok=True)
        print(f"k{z}", subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip())


if __name__ == "__main__":
    main()
