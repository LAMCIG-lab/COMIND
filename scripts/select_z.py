"""
Pick n_subtypes per arm with the ordered Tukey test on held-out CVIC. Read-only.

    python scripts/select_z.py experiments/<dated> [--arms S1 S2] [--alpha 0.05]

Needs a finished study for each candidate Z (<arm>/k1, k2, ...), all run with
cvic: true. Writes <experiment>/<arm>/z_selection.json (aggregate numbers only).
"""
import argparse
import json
import os

from COMIND_transformer.experiment.config import arm_names
from COMIND_transformer.experiment.selection import select_z


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp_dir")
    ap.add_argument("--arms", nargs="+", help="default: every arm in config.yaml")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--no-bonferroni", action="store_true")
    ap.add_argument("--by", default="cv_cvic", help="trial attr picking each Z's best trial ('objective' = study objective)")
    args = ap.parse_args()

    for arm in args.arms or arm_names(os.path.join(args.exp_dir, "config.yaml")):
        res = select_z(args.exp_dir, arm, by=None if args.by == "objective" else args.by,
                       alpha=args.alpha, bonferroni=not args.no_bonferroni)
        with open(os.path.join(args.exp_dir, arm, "z_selection.json"), "w") as f:
            json.dump(res, f, indent=2)
        cvic = "  ".join(f"Z={z}:{v:.1f}" for z, v in res["cvic"].items())
        print(f"{arm}: selected Z={res['selected_n_subtypes']}  CVIC {cvic}")
        for t in res["tests"]:
            print(f"    Z={t['z']} vs larger: gap={t['gap']:.1f} se={t['se']:.1f} p={t['p']:.4f} "
                  f"{'-> go larger' if t['rejected'] else '-> stop'}  (alpha={res['alpha_per_test']:.4g})")


if __name__ == "__main__":
    main()
