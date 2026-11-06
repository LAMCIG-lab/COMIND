"""
Export one trial's model curves and parameters for scripts/plot_forcing_brains.py (model output only, no
subject data).

    python scripts/export_brain_data.py experiments/<dated> --arm W0            # Z and trial from z_selection.json
    python scripts/export_brain_data.py experiments/<dated> --arm W50 --n-subtypes 3 [--trial 41]
    python scripts/export_brain_data.py --study experiments/<dated>/W0/k4 --trial 27

Default Z = the arm's z_selection.json choice; default trial = that Z's chosen trial (lowest CVIC). Needs
$COMIND_DATA, since the trial's dataset is rebuilt from the study manifest. Writes to
<experiment>/brain_maps/<arm>_Z<Z>_trial<T>/ (or --out):

    forcing_all.csv                       f, every biomarker x subtype
    comind_cortical_forcing_subtype<k>_brainutils.csv   cortical f, region = L_/R_<aparc name>, FX_size = f
    trajectories_x.csv, trajectories_z.csv              x(t) in [0, 1] and the fitted-scale curve, per subtype
    parameters.csv                        s, kappa, f per biomarker
    model_info.json                       Z, trial, t_max, train subjects per subtype, ...

Then:  python scripts/plot_forcing_brains.py still|movie <out dir>
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_forcing_brains import parse_name  # noqa: E402

from COMIND_transformer.experiment import results as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp_dir", nargs="?", help="experiment folder (with config.yaml and <arm>/k<Z>/)")
    ap.add_argument("--arm", default="W0")
    ap.add_argument("--n-subtypes", type=int, help="default: selected_n_subtypes in <arm>/z_selection.json")
    ap.add_argument("--study", help="a study folder (<exp>/<arm>/k<Z>) instead of exp_dir / --arm / --n-subtypes")
    ap.add_argument("--trial", type=int, help="default: the chosen trial in z_selection.json, else lowest cv_cvic")
    ap.add_argument("--out")
    a = ap.parse_args()

    if a.study:
        study = a.study.rstrip("/")
        arm_dir = os.path.dirname(study)
        exp_dir, arm = os.path.dirname(arm_dir), os.path.basename(arm_dir)
    elif a.exp_dir:
        exp_dir, arm = a.exp_dir.rstrip("/"), a.arm
        arm_dir = os.path.join(exp_dir, arm)
    else:
        ap.error("give exp_dir or --study")
    sel_path = os.path.join(arm_dir, "z_selection.json")
    sel = json.load(open(sel_path)) if os.path.exists(sel_path) else {}
    if a.study:
        Z = int(os.path.basename(study).lstrip("k"))
    else:
        Z = a.n_subtypes or sel.get("selected_n_subtypes")
        if Z is None:
            ap.error(f"no {sel_path}; pass --n-subtypes")
        Z = int(Z)
        study = os.path.join(arm_dir, f"k{Z}")
    trial = a.trial
    if trial is None:
        chosen = sel.get("best_trials", {}).get(str(Z))
        trial = int(chosen["trial"]) if chosen else int(R.trials_table(study)["cv_cvic"].idxmin())

    t = R.load_trial(study, trial)
    est, names = t.est, list(t.data.x_cols)
    out = a.out or os.path.join(exp_dir, "brain_maps", f"{arm}_Z{Z}_trial{trial}")
    os.makedirs(out, exist_ok=True)

    F = np.stack([np.ravel(f) for f in est.cluster_f]).astype(float)
    cols = [f"subtype {k + 1}" for k in range(Z)]
    pd.DataFrame(F.T, index=pd.Index(names, name="biomarker"), columns=cols).to_csv(
        os.path.join(out, "forcing_all.csv"), float_format="%.6g")
    ctx = [(b, f"{p[1]}_{p[2]}") for b, n in enumerate(names) if (p := parse_name(n)) and p[0] == "ctx"]
    for k in range(Z):
        pd.DataFrame({"region": [r for _, r in ctx], "FX_size": [F[k, b] for b, _ in ctx]}).to_csv(
            os.path.join(out, f"comind_cortical_forcing_subtype{k + 1}_brainutils.csv"), index=False)

    t_span, x = R.trajectories(est)
    _, zt = R.trajectories(est, z_scale=True)
    for arr, tag in ((x, "x"), (zt, "z")):
        df = pd.concat([pd.DataFrame(arr[k], columns=names).assign(subtype=k + 1, time=t_span) for k in range(Z)])
        df[["subtype", "time"] + names].to_csv(os.path.join(out, f"trajectories_{tag}.csv"), index=False,
                                               float_format="%.6g")

    s = getattr(est, "final_s_clip", None)
    s = np.ravel(est.final_s if s is None else s).astype(float)
    clin = set(t.data.groups.get("clinical", []))
    pd.DataFrame({"biomarker": names, "group": ["clinical" if b in clin else "imaging" for b in range(len(names))],
                  "s": s, "kappa": np.ravel(est.final_kappa).astype(float),
                  **{f"f_subtype{k + 1}": F[k] for k in range(Z)}}).to_csv(
        os.path.join(out, "parameters.csv"), index=False, float_format="%.6g")

    sub = t.subjects
    info = {"experiment": os.path.basename(os.path.abspath(exp_dir)), "arm": arm, "n_subtypes": Z, "trial": trial,
            "t_max": float(est.t_max), "scalar_K": float(est.final_scalar_K),
            "n_train_per_subtype": [int(((sub.split == "train") & (sub.subtype == k)).sum()) for k in range(Z)]}
    json.dump(info, open(os.path.join(out, "model_info.json"), "w"), indent=2)
    unplaced = [n for n in names if parse_name(n) is None and n not in {names[b] for b in clin}]
    print(f"wrote {out}: Z={Z}, trial {trial}, {len(ctx)} cortical regions on the surface"
          + (f"; not shown on the brain (name not recognised): {unplaced}" if unplaced else ""))


if __name__ == "__main__":
    main()
