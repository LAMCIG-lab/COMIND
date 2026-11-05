"""Build analyze_results[_W50].ipynb for a zsym + CVIC experiment folder (+ an optional test script).

    python scripts/build_analysis_notebook.py ADNI|PPMI W0|W50 <experiment folder name> [--out nb.ipynb]
    python scripts/build_analysis_notebook.py PPMI W0 2026_11_04_PPMI_zsym_cvic --out /tmp/t.ipynb --test /tmp/t.py

The notebook reads the experiment's z_selection.json (fresh-fold CVIC, from rescore_cvic.py +
select_z.py) and loads the chosen trial through COMIND_transformer.experiment.results. ``--test``
also writes the code cells as a plain script that saves every figure next to it (headless check).
The dataset-specific validation sections exist for ADNI and PPMI; add a branch for a new dataset.
"""
import argparse
import json
import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("ds")
ap.add_argument("arm")
ap.add_argument("exp_name")
ap.add_argument("--out")
ap.add_argument("--test")
args = ap.parse_args()
DS, ARM, EXP_NAME, TEST = args.ds, args.arm, args.exp_name, args.test
EXP = os.path.join(ROOT, "experiments", EXP_NAME)
COMIND_DATA = os.environ.get("COMIND_DATA", "/home/dsemchin/data")
OUT = args.out or (f"{EXP}/analyze_results.ipynb" if ARM == "W0" else f"{EXP}/analyze_results_{ARM}.ipynb")
CFG = yaml.safe_load(open(f"{EXP}/config.yaml"))
S_LO, S_HI = float(CFG["estimator"]["s_lower_value"]), float(CFG["estimator"]["s_upper"])
S_MID = 0.5 * (S_LO + S_HI)
N_TRIALS_TEXT = ("80 trials per Z, extended overnight to 200" if "s2to4" in EXP_NAME
                 else f"{int(CFG['n_trials'])} trials per Z")
N_CLIN = {"ADNI": 2, "PPMI": 3}[DS]
WEIGHTS = {
    "W0": "**Weights (W0):** every biomarker counts equally in the loss.",
    "W50": f"**Weights (W50):** the {N_CLIN} clinical scores together carry half of the loss, so each one weighs "
           f"{82 / N_CLIN:.1f}x an imaging region. CV LSE and val LSE below are weighted (the fitted loss); "
           "the unweighted val LSE is shown next to them. CVIC does not use the weights.",
}[ARM]
COHORT = {
    "ADNI": "CN A+, MCI A+ (and A? who became demented), AD A+ / A?; at least 2 MRI visits. "
            "CN A+ are in the main train / val split.",
    "PPMI": "PD only. Train = subjects with >= 2 MRI visits; val = exactly 1 MRI (cross-sectional imaging).",
}[DS]
INPUTS = {
    "ADNI": "imaging = flipped W-scores (|z| > 5 masked); MoCA / ADAS13 = placeholder z vs training first visits.",
    "PPMI": "imaging = flipped normative z (`*_Z_predict`, |z| > 5 masked); MoCA / TD / PIGD = placeholder z "
            "vs training first visits.",
}[DS]
CLINICAL = {"ADNI": '["MOCA", "ADAS13"]', "PPMI": '["MCATOT", "TD_score", "PIGD_score"]'}[DS]
RAW_LABEL = {"ADNI": "imaging W-scores", "PPMI": "imaging normative z"}[DS]

cells = []


def md(s):
    cells.append(("markdown", s.strip("\n")))


def code(s):
    cells.append(("code", s.strip("\n")))


md(f"""
# {DS} zsym + CVIC: results ({ARM}, s in [{S_LO:g}, {S_HI:g}])

Experiment `{EXP_NAME}` (see `config.yaml` / `NOTES.md` in this folder).

- **Model:** zsym. The ODE state x(t) is in [0, 1]; the prediction on the z scale is `s (2x - 1)`,
  with `s` the clipping level in [{S_LO:g}, {S_HI:g}], starting at {S_MID:g} (the only scale parameter; `c = -s`).
- **Inputs:** {INPUTS}
- {WEIGHTS}
- **Cohort:** {COHORT}
- **Search:** Optuna on CV LSE, {N_TRIALS_TEXT}, Z = 1..5. Each trial also records held-out CVIC.
- **Selection:** within each Z the trial with the lowest search-fold CVIC; that trial is re-scored on fresh
  folds, and Z is chosen by the ordered Tukey test on the fresh-fold CVIC (`z_selection.json`).

**Plots are in ODE space, x in [0, 1],** unless a section says "raw z scale". Model curves are x(t).
Observations are mapped with the inverse of the observation model, `x_obs = (z + s) / (2 s)`, and clipped to
[0, 1] (values beyond +-s sit on the plateaus). The fit itself uses the raw z; the clipping is only for display
(`CLIP_OBS`). The raw-z-scale sections repeat the spaghetti plots without any clipping or mapping.
""")

code(f'''
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.stats import chi2_contingency, kendalltau, spearmanr
from sklearn.metrics import adjusted_mutual_info_score

try:
    from IPython.display import display
except ImportError:
    display = print

os.environ.setdefault("COMIND_DATA", "{COMIND_DATA}")
from COMIND_transformer.experiment import results as R

EXP_DIR = "{EXP}"
ARM = "{ARM}"
N_SUBTYPES = None    # None = the Tukey choice below; set an int to inspect another Z
TRIAL = None         # None = the chosen (lowest search-CVIC) trial of that Z
CLIP_OBS = True      # clip observations to [0, 1] in ODE space
MAX_PATIENTS = 500   # spaghetti plots: patients sampled per split
TOP_N = 10           # regions in the "top trajectories" panels
CLINICAL = {CLINICAL}
S_MID = {S_MID:g}            # middle of the s bounds (the starting s)
rng = np.random.default_rng(75)
''')

md("## Choosing Z: ordered Tukey test on fresh-fold CVIC\n\n"
   "For each Z, the search picks its trial (lowest CVIC on the 3 search folds). That trial is then refit on a "
   "fresh 3-fold split never used for picking, and those fresh-fold scores are what is shown and tested here.")

code('''
path = os.path.join(EXP_DIR, ARM, "z_selection.json")
with open(path) as f:
    selection = json.load(f)
if not selection.get("rescored"):
    raise RuntimeError(f"{path} is not from fresh-fold scores; run scripts/rescore_cvic.py, then scripts/select_z.py")
zs = [int(z) for z in selection["candidates"]]
best = {int(z): v for z, v in selection["best_trials"].items()}
cvic = {int(z): v for z, v in selection["cvic"].items()}

rows = []
for z in zs:
    df_z = R.trials_table(os.path.join(EXP_DIR, ARM, f"k{z}"))
    b = best[z]
    r = df_z.loc[b["trial"]]
    rows.append({"Z": z, "trials": len(df_z), "chosen trial": b["trial"], "CVIC": cvic[z],
                 "cv_lse": r["cv_lse"], "val_lse": r["val_lse"],
                 **({"val_lse unweighted": r["val_lse_unweighted"]} if ARM != "W0" else {})})
z_table = pd.DataFrame(rows).set_index("Z")
display(z_table.round(4))
for t_ in selection["tests"]:
    verdict = "go larger" if t_["rejected"] else "stop"
    print(f"Z={t_['z']} vs larger: CVIC gap {t_['gap']:.1f} (SE {t_['se']:.1f}), "
          f"p={t_['p']:.4g} vs alpha {selection['alpha_per_test']:.4g} -> {verdict}")
print(f"Tukey choice: Z = {selection['selected_n_subtypes']}")

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(z_table.index, z_table["CVIC"], "o-")
ax.axvline(selection["selected_n_subtypes"], color="C3", linestyle="--", alpha=0.6, label="Tukey choice")
ax.set_xlabel("Z (number of subtypes)")
ax.set_ylabel("CVIC, fresh folds (lower is better)")
ax.set_xticks(zs)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
''')

code('''
Z = int(N_SUBTYPES or selection["selected_n_subtypes"])
STUDY = os.path.join(EXP_DIR, ARM, f"k{Z}")
trials = R.trials_table(STUDY)
if TRIAL is None:
    TRIAL = int(best[Z]["trial"]) if Z in best else int(trials["cv_cvic"].idxmin())
t = R.load_trial(STUDY, TRIAL)   # rebuilds the dataset from the study manifest (~20 s)
est = t.est
names = t.data.x_cols
subjects = t.subjects
img_idx = t.data.groups["imaging"]
clin_idx = t.data.groups["clinical"]
row = trials.loc[TRIAL]
print(f"Z={Z} trial {TRIAL}: CVIC (fresh folds)={cvic.get(Z, float('nan')):.1f}  cv_lse={row.cv_lse:.4f}  "
      f"val_lse={row.val_lse:.4f}")
print("penalties:", {k: f"{row[k]:.3g}" for k in ("lambda_f", "lambda_kappa", "lambda_scalar")})
print(f"train n={int((subjects.split == 'train').sum())}, val n={int((subjects.split == 'val').sum())}, "
      f"{len(names)} biomarkers")
''')

md("## Search progress\n\nThe search minimises CV LSE (the Optuna objective" + ("" if ARM == "W0" else ", weighted") + ").")

code('''
fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
order = trials.sort_index()
axes[0].scatter(order.index, order["cv_lse"], s=12, alpha=0.4, label="trials")
axes[0].plot(order.index, order["cv_lse"].cummin(), color="C3", lw=2, label="best so far")
axes[0].scatter([TRIAL], [row.cv_lse], color="C3", s=80, zorder=5, label=f"chosen #{TRIAL}")
axes[0].set(xlabel="Optuna trial", ylabel="CV LSE", title=f"CV LSE by trial (Z={Z})")
axes[0].legend(fontsize=8)

sc = axes[1].scatter(order["lambda_f"], order["cv_lse"], c=np.log10(order["lambda_kappa"]), s=14, cmap="viridis")
axes[1].set_xscale("log")
axes[1].set(xlabel="lambda_f", ylabel="CV LSE", title="CV LSE vs lambda_f")
plt.colorbar(sc, ax=axes[1], label="log10 lambda_kappa")

lse = np.asarray(est.lse_history, float).ravel()
lse = lse[np.isfinite(lse)]
axes[2].plot(np.arange(len(lse)), lse, marker="o", ms=3)
axes[2].set(xlabel="EM iteration", ylabel="training LSE", title=f"Trial {TRIAL}: EM ({est.stop_reason_})")
for ax in axes:
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
''')

md("## Fitted parameters")

code('''
F = np.stack([np.ravel(f) for f in est.cluster_f]).astype(float)
kappa = np.ravel(est.final_kappa).astype(float)
s_par = np.ravel(est.final_s_clip if getattr(est, "final_s_clip", None) is not None else est.final_s).astype(float)


def describe(a, name):
    a = np.asarray(a, float).ravel()
    q = np.percentile(a, [25, 50, 75, 95])
    return pd.Series({"n": a.size, "frac < 1e-8": np.mean(a < 1e-8), "frac >= 0.01": np.mean(a >= 0.01),
                      "mean": a.mean(), "min": a.min(), "p25": q[0], "median": q[1], "p75": q[2],
                      "p95": q[3], "max": a.max()}, name=name)


table = pd.DataFrame([describe(F[k], f"f subtype {k + 1}") for k in range(Z)]
                     + [describe(kappa, "kappa"), describe(s_par[img_idx], "s imaging"),
                        describe(s_par[clin_idx], "s clinical")]).T
print(f"scalar_K = {float(est.final_scalar_K):.4g}")
display(table.round(4))

fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
for k in range(Z):
    axes[0].hist(F[k], bins=30, alpha=0.55, edgecolor="white", label=f"subtype {k + 1}")
axes[0].set(xlabel="f (forcing)", ylabel="count", title="f by subtype")
axes[0].legend(fontsize=8)
axes[1].hist(kappa, bins=30, color="#55A868", edgecolor="white")
axes[1].set(xlabel="kappa", title="kappa")
axes[2].hist(s_par[img_idx], bins=30, color="#C44E52", edgecolor="white", label="imaging")
axes[2].scatter(s_par[clin_idx], np.zeros(len(clin_idx)), color="k", zorder=5, label="clinical")
axes[2].set(xlabel="s (clipping level, z units)", title="s")
axes[2].legend(fontsize=8)
for ax in axes:
    ax.grid(True, alpha=0.3)
plt.suptitle(f"Fitted parameters, trial {TRIAL}", y=1.02)
plt.tight_layout()
plt.show()
''')

md("## Patient disease time (beta)")

code('''
bins = np.linspace(0, est.t_max, 31)
fig, axes = plt.subplots(1, Z + 1, figsize=(4 * (Z + 1), 3.4), sharey=True)
axes = np.atleast_1d(axes)
groups = [("all", subjects)] + [(f"subtype {k + 1}", subjects[subjects.subtype == k]) for k in range(Z)]
for ax, (label, sub) in zip(axes, groups):
    for split, color in (("train", "#4C72B0"), ("val", "#DD8452")):
        b = sub.loc[sub.split == split, "beta"]
        ax.hist(b, bins=bins, density=True, alpha=0.6, color=color, edgecolor="white", label=f"{split} n={len(b)}")
    ax.set(title=label, xlabel="beta (years)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
axes[0].set_ylabel("density")
plt.tight_layout()
plt.show()
''')

md("## Fitted trajectories (ODE space, x in [0, 1])")

code('''
t_span, traj = R.trajectories(est)          # (Z, T, B): ODE state x(t) in [0, 1]
B = traj.shape[2]
colors = [plt.cm.tab10(k % 10) for k in range(Z)]


def clean(name):
    if name in CLINICAL:
        return name
    name = name.removesuffix("_wscore").removesuffix("_Z_predict")
    for side, pre in (("L", "ctx-lh-"), ("R", "ctx-rh-"), ("L", "lh_"), ("R", "rh_")):
        if name.startswith(pre):
            return f"{side} " + name[len(pre):].removesuffix("_z").replace("-", " ").replace("_", " ")
    return (name.removesuffix("_etivnorm_z").removesuffix("_z").replace("Left-", "L ")
            .replace("Right-", "R ").replace("-Proper", "").replace("-area", "").replace("-", " ").replace("_", " "))


labels = [clean(n) for n in names]
fig, axes = plt.subplots(Z, 1, figsize=(12, max(3.5, 3.0 * Z)), sharex=True, squeeze=False)
for k, ax in enumerate(axes[:, 0]):
    for b in img_idx:
        ax.plot(t_span, traj[k, :, b], lw=0.7, alpha=0.35)
    for b in clin_idx:
        ax.plot(t_span, traj[k, :, b], lw=2, color="k", ls="--")
    n_k = int(((subjects.split == "train") & (subjects.subtype == k)).sum())
    ax.set(title=f"Subtype {k + 1} (n train = {n_k}); dashed = clinical", ylabel="x(t)", ylim=(-0.02, 1.02))
    ax.grid(True, alpha=0.25)
axes[-1, 0].set_xlabel("disease time (years)")
plt.tight_layout()
plt.show()
''')

md("## Top regional trajectories\n\nImaging regions ranked by the largest pairwise L2 distance between subtype curves of x(t) "
   "(range of the curve for Z = 1).")

code('''
w = np.gradient(t_span)
if Z > 1:
    score = np.array([max(np.sqrt(np.sum((traj[i, :, b] - traj[j, :, b]) ** 2 * w))
                          for i in range(Z) for j in range(i + 1, Z)) for b in range(B)])
else:
    score = np.ptp(traj[0], axis=0)
top = [b for b in np.argsort(score)[::-1] if b in set(img_idx)][:TOP_N]
for rank, b in enumerate(top, 1):
    print(f"{rank:2d}. {labels[b]:30s} score={score[b]:.3f}  s={s_par[b]:.2f}")

n_cols, n_rows = 2, int(np.ceil(len(top) / 2))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2.6 * n_rows), sharex=True, squeeze=False)
for ax, b in zip(axes.ravel(), top):
    for k in range(Z):
        ax.plot(t_span, traj[k, :, b], color=colors[k], lw=2, label=f"subtype {k + 1}")
    ax.set(title=labels[b], ylabel="x(t)", ylim=(-0.02, 1.02))
    ax.grid(True, alpha=0.25)
for ax in axes.ravel()[len(top):]:
    ax.set_visible(False)
axes[0, 0].legend(fontsize=8)
plt.tight_layout()
plt.show()
''')

md("## Observed data on disease time (spaghetti, ODE space)\n\n"
   "Each line is one patient (a dot when it has one visit), placed at its fitted / inferred beta and coloured by "
   "subtype; observations are `x_obs = (z + s) / (2 s)`. Thick lines are the model x(t).")

code('''
V = R.visits(t, clip=CLIP_OBS)   # id, split, subtype, time = beta + dt, biomarker, value (z), x (ODE space)
raw = R.visits(t, clip=False) if CLIP_OBS else V
for split in ("train", "val"):
    r = raw[raw.split == split]
    print(f"{split}: {len(r)} observations; x < 0: {np.mean(r.x < 0):.1%}, x > 1: {np.mean(r.x > 1):.1%} "
          f"(beyond the -s / +s plateaus{', clipped' if CLIP_OBS else ''})")


def spaghetti(split, cols, title):
    sv = V[V.split == split]
    ids = sv.id.unique()
    if len(ids) > MAX_PATIENTS:
        ids = rng.choice(ids, MAX_PATIENTS, replace=False)
    sv = sv[sv.id.isin(set(ids))]
    n_cols, n_rows = 2, int(np.ceil(len(cols) / 2))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2.8 * n_rows), sharex=True, squeeze=False)
    for ax, b in zip(axes.ravel(), cols):
        for pid, g in sv[sv.biomarker == names[b]].groupby("id"):
            g = g.sort_values("time")
            ax.plot(g.time, g.x, color=colors[int(g.subtype.iloc[0])], lw=0.8, alpha=0.5,
                    marker="." if len(g) == 1 else None, ms=3)
        for k in range(Z):
            ax.plot(t_span, traj[k, :, b], color=colors[k], lw=2.8)
        ax.set(title=f"{labels[b]} (s={s_par[b]:.2f})", ylabel="x", ylim=(-0.03, 1.03))
        ax.grid(True, alpha=0.25)
    for ax in axes.ravel()[len(cols):]:
        ax.set_visible(False)
    for ax in axes[-1, :]:
        ax.set_xlabel("disease time = beta + follow-up (years)")
    axes[0, 0].legend(handles=[Line2D([0], [0], color=colors[k], lw=2, label=f"subtype {k + 1}") for k in range(Z)],
                      fontsize=7)
    fig.suptitle(f"{title} (n = {len(ids)} patients)", y=1.005)
    plt.tight_layout()
    plt.show()


spaghetti("val", top, "Holdout: top regions")
spaghetti("train", top, "Train: top regions")
''')

code('''
clin_cols = [names.index(c) for c in CLINICAL]
spaghetti("val", clin_cols, "Holdout: clinical (placeholder z, ODE space)")
spaghetti("train", clin_cols, "Train: clinical (placeholder z, ODE space)")
''')

md("### Regions by fitted s (ODE space)")

code('''
SPLIT = "both"   # "train", "val", or "both"

img = np.asarray(img_idx)
cols = img[np.argsort(s_par[img])[::-1]][:10]
print("largest s")
for rank, b in enumerate(cols, 1):
    print(f"{rank:2d}. {labels[b]:30s} s={s_par[b]:.2f}")

splits = ("train", "val") if SPLIT == "both" else (SPLIT,)
if set(splits) - {"train", "val"}:
    raise ValueError("SPLIT must be 'train', 'val', or 'both'")
for split in splits:
    spaghetti(split, cols, f"{split}: largest s")
''')

code('''
SPLIT = "both"   # "train", "val", or "both"

img = np.asarray(img_idx)
cols = img[np.argsort(np.abs(s_par[img] - S_MID))][:10]
print(f"closest to s = {S_MID:g}")
for rank, b in enumerate(cols, 1):
    print(f"{rank:2d}. {labels[b]:30s} s={s_par[b]:.2f}  |s-{S_MID:g}|={abs(s_par[b] - S_MID):.3f}")

splits = ("train", "val") if SPLIT == "both" else (SPLIT,)
if set(splits) - {"train", "val"}:
    raise ValueError("SPLIT must be 'train', 'val', or 'both'")
for split in splits:
    spaghetti(split, cols, f"{split}: s closest to {S_MID:g}")
''')

md("## Observed data on the raw z scale (no clipping)\n\n"
   f"Observations are the z values the model is fitted on ({RAW_LABEL} and clinical placeholder z, sign-flipped "
   "so worse is positive); curves are the model prediction `s (2 x(t) - 1)`; dotted lines are its plateaus at "
   "+-s; the grey line is 0. Each title gives the share of observations beyond +-s, which is the share clipped "
   "in the ODE-space plots.")

code('''
SHOW_ORIGINAL_SIGN = False   # True: plot the original sign (imaging atrophy negative); curves are flipped to match

_sign = -1.0 if SHOW_ORIGINAL_SIGN else 1.0
_t_span, _z_traj = R.trajectories(t.est, z_scale=True)    # (Z, T, B)
_V_raw = R.visits(t, clip=False)                          # "value" = z as fitted, never clipped
_rng_raw = np.random.default_rng(75)


def spaghetti_raw(split, cols, title, max_patients=MAX_PATIENTS):
    sv = _V_raw[_V_raw.split == split]
    ids = sv.id.unique()
    if len(ids) > max_patients:
        ids = _rng_raw.choice(ids, max_patients, replace=False)
    sv = sv[sv.id.isin(set(ids))]
    n_rows = int(np.ceil(len(cols) / 2))
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 2.8 * n_rows), sharex=True, squeeze=False)
    for ax, b in zip(axes.ravel(), cols):
        obs = sv[sv.biomarker == names[b]]
        for pid, g in obs.groupby("id"):
            g = g.sort_values("time")
            ax.plot(g.time, _sign * g.value, color=colors[int(g.subtype.iloc[0])], lw=0.8, alpha=0.5,
                    marker="." if len(g) == 1 else None, ms=3)
        for k in range(Z):
            ax.plot(_t_span, _sign * _z_traj[k, :, b], color=colors[k], lw=2.8)
        for level in (-s_par[b], s_par[b]):
            ax.axhline(_sign * level, color="k", ls=":", lw=1)
        ax.axhline(0, color="0.6", lw=0.8)
        beyond = np.mean(np.abs(obs.value) > s_par[b]) if len(obs) else np.nan
        ax.set_title(f"{labels[b]}  (s={s_par[b]:.2f}, {beyond:.0%} of obs beyond +/-s)", fontsize=10)
        ax.set_ylabel("z (original sign)" if SHOW_ORIGINAL_SIGN else "z (worse positive)")
        ax.grid(True, alpha=0.25)
    for ax in axes.ravel()[len(cols):]:
        ax.set_visible(False)
    for ax in axes[-1, :]:
        ax.set_xlabel("disease time = beta + follow-up (years)")
    handles = [Line2D([0], [0], color=colors[k], lw=2, label=f"subtype {k + 1}") for k in range(Z)]
    handles.append(Line2D([0], [0], color="k", ls=":", lw=1, label="model plateau (+/- s)"))
    axes[0, 0].legend(handles=handles, fontsize=7)
    fig.suptitle(f"{title}: raw z scale, unclipped (n = {len(ids)} patients)", y=1.005)
    plt.tight_layout()
    plt.show()


spaghetti_raw("val", top, "Holdout: top regions")
spaghetti_raw("train", top, "Train: top regions")
''')

code('''
spaghetti_raw("val", clin_cols, "Holdout: clinical (placeholder z)")
spaghetti_raw("train", clin_cols, "Train: clinical (placeholder z)")
''')

code('''
SPLIT = "both"   # "train", "val", or "both"

img = np.asarray(img_idx)
cols = img[np.argsort(s_par[img])[::-1]][:10]
splits = ("train", "val") if SPLIT == "both" else (SPLIT,)
for split in splits:
    spaghetti_raw(split, cols, f"{split}: largest s")
''')

code('''
SPLIT = "both"   # "train", "val", or "both"

img = np.asarray(img_idx)
cols = img[np.argsort(np.abs(s_par[img] - S_MID))][:10]
splits = ("train", "val") if SPLIT == "both" else (SPLIT,)
for split in splits:
    spaghetti_raw(split, cols, f"{split}: s closest to {S_MID:g}")
''')

if DS == "ADNI":
    md("## CSF pTau vs disease time\n\nPer visit: CSF pTau against disease time (beta + follow-up). Per subject: "
       "baseline pTau (`PTAU_bl`) against beta. Values like `<8` are read as 8.")
    code('''
VM = R.visit_metadata(t)


def csf(x):
    return pd.to_numeric(pd.Series(x).astype(str).str.strip().str.lstrip("<>"), errors="coerce").to_numpy()


VM["PTAU_num"] = csf(VM["PTAU"])
base = VM.sort_values("time").groupby(["split", "id"]).first().reset_index()
base["PTAU_bl_num"] = csf(base["PTAU_bl"])

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for i, split in enumerate(("train", "val")):
    v = VM[(VM.split == split) & np.isfinite(VM.PTAU_num)]
    b = base[(base.split == split) & np.isfinite(base.PTAU_bl_num)]
    for k in range(Z):
        axes[i, 0].scatter(v.time[v.subtype == k], v.PTAU_num[v.subtype == k], s=10, alpha=0.5, color=colors[k],
                           label=f"subtype {k + 1}")
        axes[i, 1].scatter(b.beta[b.subtype == k], b.PTAU_bl_num[b.subtype == k], s=12, alpha=0.6, color=colors[k])
    rv = spearmanr(v.time, v.PTAU_num) if len(v) > 2 else None
    rb = spearmanr(b.beta, b.PTAU_bl_num) if len(b) > 2 else None
    axes[i, 0].set(title=f"{split}: pTau per visit vs disease time" + (f"  rho={rv.correlation:.2f} (n={len(v)})" if rv else ""),
                   xlabel="disease time (years)", ylabel="CSF pTau")
    axes[i, 1].set(title=f"{split}: baseline pTau vs beta" + (f"  rho={rb.correlation:.2f} (n={len(b)})" if rb else ""),
                   xlabel="beta (years)", ylabel="CSF pTau (baseline)")
axes[0, 0].legend(fontsize=8)
for ax in axes.ravel():
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

rows = []
for split in ("train", "val"):
    for k in list(range(Z)) + [None]:
        v = VM[(VM.split == split) & np.isfinite(VM.PTAU_num) & ((VM.subtype == k) if k is not None else True)]
        if len(v) > 2:
            rows.append({"split": split, "subtype": "all" if k is None else k + 1, "n visits": len(v),
                         "spearman(time, pTau)": spearmanr(v.time, v.PTAU_num).correlation})
display(pd.DataFrame(rows).round(3))
''')
    md("## Subtype associations\n\nLatest-visit DX, baseline DX, APOE4 and amyloid label against subtype, for train and "
       "holdout. External checks only: none of these were used to fit or select.")
    code('''
first_valid = lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan
per = VM.sort_values("time").groupby(["split", "id"]).agg(
    subtype=("subtype", "first"), latest_dx=("DX", lambda s: s.dropna().iloc[-1] if s.notna().any() else np.nan),
    dx_bl=("DX_bl", first_valid), APOE4=("APOE4", first_valid), amyloid=("amyloid", first_valid)).reset_index()
per["APOE4"] = pd.to_numeric(per["APOE4"], errors="coerce")
sub_names = [f"subtype {k + 1}" for k in range(Z)]
per["subtype_name"] = per.subtype.map(lambda k: f"subtype {k + 1}")


def heat(ct, title, ax, pct=False):
    sns.heatmap(ct, ax=ax, annot=True, fmt=".1f" if pct else "d", cmap="YlOrRd" if pct else "Blues",
                vmin=0 if pct else None, vmax=100 if pct else None, linewidths=0.5, linecolor="white")
    ax.set_title(title, fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)


def chi2_text(ct):
    ct = ct.loc[ct.sum(axis=1) > 0, ct.sum(axis=0) > 0]
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return "n/a"
    chi2, pval, dof, _ = chi2_contingency(ct)
    return f"chi2={chi2:.1f}, dof={dof}, p={pval:.3g}"


for split in ("train", "val"):
    p = per[per.split == split]
    cts = {
        "latest DX": pd.crosstab(p.latest_dx, p.subtype_name).reindex(index=["CN", "MCI", "Dementia"], columns=sub_names, fill_value=0),
        "DX_bl": pd.crosstab(p.dx_bl, p.subtype_name).reindex(columns=sub_names, fill_value=0),
        "APOE4 alleles": pd.crosstab(p.APOE4.dropna().astype(int), p.loc[p.APOE4.notna(), "subtype_name"]).reindex(columns=sub_names, fill_value=0),
        "amyloid": pd.crosstab(p.amyloid, p.subtype_name).reindex(columns=sub_names, fill_value=0),
    }
    for name, ct in cts.items():
        print(f"{split:5s} {name:13s} {chi2_text(ct)}")
    dxm = p.dropna(subset=["latest_dx"])
    if Z > 1 and len(dxm):
        print(f"{split:5s} AMI(subtype, latest DX) = {adjusted_mutual_info_score(dxm.latest_dx, dxm.subtype):.4f}")
    fig, axes = plt.subplots(1, 5, figsize=(22, 3.6))
    heat(cts["latest DX"], f"{split}: latest DX (counts)", axes[0])
    heat(cts["DX_bl"], f"{split}: DX_bl (counts)", axes[1])
    heat(cts["APOE4 alleles"], f"{split}: APOE4 (counts)", axes[2])
    apoe_pct = (100 * cts["APOE4 alleles"].div(cts["APOE4 alleles"].sum(axis=1).replace(0, np.nan), axis=0)).fillna(0)
    heat(apoe_pct, f"{split}: APOE4 (row %)", axes[3], pct=True)
    heat(cts["amyloid"], f"{split}: amyloid (counts)", axes[4])
    plt.tight_layout()
    plt.show()
''')
    code('''
SPLIT = "both"   # "train", "val", or "both"

p = per.dropna(subset=["APOE4"]).copy()
p["APOE"] = np.where(p["APOE4"] >= 1, "1+", "0")
splits = ("train", "val") if SPLIT == "both" else (SPLIT,)
if set(splits) - {"train", "val"}:
    raise ValueError("SPLIT must be 'train', 'val', or 'both'")

fig, axes = plt.subplots(len(splits), 2, figsize=(9, 3.2 * len(splits)), squeeze=False)
for i, split in enumerate(splits):
    g = p[p.split == split]
    ct = pd.crosstab(g.APOE, g.subtype_name).reindex(index=["0", "1+"], columns=sub_names, fill_value=0)
    pct = (100 * ct.div(ct.sum(axis=0).replace(0, np.nan), axis=1)).fillna(0)
    print(f"{split:5s} APOE 0 vs 1+  {chi2_text(ct)}  n={len(g)}")
    heat(ct, f"{split}: APOE (counts)", axes[i, 0])
    heat(pct, f"{split}: APOE (% within subtype)", axes[i, 1], pct=True)
plt.tight_layout()
plt.show()
''')
    code('''
SPLIT = "both"   # "train", "val", or "both"

dx_order = ["CN", "SMC", "EMCI", "LMCI", "MCI", "AD", "Dementia"]
d = subjects.dropna(subset=["DX_bl", "beta"]).copy()
d["DX_bl"] = d["DX_bl"].astype(str)
splits = ("train", "val") if SPLIT == "both" else (SPLIT,)
if set(splits) - {"train", "val"}:
    raise ValueError("SPLIT must be 'train', 'val', or 'both'")

fig, axes = plt.subplots(1, len(splits), figsize=(5.5 * len(splits), 4.2), sharey=True, squeeze=False)
for ax, split in zip(axes[0], splits):
    g = d[d.split == split]
    present = [c for c in dx_order if c in set(g.DX_bl)]
    present += [c for c in sorted(g.DX_bl.unique()) if c not in present]
    sns.violinplot(data=g, x="DX_bl", y="beta", order=present, ax=ax, inner="box", cut=0, color="#4C72B0")
    sns.stripplot(data=g, x="DX_bl", y="beta", order=present, ax=ax, color="k", size=2, alpha=0.25, jitter=0.25)
    counts = g.DX_bl.value_counts()
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([f"{c}\\n(n={int(counts.get(c, 0))})" for c in present])
    ax.set(title=split, xlabel="DX_bl", ylabel="beta (years)")
    ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

dx_rank = {"CN": 0, "SMC": 1, "EMCI": 2, "LMCI": 3, "MCI": 4, "AD": 5, "Dementia": 6}

for split in splits:
    g = d[d.split == split]
    x = g["DX_bl"].map(dx_rank)
    ok = x.notna() & np.isfinite(g["beta"])
    n_drop = int((~ok).sum())
    x, y = x[ok].to_numpy(float), g.loc[ok, "beta"].to_numpy(float)
    if np.unique(x).size < 2:
        print(f"{split}: not enough DX_bl levels")
        continue
    kt, sp = kendalltau(x, y), spearmanr(x, y)
    extra = f", dropped {n_drop} unlabeled" if n_drop else ""
    print(f"{split}: n={len(x)}{extra}  Kendall tau-b={kt.correlation:.3f} (p={kt.pvalue:.3g})  "
          f"Spearman rho={sp.correlation:.3f} (p={sp.pvalue:.3g})")
''')
else:
    md("## Staging: NSD and Hoehn & Yahr vs disease time\n\n"
       "Train: disease time at the final visit (beta + follow-up) against the stage at that visit. "
       "Val (one MRI): disease time at the first visit against the stage there. "
       "NSD: `Not NSD` = 0, 2a / 2b = 2, 5 / 6 merged into 4+. H&Y: 0 dropped, 4 / 5 merged into 4+. "
       "Kendall tau on the ordinal stage.")
    code('''
VM = R.visit_metadata(t)


def nsd_value(v):
    s = str(v).strip().lower()
    if s in ("nan", "", "none"):
        return np.nan
    if s == "not nsd":
        return 0
    if s in ("2a", "2b"):
        return 2
    try:
        n = int(float(s))
    except ValueError:
        return np.nan
    return min(n, 4) if n >= 0 else np.nan


def nhy_value(v):
    x = pd.to_numeric(v, errors="coerce")
    return np.nan if not np.isfinite(x) or int(round(x)) == 0 else min(int(round(x)), 4)


VM["NSD"] = [nsd_value(v) for v in VM["NSD_STAGE"]]
VM["NHY_ord"] = [nhy_value(v) for v in VM["NHY"]]
VM = VM.sort_values("time")
stage_rows = pd.concat([
    VM[VM.split == "train"].groupby("id").tail(1),   # final visit
    VM[VM.split == "val"].groupby("id").head(1),     # first visit
])
label = {"NSD": lambda n: "Not NSD" if n == 0 else ("4+" if n >= 4 else str(int(n))),
         "NHY_ord": lambda n: "4+" if n >= 4 else str(int(n))}
axis_order = {"NSD": ["Not NSD", "2", "3", "4+"], "NHY_ord": ["1", "2", "3", "4+"]}

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
tau_rows = []
for i, stage in enumerate(("NSD", "NHY_ord")):
    for j, split in enumerate(("train", "val")):
        d = stage_rows[(stage_rows.split == split)].dropna(subset=[stage]).copy()
        d["cat"] = d[stage].map(label[stage])
        order_ = [c for c in axis_order[stage] if c in set(d.cat)]
        ax = axes[i, j]
        if len(d) > 1:
            sns.violinplot(data=d, x="cat", y="time", order=order_, ax=ax, cut=0, inner="box", color="0.85")
            for k in range(Z):
                dk = d[d.subtype == k]
                xs = np.array([order_.index(c) for c in dk.cat]) + rng.uniform(-0.15, 0.15, len(dk))
                ax.scatter(xs, dk.time, s=10, alpha=0.7, color=colors[k], label=f"subtype {k + 1}", zorder=3)
            tau, pv = kendalltau(d[stage], d.time)
        else:
            tau, pv = np.nan, np.nan
        name = "NSD" if stage == "NSD" else "H&Y"
        ax.set(title=f"{split}: {name} vs disease time  tau={tau:.3f}, p={pv:.3g} (n={len(d)})",
               xlabel=f"{name} ({'final' if split == 'train' else 'first'} visit)", ylabel="disease time (years)")
        ax.grid(True, alpha=0.3)
        for k in [None] + list(range(Z)):
            dk = d if k is None else d[d.subtype == k]
            if len(dk) > 2 and dk[stage].nunique() > 1:
                tk, pk = kendalltau(dk[stage], dk.time)
                tau_rows.append({"stage": name, "split": split, "subtype": "all" if k is None else k + 1,
                                 "n": len(dk), "kendall tau": tk, "p": pk})
axes[0, 0].legend(fontsize=7)
plt.tight_layout()
plt.show()
display(pd.DataFrame(tau_rows).round(4))
''')
    md("## Subtype associations\n\nTD / PIGD class (Stebbins, from raw TD and PIGD item sums at the first visit) and "
       "genetic subgroup against subtype, for train and holdout. External checks only: none of these were used to "
       "fit or select.")
    code('''
def td_pigd(td_sum, pigd_sum):
    td, pigd = td_sum / 10.0, pigd_sum / 3.0
    ratio = np.inf if pigd == 0 else td / pigd
    if ratio >= 1.5 or (td > 0 and pigd == 0):
        return "TD"
    if ratio <= 1.0 or (td == 0 and pigd > 0):
        return "PIGD"
    return "Indeterminate"


place = subjects.set_index(["split", "id"])
rows = []
for split, patients in (("train", t.data.x_train), ("val", t.data.x_val)):
    for p in patients:
        cog = np.asarray(p["cog"], float)          # raw MCATOT, TD, PIGD per visit
        first = int(np.argmin(np.asarray(p["dt"], float)))
        sub = np.atleast_1d(np.asarray(p.get("subgroup"), dtype=object))
        sub = next((x for x in sub if isinstance(x, str)), np.nan)
        rows.append({"split": split, "id": str(p["id"]), "subtype": int(place.loc[(split, str(p["id"])), "subtype"]),
                     "TD/PIGD": td_pigd(cog[first, 1], cog[first, 2]) if np.isfinite(cog[first, 1:3]).all() else np.nan,
                     "genetic subgroup": sub})
per = pd.DataFrame(rows)
per["carrier"] = per["genetic subgroup"].map(lambda g: np.nan if pd.isna(g) else ("sporadic" if g == "Sporadic PD" else "carrier"))
sub_names = [f"subtype {k + 1}" for k in range(Z)]
per["subtype_name"] = per.subtype.map(lambda k: f"subtype {k + 1}")


def heat(ct, title, ax, pct=False):
    sns.heatmap(ct, ax=ax, annot=True, fmt=".1f" if pct else "d", cmap="YlOrRd" if pct else "Blues",
                vmin=0 if pct else None, vmax=100 if pct else None, linewidths=0.5, linecolor="white")
    ax.set_title(title, fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)


def chi2_text(ct):
    ct = ct.loc[ct.sum(axis=1) > 0, ct.sum(axis=0) > 0]
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return "n/a"
    chi2, pval, dof, _ = chi2_contingency(ct)
    return f"chi2={chi2:.1f}, dof={dof}, p={pval:.3g}"


for split in ("train", "val"):
    p = per[per.split == split]
    fig, axes = plt.subplots(1, 4, figsize=(20, 3.8))
    for ax_i, col in enumerate(("TD/PIGD", "genetic subgroup", "carrier")):
        d = p.dropna(subset=[col])
        ct = pd.crosstab(d[col], d.subtype_name).reindex(columns=sub_names, fill_value=0)
        ami = adjusted_mutual_info_score(d[col], d.subtype) if Z > 1 and len(d) else np.nan
        print(f"{split:5s} {col:16s} n={len(d):3d}  AMI={ami:.4f}  {chi2_text(ct)}")
        heat(ct, f"{split}: {col} (counts)", axes[ax_i])
        if col == "genetic subgroup":
            pct = (100 * ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0)).fillna(0)
            heat(pct, f"{split}: genetic subgroup (row %)", axes[3], pct=True)
    plt.tight_layout()
    plt.show()
''')


def to_lines(s):
    lines = s.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


nb = {
    "cells": [
        ({"cell_type": "markdown", "metadata": {}, "source": to_lines(s)} if kind == "markdown" else
         {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": to_lines(s)})
        for kind, s in cells
    ],
    "metadata": {
        "kernelspec": {"display_name": "comind", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell-{i:02d}"
with open(OUT, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", OUT)

if TEST:
    d = TEST.rsplit("/", 1)[0]
    with open(TEST, "w") as f:
        f.write("import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n_n = [0]\n"
                "def _show(*a, **k):\n    _n[0] += 1\n"
                f"    plt.savefig('{d}/{DS}_{ARM}_fig_%02d.png' % _n[0], dpi=60, bbox_inches='tight')\n"
                "    plt.close('all')\nplt.show = _show\n\n")
        for kind, s in cells:
            if kind == "code":
                f.write(s + "\n\n")
    print("wrote", TEST)
