#!/usr/bin/env python
"""
NSD-ISS stage prediction from clinical scores and COMIND outputs.

Clean re-implementation of NSD_ablation.ipynb. Differences from the notebook:

  1. One selection metric for every model: macro-averaged ranked probability
     score (macro RPS, lower is better).
  2. Hyperparameters are chosen by subject-grouped cross-validation INSIDE the
     training subjects. The validation subjects are touched once, at the end.
  3. All feature sets use the same rows (complete cases over every predictor),
     so the ablation compares like with like.
  4. Uncertainty comes from a subject-level bootstrap on the validation set,
     which respects repeated visits per subject. A paired bootstrap tests
     whether adding COMIND features beats clinical-only for the same model.
  5. No silent failures: every failed fit is logged with its error.

Place this file next to weighted_ordered_model.py, ord_dts.py and
ord_dts_scoring.py (it imports them), then run e.g.

    python nsd_ablation_clean.py \
        --data /home/dsemchin/data/data_ppmi_pd_with_best_model_subject_cols_with_subtype_probas.csv \
        --out results_nsd

Add --quick for a small grid while testing.

Outputs (in --out):
    results.csv          one row per (model, feature set): chosen hyperparameters,
                         inner-CV score, validation metrics with bootstrap CIs
    paired_vs_clinical.csv  paired bootstrap of the macro-RPS change from adding
                         COMIND features, per model
    tuning_log.csv       inner-CV score of every hyperparameter configuration
    val_predictions.csv  per-visit validation probabilities for every combination
    stage_counts.csv     visits per stage per split after merging
    table_nsd.tex        LaTeX table for the thesis
    confusion_best.png   confusion matrices: best model vs its clinical-only twin
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.validation import check_is_fitted

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ord_dts import AdaBoostBinaryRegressor, OrdinalDiscreteTimeSurvival  # noqa: E402
from ord_dts_scoring import (  # noqa: E402
    macro_ranked_probability_score,
    ranked_probability_score,
)
from weighted_ordered_model import WeightedOrderedModel  # noqa: E402

log = logging.getLogger("nsd")

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

CLINICAL = ["MCATOT", "TD_score", "PIGD_score"]
SUBJ_COL, TIME_COL, SPLIT_COL, RAW_STAGE_COL = "subj_id", "time", "model_split_best", "NSD_STAGE"


def feature_sets(beta_cols: list[str], subtype_dummies: list[str]) -> dict[str, list[str]]:
    """Predictor sets compared in the ablation."""
    return {
        "clinical": CLINICAL,
        "disease_time": beta_cols,
        "clinical+disease_time": CLINICAL + beta_cols,
        "clinical+disease_time+subtype": CLINICAL + beta_cols + subtype_dummies,
        # Only the disease time under the subject's assigned subtype:
        "clinical+assigned_time": CLINICAL + ["beta_assigned"],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════════════════


def load_panel(path: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Load the PPMI panel, clean NSD-ISS stage and build COMIND features.

    Stage handling (same as the notebook):
      * 2a / 2b -> 2
      * 'Not NSD' and stage 0 -> dropped
      * 6 -> 5 (stage 6 is a single visit)
    """
    df = pd.read_csv(path)
    df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce") / 12.0  # months -> years

    raw = df[RAW_STAGE_COL].astype(str).str.strip().str.lower()
    raw = raw.replace({"2a": "2", "2b": "2"})
    stage = pd.to_numeric(raw, errors="coerce")
    stage = stage.where(stage > 0)
    stage = stage.where(stage != 6, 5)
    df["stage"] = stage

    beta_cols = sorted(
        [c for c in df.columns if c.startswith("beta_") and c[5:].isdigit()],
        key=lambda c: int(c[5:]),
    )
    if not beta_cols:
        raise ValueError("No beta_<k> columns found.")
    n_sub = len(beta_cols)

    sub = pd.to_numeric(df["subtype_best"], errors="coerce")
    beta_mat = df[beta_cols].apply(pd.to_numeric, errors="coerce").to_numpy()
    assigned = np.full(len(df), np.nan)
    ok = sub.notna().to_numpy() & sub.between(0, n_sub - 1).to_numpy()
    assigned[ok] = beta_mat[ok, sub[ok].astype(int).to_numpy()]
    df["beta_assigned"] = assigned

    # One-hot subtype, first subtype as reference.
    subtype_dummies = [f"subtype_d{k}" for k in range(1, n_sub)]
    for k in range(1, n_sub):
        df[f"subtype_d{k}"] = (sub == k).astype(float).where(sub.notna())

    df = df[df[SPLIT_COL].isin(["train", "val"])].copy()
    return df, beta_cols, subtype_dummies


def complete_cases(df: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    """Keep rows with a stage and every predictor, so all feature sets share rows."""
    cols = ["stage", SUBJ_COL, TIME_COL] + predictors
    out = df.copy()
    out[predictors] = out[predictors].apply(pd.to_numeric, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=cols)
    return out.sort_values([SUBJ_COL, TIME_COL]).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# Models: every wrapper exposes fit(X, y) and predict_proba(X) -> (n, K)
# with columns in the global stage order. X holds subj_id, time and features.
# ═══════════════════════════════════════════════════════════════════════════


def _pad_proba(proba: np.ndarray, fitted: np.ndarray, stages: np.ndarray) -> np.ndarray:
    """Place probabilities for the fitted classes into the full stage order."""
    full = np.zeros((proba.shape[0], len(stages)))
    for j, c in enumerate(fitted):
        full[:, int(np.where(stages == c)[0][0])] = proba[:, j]
    return full


class ProportionalOdds(BaseEstimator):
    """M1: proportional-odds model with class-balanced likelihood weights."""

    def __init__(self, features, stages, distr="logit", method="bfgs"):
        self.features, self.stages = features, stages
        self.distr, self.method = distr, method

    def fit(self, X, y):
        self.scaler_ = StandardScaler().fit(X[self.features])
        Z = pd.DataFrame(self.scaler_.transform(X[self.features]), columns=self.features)
        self.fitted_classes_ = np.sort(np.unique(y))
        endog = pd.Series(pd.Categorical(y, categories=self.fitted_classes_, ordered=True))
        w = compute_sample_weight("balanced", y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = WeightedOrderedModel(endog, Z, sample_weights=w, distr=self.distr)
            self.res_ = mod.fit(method=self.method, disp=False, maxiter=1000)
        conv = getattr(self.res_, "mle_retvals", {}) or {}
        if not conv.get("converged", True):
            raise RuntimeError("proportional-odds fit did not converge")
        return self

    def predict_proba(self, X):
        Z = pd.DataFrame(self.scaler_.transform(X[self.features]), columns=self.features)
        p = np.asarray(self.res_.model.predict(self.res_.params, exog=Z))
        return _pad_proba(p, self.fitted_classes_, self.stages)


class SVMBinaryClassifier(BaseEstimator):
    """SVM hazard model; decision scores mapped to (0, 1) with a sigmoid.
    The survival wrapper's calibration step corrects the scale."""

    def __init__(self, C=1.0, kernel="rbf", gamma="scale"):
        self.C, self.kernel, self.gamma = C, kernel, gamma

    def fit(self, X, y, sample_weight=None):
        self.clf_ = SVC(C=self.C, kernel=self.kernel, gamma=self.gamma, probability=False)
        self.clf_.fit(X, np.asarray(y).astype(int), sample_weight=sample_weight)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "clf_")
        p1 = np.clip(1.0 / (1.0 + np.exp(-self.clf_.decision_function(X))), 1e-6, 1 - 1e-6)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class SurvivalOrdinal(BaseEstimator):
    """M2-M4: discrete-time survival (continuation-ratio) ordinal model."""

    def __init__(self, features, stages, base_factory, include_visit_index=True,
                 enforce_threshold_ordering=True, calibration_method=None,
                 calibration_fraction=0.25, seed=0):
        self.features, self.stages, self.base_factory = features, stages, base_factory
        self.include_visit_index = include_visit_index
        self.enforce_threshold_ordering = enforce_threshold_ordering
        self.calibration_method = calibration_method
        self.calibration_fraction = calibration_fraction
        self.seed = seed

    def _prep(self, X):
        Z = X[[SUBJ_COL, TIME_COL]].copy()
        Z[self.features] = self.scaler_.transform(X[self.features])
        return Z

    def fit(self, X, y):
        self.scaler_ = StandardScaler().fit(X[self.features])
        self.model_ = OrdinalDiscreteTimeSurvival(
            order=self.stages,
            base_estimator=self.base_factory(),
            subj_id_col=SUBJ_COL,
            time_col=TIME_COL,
            include_visit_index=self.include_visit_index,
            class_weight="balanced",
            enforce_threshold_ordering=self.enforce_threshold_ordering,
            calibration_method=self.calibration_method,
            calibration_fraction=self.calibration_fraction,
            calibration_random_state=self.seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model_.fit(self._prep(X), np.asarray(y))
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(self._prep(X))


@dataclass
class ModelSpec:
    name: str
    grid: list[dict]
    build: Callable[[list[str], np.ndarray, dict], BaseEstimator]


def _survival_grid(extra: dict, quick: bool) -> list[dict]:
    common = {
        "include_visit_index": [True, False],
        "enforce_threshold_ordering": [True, False],
        "calibration_method": ["platt", "prior", None],
    }
    if quick:
        common = {"include_visit_index": [True], "enforce_threshold_ordering": [True],
                  "calibration_method": ["prior", None]}
    keys = list(extra) + list(common)
    vals = list(extra.values()) + list(common.values())
    return [dict(zip(keys, v)) for v in itertools.product(*vals)]


def model_specs(quick: bool, seed: int) -> list[ModelSpec]:
    def build_m2(feats, stages, p):
        return SurvivalOrdinal(
            feats, stages,
            base_factory=lambda: AdaBoostBinaryRegressor(
                n_estimators=p["n_estimators"], learning_rate=p["learning_rate"],
                loss="square", random_state=seed),
            include_visit_index=p["include_visit_index"],
            enforce_threshold_ordering=p["enforce_threshold_ordering"],
            calibration_method=p["calibration_method"], seed=seed)

    def build_m3(feats, stages, p):
        return SurvivalOrdinal(
            feats, stages,
            base_factory=lambda: LogisticRegression(
                C=p["C"], solver="lbfgs", max_iter=2000),
            include_visit_index=p["include_visit_index"],
            enforce_threshold_ordering=p["enforce_threshold_ordering"],
            calibration_method=p["calibration_method"], seed=seed)

    def build_m4(feats, stages, p):
        return SurvivalOrdinal(
            feats, stages,
            base_factory=lambda: SVMBinaryClassifier(C=p["C"], kernel=p["kernel"]),
            include_visit_index=p["include_visit_index"],
            enforce_threshold_ordering=p["enforce_threshold_ordering"],
            calibration_method=p["calibration_method"], seed=seed)

    m1_grid = [{"distr": d, "method": m}
               for d, m in itertools.product(["logit", "probit", "cloglog"], ["bfgs", "lbfgs"])]
    if quick:
        m1_grid = [{"distr": "logit", "method": "bfgs"}]

    m2_extra = ({"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.05, 0.1, 0.5]}
                if not quick else {"n_estimators": [50], "learning_rate": [0.05]})
    # C = 1e6 is effectively unpenalized (matches the notebook's penalty=None).
    m3_extra = {"C": [0.1, 1.0, 1e6]} if not quick else {"C": [1e6]}
    m4_extra = ({"C": [0.1, 1.0, 10.0, 100.0], "kernel": ["linear", "rbf"]}
                if not quick else {"C": [1.0], "kernel": ["linear"]})

    return [
        ModelSpec("M1 Proportional odds", m1_grid,
                  lambda f, s, p: ProportionalOdds(f, s, **p)),
        ModelSpec("M2 Survival AdaBoost", _survival_grid(m2_extra, quick), build_m2),
        ModelSpec("M3 Survival logistic", _survival_grid(m3_extra, quick), build_m3),
        ModelSpec("M4 Survival SVM", _survival_grid(m4_extra, quick), build_m4),
    ]


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════


def metrics(y: np.ndarray, proba: np.ndarray, stages: np.ndarray) -> dict:
    y = np.asarray(y, dtype=float)
    pred = stages[np.argmax(proba, axis=1)].astype(float)
    sw = compute_sample_weight("balanced", y)
    labs = np.unique(y)
    if len(np.unique(pred)) > 1 and len(labs) > 1:
        tau = kendalltau(y, pred).statistic
    else:
        tau = np.nan
    return {
        "mrps": macro_ranked_probability_score(y, proba, classes=stages),
        "rps": ranked_probability_score(y, proba, classes=stages),
        "mae": mean_absolute_error(y, pred),
        "wmae": mean_absolute_error(y, pred, sample_weight=sw),
        "tau": tau,
        "kappa": cohen_kappa_score(y, pred, weights="quadratic"),
        "f1": f1_score(y, pred, average="macro", labels=labs, zero_division=0),
        "wacc": accuracy_score(y, pred, sample_weight=sw),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Tuning (inner CV on training subjects) and evaluation
# ═══════════════════════════════════════════════════════════════════════════


def inner_cv_score(spec, params, feats, train, stages, folds) -> float:
    """Pooled out-of-fold macro RPS over subject-grouped folds."""
    oof = np.full((len(train), len(stages)), np.nan)
    y = train["stage"].to_numpy()
    for tr_idx, te_idx in folds:
        est = spec.build(feats, stages, params)
        est.fit(train.iloc[tr_idx], y[tr_idx])
        oof[te_idx] = est.predict_proba(train.iloc[te_idx])
    return macro_ranked_probability_score(y, oof, classes=stages)


def subject_bootstrap_indices(df: pd.DataFrame, n_boot: int, rng) -> list[np.ndarray]:
    """Row indices for resampled-with-replacement subjects (same draws reused
    for every model so comparisons are paired)."""
    groups = df.groupby(SUBJ_COL).indices
    subjects = np.array(list(groups))
    out = []
    for _ in range(n_boot):
        draw = rng.choice(subjects, size=len(subjects), replace=True)
        out.append(np.concatenate([groups[s] for s in draw]))
    return out


def run(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    df, beta_cols, subtype_dummies = load_panel(args.data)
    fsets = feature_sets(beta_cols, subtype_dummies)
    all_preds = sorted({c for cols in fsets.values() for c in cols})
    data = complete_cases(df, all_preds)

    train = data[data[SPLIT_COL] == "train"].reset_index(drop=True)
    val = data[data[SPLIT_COL] == "val"].reset_index(drop=True)
    stages = np.sort(train["stage"].unique()).astype(float)
    dropped = ~val["stage"].isin(stages)
    if dropped.any():
        log.warning("Dropping %d validation visits with stages unseen in training", dropped.sum())
        val = val[~dropped].reset_index(drop=True)

    counts = (data.groupby([SPLIT_COL, "stage"]).size().unstack(fill_value=0)
              .assign(subjects=data.groupby(SPLIT_COL)[SUBJ_COL].nunique()))
    counts.to_csv(out / "stage_counts.csv")
    log.info("Stages %s | train %d visits / %d subjects | val %d visits / %d subjects",
             stages.astype(int).tolist(), len(train), train[SUBJ_COL].nunique(),
             len(val), val[SUBJ_COL].nunique())
    log.info("Stage counts:\n%s", counts)

    # Subject-grouped, stage-stratified folds, shared by every model and feature set.
    sgkf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    folds = list(sgkf.split(train, train["stage"].astype(int), groups=train[SUBJ_COL]))

    boot = subject_bootstrap_indices(val, args.n_boot, rng)
    y_val = val["stage"].to_numpy()

    rows, tuning_rows, pred_frames, val_proba = [], [], [], {}
    for spec in model_specs(args.quick, args.seed):
        for fs_name, feats in fsets.items():
            t0 = time.time()
            best_score, best_params = np.inf, None
            for params in spec.grid:
                try:
                    score = inner_cv_score(spec, params, feats, train, stages, folds)
                    err = ""
                except Exception as e:  # logged, never silent
                    score, err = np.inf, f"{type(e).__name__}: {e}"
                tuning_rows.append({"model": spec.name, "features": fs_name,
                                    "params": repr(params), "cv_mrps": score, "error": err})
                if score < best_score:
                    best_score, best_params = score, params
            if best_params is None:
                log.error("%s | %s: every configuration failed", spec.name, fs_name)
                continue

            est = spec.build(feats, stages, best_params)
            est.fit(train, train["stage"].to_numpy())
            p_val = est.predict_proba(val)
            val_proba[(spec.name, fs_name)] = p_val

            m = metrics(y_val, p_val, stages)
            bm = pd.DataFrame([metrics(y_val[i], p_val[i], stages) for i in boot])
            row = {"model": spec.name, "features": fs_name, "params": repr(best_params),
                   "cv_mrps": best_score}
            for k, v in m.items():
                row[f"val_{k}"] = v
                row[f"val_{k}_lo"], row[f"val_{k}_hi"] = np.nanpercentile(bm[k], [2.5, 97.5])
            rows.append(row)

            pf = val[[SUBJ_COL, TIME_COL, "stage"]].copy()
            pf["model"], pf["features"] = spec.name, fs_name
            for j, s in enumerate(stages):
                pf[f"p_stage_{int(s)}"] = p_val[:, j]
            pf["pred_stage"] = stages[np.argmax(p_val, axis=1)]
            pred_frames.append(pf)

            log.info("%-22s %-30s cv_mRPS=%.3f  val_mRPS=%.3f [%.3f, %.3f]  tau=%.2f  (%.0fs)",
                     spec.name, fs_name, best_score, m["mrps"],
                     row["val_mrps_lo"], row["val_mrps_hi"], m["tau"], time.time() - t0)

    results = pd.DataFrame(rows)
    results.to_csv(out / "results.csv", index=False)
    pd.DataFrame(tuning_rows).to_csv(out / "tuning_log.csv", index=False)
    pd.concat(pred_frames).to_csv(out / "val_predictions.csv", index=False)

    # Paired bootstrap: change in macro RPS vs clinical-only, same model.
    paired = []
    for (model, fs_name), p in val_proba.items():
        if fs_name == "clinical" or (model, "clinical") not in val_proba:
            continue
        p0 = val_proba[(model, "clinical")]
        d = np.array([
            macro_ranked_probability_score(y_val[i], p[i], classes=stages)
            - macro_ranked_probability_score(y_val[i], p0[i], classes=stages)
            for i in boot
        ])
        point = (macro_ranked_probability_score(y_val, p, classes=stages)
                 - macro_ranked_probability_score(y_val, p0, classes=stages))
        paired.append({"model": model, "features": fs_name, "delta_mrps": point,
                       "delta_lo": np.percentile(d, 2.5), "delta_hi": np.percentile(d, 97.5),
                       "p_improves": float(np.mean(d < 0))})
    paired = pd.DataFrame(paired)
    paired.to_csv(out / "paired_vs_clinical.csv", index=False)

    write_latex_table(results, out / "table_nsd.tex")
    plot_confusions(results, val_proba, y_val, stages, out / "confusion_best.png")

    pd.set_option("display.width", 200)
    cols = ["model", "features", "cv_mrps", "val_mrps", "val_mrps_lo", "val_mrps_hi",
            "val_tau", "val_kappa", "val_wmae"]
    print("\n== Results (validation, 95% subject-bootstrap CI) ==")
    print(results[cols].round(3).to_string(index=False))
    if not paired.empty:
        print("\n== Change in macro RPS vs clinical-only (negative = better) ==")
        print(paired.round(3).to_string(index=False))
    best = results.loc[results["cv_mrps"].idxmin()]
    print(f"\nBest by inner-CV macro RPS: {best['model']} | {best['features']} "
          f"| val mRPS {best['val_mrps']:.3f}")


# ═══════════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════════

FS_LABELS = {
    "clinical": "Clinical",
    "disease_time": "Disease time",
    "clinical+disease_time": "Clinical + disease time",
    "clinical+disease_time+subtype": "Clinical + disease time + subtype",
    "clinical+assigned_time": "Clinical + assigned disease time",
}


def write_latex_table(results: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{table}[ht]",
        r"\caption{Validation performance for NSD-ISS stage prediction. "
        r"Macro RPS: lower is better. Brackets: 95\% subject-level bootstrap interval}",
        r"\centering",
        r"\begin{tabular}{l l c c c c}",
        r"\hline\hline",
        r"Model & Predictors & Macro RPS & Kendall $\tau$ & Quadratic $\kappa$ & Weighted MAE \\ [0.5ex]",
        r"\hline",
    ]
    for model, grp in results.groupby("model", sort=False):
        for i, (_, r) in enumerate(grp.iterrows()):
            lines.append(
                f"{model.split()[0] if i == 0 else ''} & {FS_LABELS.get(r['features'], r['features'])} & "
                f"{r['val_mrps']:.3f} [{r['val_mrps_lo']:.2f}, {r['val_mrps_hi']:.2f}] & "
                f"{r['val_tau']:.2f} & {r['val_kappa']:.2f} & {r['val_wmae']:.2f} \\\\"
            )
        lines.append(r"\hline")
    lines += [r"\end{tabular}", r"\label{tab:nsd_results}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n")


def plot_confusions(results, val_proba, y_val, stages, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    best = results.loc[results["cv_mrps"].idxmin()]
    pairs = [(best["model"], best["features"])]
    if best["features"] != "clinical" and (best["model"], "clinical") in val_proba:
        pairs.insert(0, (best["model"], "clinical"))

    fig, axes = plt.subplots(1, len(pairs), figsize=(4.5 * len(pairs), 4), squeeze=False)
    for ax, key in zip(axes[0], pairs):
        pred = stages[np.argmax(val_proba[key], axis=1)]
        cm = confusion_matrix(y_val, pred, labels=stages)
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black")
        ticks = [str(int(s)) for s in stages]
        ax.set_xticks(range(len(stages)), ticks)
        ax.set_yticks(range(len(stages)), ticks)
        ax.set_xlabel("Predicted NSD-ISS stage")
        ax.set_ylabel("Observed NSD-ISS stage")
        ax.set_title(f"{key[0].split()[0]} | {FS_LABELS.get(key[1], key[1])}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True, help="PPMI panel CSV with COMIND outputs")
    ap.add_argument("--out", default="results_nsd", help="output directory")
    ap.add_argument("--folds", type=int, default=3,
                    help="inner CV folds (few subjects reach stage 5, so keep this small)")
    ap.add_argument("--n-boot", type=int, default=1000, help="subject bootstrap draws")
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--quick", action="store_true", help="tiny grids for a smoke test")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    run(args)


if __name__ == "__main__":
    main()