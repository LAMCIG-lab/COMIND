"""
PPMI loader (z-score input): PD subjects, deviation-score imaging flipped and
masked at |z| > 5, MoCA / TD / PIGD z-scored on the training split. Train =
subjects with >= 2 MRI visits, val = exactly 1. Ported from the 2026_11_02
zsym driver (``--input zscore``); the MixedLM beta start now uses training
subjects only.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..preprocessing import build_connectome, parse_data
from ..utils import fit_mixedlm_beta_from_clinical
from .adni import _cramers_v
from .data import Dataset, biomarker_groups, clinical_to_z, mask_and_flip, shift_beta

ID_COL, TIME_COL = "subj_id", "time"
FEATURE_SUFFIX = "_Z_predict"
N_IMAGING = 82
CLINICAL_COLS = ["MCATOT", "TD_score", "PIGD_score"]
COG_COLS = [f"{c}__raw" for c in CLINICAL_COLS]
CLINICAL_Z = {"MCATOT": ("MCATOT__raw", -1.0), "TD_score": ("TD_score__raw", 1.0),
              "PIGD_score": ("PIGD_score__raw", 1.0)}
METADATA_CANDIDATES = ("NHY", "NSD_STAGE", "PDSTATE", "group", "subgroup", "SEX", "AGE_AT_VISIT")
GENETIC_LEVELS = ("Sporadic PD", "GBA", "PRKN", "LRRK2", "LRRK2 + GBA")


def resolve_pdstate(df: pd.DataFrame) -> pd.DataFrame:
    """Drop ON rows where an OFF row exists for the same visit."""
    has_off = df.groupby([ID_COL, TIME_COL])["PDSTATE"].transform(lambda s: "OFF" in s.values)
    return df.loc[~((df["PDSTATE"] == "ON") & has_off)].copy()


def merge_duplicate_visits(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Coalesce duplicate (subject, time) rows; raise if a modeling column conflicts."""
    dup = df.duplicated(subset=[ID_COL, TIME_COL], keep=False)
    if not dup.any():
        return df.copy()
    for key, g in df.loc[dup].groupby([ID_COL, TIME_COL]):
        bad = [c for c in cols if g[c].dropna().nunique() > 1]
        if bad:
            raise ValueError(f"duplicate visit {key} has conflicting values in {bad}")
    return (
        df.sort_values([ID_COL, TIME_COL])
        .groupby([ID_COL, TIME_COL], as_index=False)
        .apply(lambda g: g.apply(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan),
               include_groups=False)
        .reset_index(drop=True)
    )


def load(data_csv: str, connectome: str, *, t_max: float = 40.0, seed: int = 75) -> Dataset:
    df = pd.read_csv(data_csv, low_memory=False).replace([np.inf, -np.inf], np.nan)
    img = list(pd.read_csv(connectome, index_col=0).index)
    if len(img) != N_IMAGING or any(not c.endswith(FEATURE_SUFFIX) or c not in df.columns for c in img):
        raise ValueError(f"connectome labels must be the {N_IMAGING} *{FEATURE_SUFFIX} columns")
    x_cols = img + CLINICAL_COLS
    metadata = [c for c in METADATA_CANDIDATES if c in df.columns]

    df = merge_duplicate_visits(resolve_pdstate(df), x_cols + metadata)
    df = df.loc[df["group"].astype(str) == "PD"].copy()
    for raw, col in zip(COG_COLS, CLINICAL_COLS):
        df[raw] = df[col].to_numpy(dtype=float)
    df[TIME_COL] = df[TIME_COL].to_numpy(dtype=float) / 12.0  # months -> years
    df[img], n_masked = mask_and_flip(df[img].to_numpy(dtype=float))
    df = df.loc[df[x_cols].notna().any(axis=1) & df[COG_COLS].notna().all(axis=1)].copy()

    n_mri = df.groupby(ID_COL)[img].apply(lambda g: int(g.notna().any(axis=1).sum()))
    train_ids = [str(k) for k, v in n_mri.items() if v >= 2]
    val_ids = [str(k) for k, v in n_mri.items() if v == 1]
    df, z_info = clinical_to_z(df, ID_COL, TIME_COL, CLINICAL_Z, train_ids)

    patients = parse_data(df, subj_id_col=ID_COL, time_col=TIME_COL, X_cols=x_cols,
                          clinical_cols=COG_COLS, metadata_cols=metadata)
    by_id = {str(p["id"]): p for p in patients}
    x_train = [by_id[i] for i in train_ids]

    # MixedLM on the raw clinical scores of the training subjects.
    raw = df.loc[df[ID_COL].astype(str).isin(set(train_ids)), [ID_COL, TIME_COL] + COG_COLS]
    raw.columns = [ID_COL, TIME_COL] + CLINICAL_COLS
    _, pid_to_beta, _ = fit_mixedlm_beta_from_clinical(
        df=raw, ids=raw[ID_COL].to_numpy(), dt=raw[TIME_COL].to_numpy(dtype=float), t_max=t_max,
        rng=np.random.default_rng(seed),
    )
    shift_beta({str(k): v for k, v in pid_to_beta.items()}, x_train, t_max)

    K, disconnected = build_connectome(connectome, x_cols, pad_missing=True)
    if set(disconnected) & set(img):
        raise ValueError("imaging columns missing from the connectome")
    return Dataset(
        x_train=x_train, x_val=[by_id[i] for i in val_ids], K=K, x_cols=x_cols,
        groups=biomarker_groups(x_cols, CLINICAL_COLS),
        info={"n_train": len(x_train), "n_val": len(val_ids), "n_imaging_cells_masked": n_masked,
              "clinical_z": z_info},
    )


def _visit(p: Dict[str, Any], key: str, idx: int):
    arr = np.atleast_1d(np.asarray(p.get(key, []), dtype=object))
    return arr[idx] if len(arr) > idx else None


def nsd_stage(v) -> Optional[int]:
    """Positive NSD stage ("2a"/"2b" -> 2); None for Not NSD / missing."""
    s = str(v).strip().lower()
    if s in ("2a", "2b"):
        return 2
    try:
        n = int(float(s))
    except ValueError:
        return None
    return n if n > 0 else None


def nhy(v) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) and int(round(x)) != 0 else None


def staging_spearman(patients, betas, split: str) -> Dict[str, float]:
    """Spearman of disease time with NSD stage and H&Y.

    train: beta + last dt vs the stage at the last visit; val: the first visit.
    """
    pairs = {"nsd": ([], []), "nhy": ([], [])}
    for p, b in zip(patients, np.asarray(betas, float)):
        dts = np.asarray(p["dt"], dtype=float)
        idx = int(np.argmax(dts)) if split == "train" else int(np.argmin(dts))
        for name, parse, key in (("nsd", nsd_stage, "NSD_STAGE"), ("nhy", nhy, "NHY")):
            y = parse(_visit(p, key, idx))
            if y is not None and np.isfinite(b):
                pairs[name][0].append(dts[idx] + b)
                pairs[name][1].append(y)
    out = {}
    for name, (x, y) in pairs.items():
        ok = len(x) >= 3 and len(set(y)) >= 2
        out[f"spearman_{name}_{split}"] = float(spearmanr(x, y)[0]) if ok else float("nan")
    return out


def subtype_by_genetics(subtypes, patients, n_subtypes: int) -> Dict[str, Any]:
    labels = []
    for p in patients:
        vals = [x for x in np.atleast_1d(np.asarray(p.get("subgroup"), dtype=object)) if isinstance(x, str)]
        labels.append(vals[0] if vals else "UNK")
    levels = [g for g in GENETIC_LEVELS if g in labels]
    table = np.zeros((n_subtypes, len(levels)), dtype=int)
    carrier = np.zeros((n_subtypes, 2), dtype=int)
    for z, lab in zip(np.asarray(subtypes, int), labels):
        if lab in levels:
            table[z, levels.index(lab)] += 1
            carrier[z, int(lab != "Sporadic PD")] += 1
    return {"levels": levels, "table": table.tolist(), "cramers_v": _cramers_v(table),
            "carrier_table": carrier.tolist(), "carrier_cramers_v": _cramers_v(carrier)}


def external_checks(est, data: Dataset, val_beta, val_subtypes) -> Dict[str, Any]:
    """Staging correlations and subtype x genetics on train and val. Never used for selection."""
    z = int(est.n_subtypes)
    gen_tr = subtype_by_genetics(est.final_assignments, data.x_train, z)
    gen_va = subtype_by_genetics(val_subtypes, data.x_val, z)
    return {
        **staging_spearman(data.x_train, est.final_beta_, "train"),
        **staging_spearman(data.x_val, val_beta, "val"),
        "genetics_train": gen_tr, "genetics_val": gen_va,
        "genetic_cramers_v_train": gen_tr["cramers_v"], "genetic_cramers_v_val": gen_va["cramers_v"],
        "carrier_cramers_v_train": gen_tr["carrier_cramers_v"],
        "carrier_cramers_v_val": gen_va["carrier_cramers_v"],
    }
