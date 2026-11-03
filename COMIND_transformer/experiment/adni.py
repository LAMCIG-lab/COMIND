"""
ADNI loader (z-score input): W-score imaging flipped and masked at |z| > 5,
MoCA (MMSE-filled) and ADAS13 z-scored on the training split, MixedLM beta
start on the training split. Ported from the 2026_10_28 zsym driver
(``--input zscore``).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split

from ..preprocessing import build_connectome, parse_data
from ..utils import fit_mixedlm_beta_from_clinical
from .data import Dataset, biomarker_groups, clinical_to_z, mask_and_flip, shift_beta

ID_COL, TIME_COL = "PTID", "dt"
DX_COL, DX_BL_COL, AMYLOID_COL = "DX", "DX_bl", "amyloid"
FEATURE_SUFFIX = "_wscore"
N_IMAGING = 82
MIN_MRI = 2
CLINICAL_COLS = ["MOCA", "ADAS13"]
# output column -> (raw column, sign): worse is positive after the flip
CLINICAL_Z = {"MOCA": ("MOCA__raw", -1.0), "ADAS13": ("ADAS13__raw", 1.0)}
METADATA_COLS = [
    "DX", "DX_bl", "AGE", "PTGENDER", "APOE4", "MMSE", "CDRSB", "obs_type", "AV45", "FBB",
    "ABETA", "amyloid", "wscore_ref", "MOCA__raw", "ADAS13__raw", "MMSE__raw",
]
MCI_BL = {"MCI", "EMCI", "LMCI"}
MOCA_TO_PRED_MMSE = {
    30: 30, 29: 30, 28: 29, 27: 29, 26: 28, 25: 28, 24: 28,
    23: 27, 22: 27, 21: 26, 20: 25, 19: 25, 18: 24, 17: 24, 16: 23, 15: 22,
}


def imaging_cols(columns: Sequence[str]) -> List[str]:
    """Cortical and subcortical W-score columns, cerebellum dropped."""
    cols = [
        c for c in columns
        if c.endswith(FEATURE_SUFFIX) and c.startswith(("ctx-", "Left-", "Right-"))
        and "erebell" not in c.lower()
    ]
    if len(cols) != N_IMAGING:
        raise ValueError(f"expected {N_IMAGING} imaging columns, found {len(cols)}")
    return cols


def connectome_name(col: str) -> str:
    region = col[: -len(FEATURE_SUFFIX)]
    for prefix, hemi in (("ctx-lh-", "L_"), ("ctx-rh-", "R_")):
        if region.startswith(prefix):
            return f"{hemi}{region[len(prefix):]}_Z_predict"
    return f"{region}_Z_predict"


def mmse_to_moca(mmse: np.ndarray) -> np.ndarray:
    """Mean MoCA per predicted-MMSE bucket of the MoCA->MMSE table, interpolated."""
    buckets: Dict[int, List[int]] = {}
    for moca, mm in MOCA_TO_PRED_MMSE.items():
        buckets.setdefault(mm, []).append(moca)
    keys = np.array(sorted(buckets), dtype=float)
    vals = np.array([np.mean(buckets[int(k)]) for k in keys])
    mmse = np.asarray(mmse, dtype=float)
    out = np.full(mmse.shape, np.nan)
    ok = np.isfinite(mmse)
    out[ok] = np.interp(np.clip(mmse[ok], keys.min(), keys.max()), keys, vals)
    return out


def subject_keep(g: pd.DataFrame, img: List[str], include_cn_amyloid_pos: bool) -> Optional[str]:
    """Cohort rule for one subject; returns "keep", "cn_apos" or None (drop).

    AD: keep A+ and A?. MCI: keep A+, and A? if ever demented. CN: keep A+ only
    with ``include_cn_amyloid_pos``. SMC, missing DX_bl and A- are dropped.
    """
    if int(g[img].notna().any(axis=1).sum()) < MIN_MRI:
        return None
    dx_bl = g[DX_BL_COL].dropna()
    bl = str(dx_bl.iloc[0]) if len(dx_bl) else ""
    amy = g[AMYLOID_COL].dropna().astype(str).str.strip()
    lab = str(amy.iloc[0]) if len(amy) else "A?"
    ever_dem = bool(g[DX_COL].dropna().astype(str).str.upper().isin(["DEMENTIA", "AD"]).any())
    if bl == "CN":
        if lab == "A+":
            return "cn_apos" if include_cn_amyloid_pos else "cn_apos_excluded"
        return None
    if bl == "AD":
        return "keep" if lab != "A-" else None
    if bl in MCI_BL:
        return "keep" if lab == "A+" or (lab == "A?" and ever_dem) else None
    return None


def apoe_label(value) -> str:
    try:
        a = float(np.atleast_1d(value)[0])
    except (TypeError, ValueError, IndexError):
        return "UNK"
    return str(int(a)) if np.isfinite(a) else "UNK"


def split_ids(first: pd.DataFrame, cn_apos: set, *, cn_apos_in_main_split: bool,
              seed: int, test_size: float):
    """(train, val, val_cn_apos) subject IDs.

    With ``cn_apos_in_main_split`` all subjects are split together, stratified
    by DX_bl x amyloid. Otherwise non-CN-A+ subjects are split stratified by
    APOE4, CN A+ subjects separately; their train part joins train.
    """
    def _split(ids, labels):
        if len(ids) < 2:
            return list(ids), []
        try:
            tr, va = train_test_split(ids, test_size=test_size, random_state=seed, stratify=labels)
        except ValueError:
            tr, va = train_test_split(ids, test_size=test_size, random_state=seed)
        return list(tr), list(va)

    ids = list(first.index)
    if cn_apos_in_main_split:
        tr, va = _split(ids, [f"{first.at[i, DX_BL_COL]}|{first.at[i, AMYLOID_COL]}" for i in ids])
        return tr, va, []
    base = [i for i in ids if i not in cn_apos]
    tr, va = _split(base, [apoe_label(first.at[i, "APOE4"]) for i in base])
    cn = [i for i in ids if i in cn_apos]
    cn_tr, cn_va = _split(cn, [apoe_label(first.at[i, "APOE4"]) for i in cn]) if cn else ([], [])
    return tr + cn_tr, va, cn_va


def train_beta(df: pd.DataFrame, x_train: List[Dict[str, Any]], t_max: float, seed: int) -> None:
    """MixedLM on the training rows with both MoCA and ADAS13; writes ``initial_beta``."""
    def z(x):
        sd = np.nanstd(x)
        return (x - np.nanmean(x)) / (sd if np.isfinite(sd) and sd > 0 else 1.0)

    ids = {str(p["id"]) for p in x_train}
    clin = df[df[ID_COL].astype(str).isin(ids)]
    moca, adas = clin["MOCA__raw"].to_numpy(float), clin["ADAS13__raw"].to_numpy(float)
    ok = np.isfinite(moca) & np.isfinite(adas)
    _, pid_to_beta, _ = fit_mixedlm_beta_from_clinical(
        ids=clin[ID_COL].astype(str).to_numpy()[ok], dt=clin[TIME_COL].to_numpy(float)[ok],
        cog=(z(moca[ok]) - z(adas[ok])) / 2.0, t_max=t_max, rng=np.random.default_rng(seed),
    )
    shift_beta({str(k): v for k, v in pid_to_beta.items()}, x_train, t_max)


def load(data_csv: str, connectome: str, *, include_cn_amyloid_pos: bool = False,
         cn_apos_in_main_split: bool = False, t_max: float = 40.0, seed: int = 75,
         test_size: float = 0.2) -> Dataset:
    df = pd.read_csv(data_csv, low_memory=False)
    ref_ids = set(df.loc[df["wscore_ref"].astype(str).isin(["True", "true", "1"]), ID_COL].astype(str))
    img = imaging_cols(df.columns)
    x_cols = img + CLINICAL_COLS
    df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce")
    if df[TIME_COL].isna().any():
        raise ValueError(f"{TIME_COL} has NaNs")

    flipped, n_masked = mask_and_flip(df[img].apply(pd.to_numeric, errors="coerce").to_numpy(float))
    df[img] = flipped
    for col in ("MOCA", "ADAS13", "MMSE"):
        df[f"{col}__raw"] = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
    moca = df["MOCA__raw"].to_numpy(float)
    fill = ~np.isfinite(moca) & np.isfinite(df["MMSE__raw"].to_numpy(float))
    moca[fill] = mmse_to_moca(df["MMSE__raw"].to_numpy(float)[fill])
    df["MOCA__raw"] = moca
    df["MOCA"], df["ADAS13"] = df["MOCA__raw"], df["ADAS13__raw"]  # replaced by clinical_to_z

    df = df.loc[df[x_cols].notna().any(axis=1)].sort_values([ID_COL, TIME_COL])
    rule = {pid: subject_keep(g, img, include_cn_amyloid_pos) for pid, g in df.groupby(ID_COL, sort=False)}
    keep = [pid for pid, r in rule.items() if r in ("keep", "cn_apos")]
    if ref_ids & set(map(str, keep)):
        raise RuntimeError("wscore_ref subjects entered the fit cohort")
    df = df[df[ID_COL].isin(keep)].copy()
    df[ID_COL] = df[ID_COL].astype(str)
    cn_apos = {str(pid) for pid, r in rule.items() if r == "cn_apos"}

    first = df.groupby(ID_COL, sort=False)[[DX_BL_COL, AMYLOID_COL, "APOE4"]].first()
    train_ids, val_ids, val_cn_ids = split_ids(
        first, cn_apos, cn_apos_in_main_split=cn_apos_in_main_split, seed=seed, test_size=test_size,
    )
    df, z_info = clinical_to_z(df, ID_COL, TIME_COL, CLINICAL_Z, train_ids)

    patients = parse_data(df, subj_id_col=ID_COL, time_col=TIME_COL, X_cols=x_cols, clinical_cols=[],
                          metadata_cols=[c for c in METADATA_COLS if c in df.columns])
    by_id = {str(p["id"]): p for p in patients}
    x_train = [by_id[i] for i in train_ids]
    train_beta(df, x_train, t_max, seed)

    K, disconnected = build_connectome(connectome, x_cols, x_to_k={c: connectome_name(c) for c in img},
                                       pad_missing=True)
    if set(disconnected) & set(img):
        raise ValueError("imaging columns missing from the connectome")
    extra = {"cn_apos": [by_id[i] for i in val_cn_ids]} if val_cn_ids else {}
    return Dataset(
        x_train=x_train, x_val=[by_id[i] for i in val_ids], K=K, x_cols=x_cols,
        groups=biomarker_groups(x_cols, CLINICAL_COLS), extra_val=extra,
        info={"n_train": len(x_train), "n_val": len(val_ids), "n_val_cn_apos": len(val_cn_ids),
              "n_imaging_cells_masked": n_masked, "n_moca_filled_from_mmse": int(fill.sum()),
              "clinical_z": z_info},
    )


def _cramers_v(table: np.ndarray) -> float:
    t = table[table.sum(axis=1) > 0][:, table.sum(axis=0) > 0]
    if t.shape[0] < 2 or t.shape[1] < 2:
        return float("nan")
    chi2 = float(chi2_contingency(t, correction=False)[0])
    return float(np.sqrt(chi2 / (t.sum() * (min(t.shape) - 1))))


def subtype_by_apoe4(subtypes, patients, n_subtypes: int) -> Dict[str, Any]:
    labels = [apoe_label(p.get("APOE4")) for p in patients]
    levels = sorted({a for a in labels if a != "UNK"})
    table = np.zeros((n_subtypes, len(levels)), dtype=int)
    for z, a in zip(np.asarray(subtypes, int), labels):
        if a != "UNK":
            table[z, levels.index(a)] += 1
    return {"levels": levels, "table": table.tolist(), "cramers_v": _cramers_v(table)}


def external_checks(est, data: Dataset, val_beta, val_subtypes) -> Dict[str, Any]:
    """Subtype x APOE4 on train and val. Never used for selection."""
    z = int(est.n_subtypes)
    tr = subtype_by_apoe4(est.final_assignments, data.x_train, z)
    va = subtype_by_apoe4(val_subtypes, data.x_val, z)
    return {"apoe4_train": tr, "apoe4_val": va,
            "apoe4_cramers_v_train": tr["cramers_v"], "apoe4_cramers_v_val": va["cramers_v"]}
