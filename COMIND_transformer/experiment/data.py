"""Shared pieces of the ADNI and PPMI loaders."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

Z_OUTLIER_SD = 5.0


@dataclass
class Dataset:
    """Patient dicts (``parse_data`` format) plus what the runner needs around them.

    ``extra_val`` holds additional held-out sets by name (e.g. ADNI CN A+).
    ``info`` holds aggregate counts only (no subject IDs), safe for manifests.
    """
    x_train: List[Dict[str, Any]]
    x_val: List[Dict[str, Any]]
    K: np.ndarray
    x_cols: List[str]
    groups: Dict[str, np.ndarray]  # "imaging" / "clinical" biomarker indices
    extra_val: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_rows_train(self) -> int:
        return int(sum(len(p["dt"]) for p in self.x_train))


def biomarker_groups(x_cols: Sequence[str], clinical_cols: Sequence[str]) -> Dict[str, np.ndarray]:
    is_clin = np.array([c in set(clinical_cols) for c in x_cols], dtype=bool)
    return {"imaging": np.flatnonzero(~is_clin), "clinical": np.flatnonzero(is_clin)}


def biomarker_weights(groups: Dict[str, np.ndarray], share: Optional[float]) -> Optional[np.ndarray]:
    """w_img = 1, w_clin = share / (1 - share) * B_img / B_clin. None = equal weights."""
    if share is None:
        return None
    n_img, n_clin = len(groups["imaging"]), len(groups["clinical"])
    if n_clin == 0:
        raise ValueError("clinical_weight_share needs at least one clinical biomarker")
    w = np.ones(n_img + n_clin)
    w[groups["clinical"]] = share / (1.0 - share) * n_img / n_clin
    return w


def mask_and_flip(z: np.ndarray) -> Tuple[np.ndarray, int]:
    """Mask |z| > 5 and flip the sign so atrophy is positive (z scale kept)."""
    z = np.asarray(z, dtype=float).copy()
    outlier = np.isfinite(z) & (np.abs(z) > Z_OUTLIER_SD)
    z[outlier] = np.nan
    return -z, int(outlier.sum())


def clinical_to_z(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    columns: Dict[str, Tuple[str, float]],
    train_ids: Sequence[Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """PLACEHOLDER clinical z-score until the Gaussianized score replaces it.

    ``columns`` maps output column -> (raw column, sign). Each score is flipped
    so worse is positive, then z-scored against the training subjects' first
    finite visit. Clinical z = 0 is the training-patient mean, not a healthy
    reference like imaging z = 0.
    """
    df = df.copy()
    train_set = set(map(str, train_ids))
    info: Dict[str, Any] = {}
    for col, (raw_col, sign) in columns.items():
        raw = sign * pd.to_numeric(df[raw_col], errors="coerce")
        ref = df.assign(_v=raw)
        ref = ref[ref[id_col].astype(str).isin(train_set) & np.isfinite(ref["_v"])]
        first = ref.sort_values([id_col, time_col]).groupby(id_col, sort=False)["_v"].first()
        if len(first) < 2:
            raise RuntimeError(f"clinical_to_z: < 2 training first visits for {col}")
        mu, sd = float(first.mean()), float(first.std(ddof=0))
        if not np.isfinite(sd) or sd <= 0:
            raise RuntimeError(f"clinical_to_z: non-positive SD for {col}")
        df[col] = (raw.to_numpy(dtype=float) - mu) / sd
        info[col] = {"sign": sign, "mean": mu, "sd": sd, "n_ref": int(len(first))}
    return df, info


def shift_beta(pid_to_beta: Dict[str, float], patients: Sequence[Dict[str, Any]], t_max: float) -> None:
    """Write ``initial_beta`` on each patient: MixedLM value shifted to min 0, clipped to t_max."""
    beta = np.array([float(pid_to_beta.get(str(p["id"]), 0.0)) for p in patients])
    beta = np.clip(beta - np.nanmin(beta), 0.0, t_max)
    for p, b in zip(patients, beta):
        p["initial_beta"] = float(b)
