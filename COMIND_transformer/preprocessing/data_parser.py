from typing import List
import warnings
import numpy as np
import pandas as pd

def parse_data(
    data: pd.DataFrame,
    subj_id_col: str,
    time_col: str,
    X_cols: List[str],
    clinical_cols: List[str],
    metadata_cols: List[str],
) -> List[dict]:
    """
    Reshape a precleaned longitudinal dataframe into COMIND's patient-list format.

    "X_cols": are biomarkers that are modeled logistically as part of the
    connectome-coupled ODE state; gets its own trajectory, subject to network
    coupling via your transition matrix K.

    Assumes X_cols are already in the domain SubtypingEM expects (non-negative
    and increasing).

    "clinical_cols": are used only as a linear predictor to initialize each
    patient's disease-time offset (beta) via a mixed-effects regression;
    never coupled through K, has no trajectory of its own.

    One dict per unique subj_id_col value, visits sorted by time_col:
      - "id": subject id
      - "X_obs": (n_visits, len(X_cols)) float array
      - "dt": (n_visits,) float array, raw time_col values
      - "cog": (n_visits, len(clinical_cols)) float array
      - one array per metadata_cols entry, raw dtype, NaN allowed

    Raises if id_col, time_col, X_cols, or clinical_cols contain missing
    values -- clinical_cols feeds a linear regression (beta initialization)
    that cannot handle NaN cleanly. Only metadata_cols may be NaN-inclusive.
    """
    required = [subj_id_col, time_col] + list(X_cols) + list(clinical_cols) + list(metadata_cols)
    missing_cols = [c for c in required if c not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    overlap = set(X_cols) & set(clinical_cols)
    if overlap:
        warnings.warn(
            f"Column(s) {overlap} appear in both X_cols and clinical_cols -- "
            f"this means they'll be modeled logistically AND used to "
            f"initialize beta. Make sure that's intentional.",
            stacklevel=2,
        )

    if data[subj_id_col].isna().any() or data[time_col].isna().any():
        raise ValueError(f"{subj_id_col} and {time_col} may not contain missing values")

    def _check_complete(cols: List[str], label: str, reason: str) -> None:
        if not cols:
            return
        col_missing = data[cols].isna()
        if col_missing.any().any():
            bad_rows = col_missing.any(axis=1)
            bad_ids = data.loc[bad_rows, subj_id_col].unique()
            raise ValueError(
                f"{label} contain missing values in {bad_rows.sum()} row(s) "
                f"({len(bad_ids)} subject(s), e.g. {list(bad_ids[:5])}). {reason}"
            )

    _check_complete(
        X_cols, "X_cols",
        "SubtypingEM cannot fit with NaN in X_obs -- drop or impute these rows "
        "before calling parse_data.",
    )
    _check_complete(
        clinical_cols, "clinical_cols",
        "The beta-initialization regression cannot fit with NaN in cog -- drop "
        "or impute these rows, or move this column to metadata_cols if it's "
        "not meant to inform beta initialization.",
    )

    if (data[X_cols].to_numpy(dtype=float) < 0).any():
        warnings.warn(
            "Negative values found in X_cols. SubtypingEM requires non-negative "
            "X_obs (s >= 0 constraint) -- if you haven't already, transform your "
            "biomarkers (sign flip or winsorize or shift to non-negative) before "
            "calling parse_data.",
            stacklevel=2,
        )

    patients = []
    for pid, group in data.groupby(subj_id_col, sort=False):
        group = group.sort_values(time_col)
        p = {
            "id": pid,
            "X_obs": group[X_cols].to_numpy(dtype=float),
            "dt": group[time_col].to_numpy(dtype=float),
            "cog": group[clinical_cols].to_numpy(dtype=float),
        }
        for col in metadata_cols:
            p[col] = group[col].to_numpy()
        patients.append(p)
    return patients