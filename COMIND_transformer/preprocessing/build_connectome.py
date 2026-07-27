from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def build_connectome(
    K_path: str,
    X_cols: List[str],
    x_to_k: Optional[Dict[str, str]] = None,
    pad_missing: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    """
    Load a connectome CSV and align it to X_cols order.

    K_path
        CSV with a square connectivity matrix, first column as row names,
        matching column headers (readable with pd.read_csv(path, index_col=0)).
        Row order in the file does not need to match X_cols -- named rows are
        looked up by label, not position.
    X_cols
        Biomarker names in the same order used for X_obs (parse_data's X_cols).
    x_to_k
        Optional dict, X_cols name -> connectome CSV name, for entries where
        the two use different naming conventions. Only needed for names that
        differ; anything not in this dict is looked up under its own name.
    pad_missing
        If False (default): every X_cols name (after applying x_to_k) must
        be found in the connectome, or this raises -- no silent gaps.
        If True: any X_cols name not found in the connectome is treated as a
        disconnected node (zero row/column at its position in X_cols).

    Returns
    -------
    K : (len(X_cols), len(X_cols)) array, aligned to X_cols order, diagonal
        zeroed
    disconnected : list of X_cols names with no match in the connectome
        (always [] when pad_missing=False, since that raises instead)
    """
    K_df = pd.read_csv(K_path, index_col=0)
    if K_df.index.tolist() != K_df.columns.tolist():
        raise ValueError("Connectome CSV must be square with matching row/column names")

    x_to_k = x_to_k or {}

    found, missing = [], []
    for name in X_cols:
        k_name = x_to_k.get(name, name)
        (found if k_name in K_df.index else missing).append(name)

    if missing and not pad_missing:
        raise ValueError(
            f"{len(missing)} X_cols name(s) not found in connectome: {missing}. "
            f"Pass x_to_k to rename them if they exist under a different name, "
            f"or pad_missing=True to treat genuinely absent ones as disconnected "
            f"nodes."
        )

    n = len(X_cols)
    K = np.zeros((n, n), dtype=float)
    used_k_names = set()
    if found:
        k_names = [x_to_k.get(name, name) for name in found]
        sub = K_df.loc[k_names, k_names].to_numpy(dtype=float)
        idx = [X_cols.index(name) for name in found]
        K[np.ix_(idx, idx)] = sub
        used_k_names = set(k_names)
    np.fill_diagonal(K, 0.0)

    if missing:
        print(f"{len(missing)} biomarker(s) not found in connectome, "
              f"treated as disconnected nodes: {missing}")

    unused = [name for name in K_df.index if name not in used_k_names]
    if unused:
        print(f"{len(unused)} connectome region(s) unused (not requested in X_cols): "
              f"{unused}")

    return K, missing


# K, disconnected = build_connectome(
#     "connectome.csv",
#     X_cols=["putamen_left", "putamen_right", "MCATOT"],
#     x_to_k={
#         "putamen_left": "Left-Putamen",
#         "putamen_right": "Right-Putamen",
#         # MCATOT omitted not in connectome, and there's no map for it,
#         # so it auto-pads as a disconnected node (needs pad_missing=True)
#     },
#     pad_missing=True,
# )