"""Load fitted EM state from experiment .npz checkpoints for warm-start fitting."""
from __future__ import annotations

from typing import Sequence, Union

import numpy as np

PathLike = Union[str, "np.lib.npyio.NpzFile"]


def _align_by_patient_id(
    saved_ids: np.ndarray,
    saved_values: np.ndarray,
    patient_ids: Sequence,
    *,
    name: str,
) -> np.ndarray:
    """Map per-patient arrays from a saved training cohort onto ``patient_ids`` order."""
    saved_ids = np.asarray(saved_ids)
    patient_ids = list(patient_ids)
    lookup = {pid: i for i, pid in enumerate(saved_ids)}
    if saved_values.ndim == 1:
        out = np.empty(len(patient_ids), dtype=saved_values.dtype)
        for i, pid in enumerate(patient_ids):
            if pid not in lookup:
                raise KeyError(
                    f"Patient id {pid!r} not in saved {name} (n_saved={len(saved_ids)})."
                )
            out[i] = saved_values[lookup[pid]]
        return out
    raise ValueError(f"Expected 1D saved {name}, got shape {saved_values.shape}")


def load_warm_start_from_npz(
    npz_path: PathLike,
    patient_ids: Sequence,
    *,
    ids_key: str = "train_ids",
    assignments_key: str = "train_assignments",
    beta_history_key: str = "beta_history",
    beta_column: int = -1,
    cluster_f_key: str = "cluster_f",
    cog_a_key: str = "cluster_cog_a",
    cog_b_key: str = "cluster_cog_b",
    s_key: str = "final_s",
    scalar_K_key: str = "final_scalar_K",
    kappa_key: str = "final_kappa",
) -> dict:
    """
    Build a warm-start payload for :class:`~COMIND_transformer.subtyping_em_transformer.SubtypingEM`.

    Parameters
    ----------
    npz_path : str or npz file
        Checkpoint written by ``run_comind_ppmi.py`` (or compatible).
    patient_ids : sequence
        Patient ids in the **same order** as the list passed to ``fit()`` (e.g. ``[p['id'] for p in X_train]``).
    ids_key, assignments_key, ...
        Keys in the archive; override if your file uses different names.

    Returns
    -------
    dict
        Keys suitable for ``SubtypingEM`` constructor: ``initial_f``, ``initial_s``,
        ``initial_scalar_K``, ``initial_kappa``, ``initial_assignments``,
        ``initial_cluster_cog_a``, ``initial_cluster_cog_b``, ``initial_beta``.
    """
    if not isinstance(npz_path, np.lib.npyio.NpzFile):
        data = np.load(npz_path, allow_pickle=True)
        close_after = True
    else:
        data = npz_path
        close_after = False

    try:
        if ids_key not in data.files:
            raise KeyError(f"{ids_key!r} not in npz; keys={data.files}")

        saved_ids = np.asarray(data[ids_key])
        cluster_f = np.asarray(data[cluster_f_key], dtype=float)
        cluster_cog_a = np.asarray(data[cog_a_key], dtype=float)
        cluster_cog_b = np.asarray(data[cog_b_key], dtype=float)
        s = np.asarray(data[s_key], dtype=float).ravel()
        scalar_K = float(np.asarray(data[scalar_K_key]).ravel()[0])

        if kappa_key in data.files:
            kappa = np.asarray(data[kappa_key], dtype=float).ravel()
        else:
            kappa = np.zeros(s.shape[0], dtype=float)

        assignments = _align_by_patient_id(
            saved_ids,
            np.asarray(data[assignments_key]),
            patient_ids,
            name=assignments_key,
        )

        if beta_history_key in data.files:
            beta_hist = np.asarray(data[beta_history_key], dtype=float)
            beta_slice = beta_hist[:, beta_column]
            beta = _align_by_patient_id(
                saved_ids, beta_slice, patient_ids, name=beta_history_key
            )
        else:
            raise KeyError(
                f"{beta_history_key!r} not in npz; cannot warm-start beta."
            )

        n_subtypes = cluster_f.shape[0]
        if cluster_cog_a.shape[0] != n_subtypes:
            raise ValueError(
                f"cluster_f has {n_subtypes} subtypes but cluster_cog_a has shape {cluster_cog_a.shape}"
            )

        return {
            "initial_f": cluster_f,
            "initial_s": s,
            "initial_scalar_K": scalar_K,
            "initial_kappa": kappa,
            "initial_assignments": assignments.astype(int, copy=False),
            "initial_cluster_cog_a": cluster_cog_a,
            "initial_cluster_cog_b": np.asarray(cluster_cog_b, dtype=float).ravel(),
            "initial_beta": beta.astype(float, copy=False),
            "n_subtypes": n_subtypes,
        }
    finally:
        if close_after:
            data.close()


def load_legacy_presubtyping_from_npz(
    npz_path: PathLike,
    patient_ids: Sequence,
    *,
    n_biomarkers: int = 68,
    theta_history_key: str = "theta_history",
    beta_history_key: str = "beta_history",
    cog_history_key: str = "cog_history",
    theta_column: int = -1,
    beta_column: int = -1,
) -> dict:
    """
    Warm-start from pre-subtyping COMIND checkpoints (Z=1, no kappa in file).

    ``theta_history`` final column is ``[f (n), s (n), scalar_K (1)]``.
    ``cog_history`` final column is ``[cog_a (n_cog), cog_b (1)]``.
    ``beta_history`` rows must align with ``patient_ids`` order (same train split).
    """
    if not isinstance(npz_path, np.lib.npyio.NpzFile):
        data = np.load(npz_path, allow_pickle=True)
        close_after = True
    else:
        data = npz_path
        close_after = False

    try:
        theta = np.asarray(data[theta_history_key], dtype=float)[:, theta_column].ravel()
        expected = 2 * n_biomarkers + 1
        if theta.size != expected:
            raise ValueError(
                f"theta column length {theta.size} != 2*n_biomarkers+1 ({expected})"
            )
        f = theta[:n_biomarkers]
        s = theta[n_biomarkers : 2 * n_biomarkers]
        scalar_K = float(theta[2 * n_biomarkers])

        cog_hist = np.asarray(data[cog_history_key], dtype=float)
        if cog_hist.ndim != 2 or cog_hist.shape[0] < 2:
            raise ValueError(f"unexpected cog_history shape {cog_hist.shape}")
        cog_vec = cog_hist[:, theta_column]
        # cog_history: (n_cog + 1, T) — coefficients then intercept
        n_cog = cog_hist.shape[0] - 1
        cog_a = cog_vec[:n_cog]
        cog_b = float(cog_vec[n_cog])

        beta_hist = np.asarray(data[beta_history_key], dtype=float)
        if beta_hist.ndim != 2:
            raise ValueError(f"beta_history must be 2-D, got {beta_hist.shape}")
        if beta_hist.shape[0] != len(patient_ids):
            raise ValueError(
                f"beta_history rows {beta_hist.shape[0]} != len(patient_ids) "
                f"({len(patient_ids)}); ensure the same train split/order."
            )
        beta = beta_hist[:, beta_column].astype(float, copy=False)

        n_subtypes = 1
        return {
            "initial_f": f.reshape(1, n_biomarkers),
            "initial_s": s,
            "initial_scalar_K": scalar_K,
            "initial_kappa": np.zeros(n_biomarkers, dtype=float),
            "initial_assignments": np.zeros(len(patient_ids), dtype=int),
            "initial_cluster_cog_a": cog_a.reshape(1, n_cog),
            "initial_cluster_cog_b": np.array([cog_b], dtype=float),
            "initial_beta": beta,
            "n_subtypes": n_subtypes,
        }
    finally:
        if close_after:
            data.close()
