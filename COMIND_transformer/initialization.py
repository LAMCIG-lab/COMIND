"""
Initialization helpers for SubtypingEM.

Restart 0 is data-driven (k-means on each subject's imaging atrophy pattern).
Other restarts draw a small random f, different per subtype, so the hard E-step
does not send every patient to subtype 0. Network terms start off (kappa = 0)
except an optional fourth restart with kappa = 0.1.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from sklearn.cluster import KMeans


F_DATA_DRIVEN_LO = 0.002
F_DATA_DRIVEN_SPAN = 0.028  # 0.002 + 0.028 = 0.03 at the peak biomarker
F_RANDOM_LO = 0.005
F_RANDOM_HI = 0.02
FAILED_INIT_RATIO = 1.0
FAILED_SUBTYPE_FRAC = 0.02
FAILED_SUBTYPE_ITER = 2


def _patient_mean_profile(
    patient: dict, c0: np.ndarray, zsym_s0: Optional[float] = None
) -> np.ndarray:
    """Visit-mean of ``max(X - c0, 0)``, NaNs treated as missing.

    With ``zsym_s0``: visit-mean of the model-space value
    ``z_t = clip((z + s0) / (2 s0), 0, 1)`` instead (``c0`` unused).
    """
    X = np.asarray(patient["X_obs"], dtype=float)
    if zsym_s0 is not None:
        s0 = float(zsym_s0)
        resid = np.clip((X + s0) / (2.0 * s0), 0.0, 1.0)
    else:
        resid = np.maximum(X - np.asarray(c0, dtype=float).ravel()[None, :], 0.0)
    with np.errstate(all="ignore"):
        profile = np.nanmean(resid, axis=0)
    return np.where(np.isfinite(profile), profile, 0.0)


def _ensure_all_subtypes(z0: np.ndarray, Z: int, rng: np.random.Generator) -> np.ndarray:
    """Move a patient from the largest cluster into any empty subtype."""
    z0 = np.asarray(z0, dtype=int).copy()
    n = z0.size
    if n == 0:
        return z0
    for z in range(Z):
        if np.any(z0 == z):
            continue
        counts = np.bincount(z0, minlength=Z)
        donor = int(np.argmax(counts))
        donors = np.flatnonzero(z0 == donor)
        z0[int(rng.choice(donors))] = z
    return z0


def data_driven_f_and_assignments(
    patients: Sequence[dict],
    Z: int,
    c0: np.ndarray,
    rng: np.random.Generator,
    pattern_cols: Optional[Sequence[int]] = None,
    zsym_s0: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    k-means on each subject's normalized imaging atrophy pattern.

    Returns
    -------
    f0 : (Z, B)
        ``0.002 + 0.028 * (m_z / max m_z)**2``, all entries > 0, rows differ.
    z0 : (n,)
        Starting assignments; every subtype is non-empty when ``n >= Z``.
    """
    c0 = np.asarray(c0, dtype=float).ravel()
    B = c0.size
    n = len(patients)
    Z = int(Z)
    if n == 0:
        raise ValueError("data_driven_f_and_assignments needs at least one patient")
    profiles = np.vstack([_patient_mean_profile(p, c0, zsym_s0) for p in patients])
    cols = (
        np.arange(B)
        if pattern_cols is None
        else np.asarray(pattern_cols, dtype=int).ravel()
    )
    patterns = profiles[:, cols].copy()
    norms = np.linalg.norm(patterns, axis=1, keepdims=True)
    nonzero = norms[:, 0] > 0
    patterns[nonzero] /= norms[nonzero]
    n_clusters = int(min(max(Z, 1), n))
    seed = int(rng.integers(0, 2**31 - 1))
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    z0 = km.fit_predict(patterns).astype(int)
    z0 = _ensure_all_subtypes(z0, Z, rng)

    f0 = np.full((Z, B), F_DATA_DRIVEN_LO, dtype=float)
    for z in range(Z):
        members = profiles[z0 == z]
        if members.size == 0:
            continue
        m_z = members.mean(axis=0)
        peak = float(np.max(m_z))
        if peak > 0:
            f0[z] = F_DATA_DRIVEN_LO + F_DATA_DRIVEN_SPAN * (m_z / peak) ** 2
        else:
            f0[z] = F_DATA_DRIVEN_LO + 1e-4 * (z + 1)
    # Never allow identical subtype rows (hard E-step would tie).
    if Z > 1 and np.any(np.all(np.abs(f0[1:] - f0[0]) < 1e-15, axis=1)):
        f0 = f0 + rng.uniform(1e-5, 3e-5, size=f0.shape)
    return f0, z0


def random_f_and_assignments(
    n_patients: int,
    B: int,
    Z: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Small random f in ``[0.005, 0.02]``, different rows, random assignments."""
    f0 = rng.uniform(F_RANDOM_LO, F_RANDOM_HI, size=(int(Z), int(B)))
    for z in range(1, int(Z)):
        if np.allclose(f0[z], f0[0]):
            f0[z] = np.clip(f0[z] + 1e-4, F_RANDOM_LO, F_RANDOM_HI)
    z0 = rng.integers(0, int(Z), size=int(n_patients))
    z0 = _ensure_all_subtypes(z0, int(Z), rng)
    return f0, z0


def null_sse(
    patients: Sequence[dict], c0: np.ndarray, weights: Optional[np.ndarray] = None
) -> float:
    """``sum(w_b (X - c0)**2)`` over finite entries: the "no disease" SSE.

    For z-score inputs pass ``c0 = 0``: the SSE of predicting the healthy mean.
    ``weights`` are the fit's biomarker weights (None = ones), so the ratio
    compares like with like.
    """
    c0 = np.asarray(c0, dtype=float).ravel()
    w = np.ones_like(c0) if weights is None else np.asarray(weights, dtype=float).ravel()
    total = 0.0
    for p in patients:
        resid = np.asarray(p["X_obs"], dtype=float) - c0[None, :]
        total += float(np.nansum(w[None, :] * resid ** 2))
    return total


def init_ratio(
    em, patients: Sequence[dict], c0: np.ndarray, weights: Optional[np.ndarray] = None
) -> float:
    """``lse_history[0] / null_sse``. Above 1 means the start over-predicts disease."""
    denom = null_sse(patients, c0, weights)
    lse0 = float(np.asarray(em.lse_history).ravel()[0])
    if denom <= 0:
        return float("inf") if lse0 > 0 else 0.0
    return lse0 / denom


def initial_values(
    restart: int,
    x_train: Sequence[dict],
    Z: int,
    c0: np.ndarray,
    imaging_idx: Sequence[int],
    scalar_K_center: float,
    seed: int,
    init_scalar_k_cap: Optional[float] = None,
    zsym_s0: Optional[float] = None,
) -> Dict:
    """Per-restart starting f, assignments, kappa, scalar_K, s and c.

    ``zsym_s0``: obs_model="zsym". Restart 0 builds its patterns from
    ``clip((z + s0) / (2 s0), 0, 1)`` and every restart starts s at ``s0``.
    """
    rng_r = np.random.default_rng(int(seed) + 1000 * int(restart))
    c0 = np.asarray(c0, dtype=float).ravel()
    B = c0.size
    if restart == 0:
        f0, z0 = data_driven_f_and_assignments(
            x_train, Z, c0, rng_r, pattern_cols=imaging_idx, zsym_s0=zsym_s0
        )
        kappa0 = np.zeros(B)
    elif restart == 3:
        f0, z0 = random_f_and_assignments(len(x_train), B, Z, rng_r)
        kappa0 = np.full(B, 0.1)
    else:
        f0, z0 = random_f_and_assignments(len(x_train), B, Z, rng_r)
        kappa0 = np.zeros(B)
    initial_scalar_K = float(scalar_K_center)
    if init_scalar_k_cap is not None:
        initial_scalar_K = min(initial_scalar_K, float(init_scalar_k_cap))
    return dict(
        initial_f=f0,
        initial_assignments=z0,
        initial_kappa=kappa0,
        initial_scalar_K=initial_scalar_K,
        initial_s=np.ones(B) if zsym_s0 is None else np.full(B, float(zsym_s0)),
        initial_c=c0.copy(),
        rng=rng_r,
    )


def restart_is_failed(
    em,
    *,
    init_ratio_value: float,
    n_patients: int,
    init_ratio_max: float = FAILED_INIT_RATIO,
    min_frac: float = FAILED_SUBTYPE_FRAC,
    at_iter: int = FAILED_SUBTYPE_ITER,
) -> bool:
    """True if the start over-predicts or a subtype is empty after iteration 2."""
    if not np.isfinite(init_ratio_value) or init_ratio_value > init_ratio_max:
        return True
    counts = np.asarray(getattr(em, "subtype_counts_history_", []), dtype=float)
    if counts.ndim != 2 or counts.size == 0 or n_patients <= 0:
        return False
    col = counts[:, at_iter] if counts.shape[1] > at_iter else counts[:, -1]
    return bool(np.any(col < min_frac * n_patients))


def penalized_objective(
    em,
    params: dict,
    *,
    lambda_c: float = 0.0,
    c_center: Optional[np.ndarray] = None,
) -> float:
    """LSE plus the same penalties the globals / f steps use."""
    F = np.stack([np.ravel(f) for f in em.cluster_f]).astype(float)
    kappa = np.ravel(np.asarray(em.final_kappa, dtype=float))
    scalar_K = max(float(em.final_scalar_K), 1e-12)
    center = max(float(params["scalar_K_center"]), 1e-12)
    obj = (
        float(em.lse_final)
        + float(params["lambda_f"]) * float(np.sum(F))
        + float(params["lambda_kappa"]) * float(np.sum(kappa))
        + 0.5 * float(params["lambda_scalar"]) * (np.log(scalar_K) - np.log(center)) ** 2
    )
    c = np.asarray(getattr(em, "final_c", np.zeros_like(kappa)), dtype=float)
    if c_center is None:
        c_center = np.zeros_like(c)
    obj += float(lambda_c) * float(np.sum((c - np.asarray(c_center, dtype=float)) ** 2))
    return float(obj)


def select_restart(records: Sequence[dict]) -> Tuple[dict, bool]:
    """Lowest penalized objective among non-failed restarts; all-failed still returns a best."""
    if not records:
        raise ValueError("select_restart needs at least one restart record")
    ok = [r for r in records if not r.get("failed", False)]
    pool = ok if ok else list(records)
    chosen = min(pool, key=lambda r: (float(r["objective"]), int(r["restart"])))
    return chosen, len(ok) == 0
