"""Per-trial scores and fit diagnostics. Every function returns JSON-safe values."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .. import jax_core as jc
from ..identifiability import f_pull, kkt_report, lambda_f_max, pack_patients
from ..model_selection import gaussian_bic
from ..optimizer_jax import solve_trajectories_jax

ACTIVE = 0.01  # same threshold as count_bic_params
ZERO = 1e-8


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None  # JSON has no NaN; Optuna user attrs must be JSON
    return obj


def _trajectories(est) -> np.ndarray:
    return np.asarray(solve_trajectories_jax(est.cluster_f, est.final_kappa, est.final_scalar_K, est.K, est.t_span))


def model_x(est, patients, beta, subtype) -> np.ndarray:
    """Model x (in [0, 1]) at every visit, (N_rows, B)."""
    traj = _trajectories(est)
    rows = []
    for p, b, z in zip(patients, np.asarray(beta, float), np.asarray(subtype, int)):
        t = b + np.asarray(p["dt"], float)
        rows.append(np.stack([np.interp(t, est.t_span, traj[z, :, j]) for j in range(traj.shape[2])], axis=1))
    return np.vstack(rows)


def held_out_scores(est, X, groups: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Per-value MSE on ``X``: weighted (the fit's loss), unweighted and per block.

    Also returns the placement (``beta`` / ``subtype``) from the one ``transform``.
    """
    lse = float(-est.score(X))
    tr = np.zeros(len(X), dtype=[("beta", "f8"), ("subtype", "i4")])
    tr["beta"], tr["subtype"] = est.beta_val, est.transform_assignments
    sse, n_obs = est.sse_per_biomarker(X, transformed=tr)
    blocks = est.score_blocks(X, groups, transformed=tr)
    return {
        "lse": lse,
        "lse_unweighted": float(sse.sum() / max(int(n_obs.sum()), 1)),
        **{f"mse_{name}": float(v) for name, v in blocks.items()},
        "beta": tr["beta"].copy(),
        "subtype": tr["subtype"].copy(),
    }


def forecast_mse(est, X) -> float:
    """Place each patient from all but the last visit, then score the last visit."""
    hist, last = [], []
    for p in X:
        if len(p["dt"]) < 2:
            continue
        cut = {k: np.asarray(p[k])[:-1] for k in ("X_obs", "dt", "cog")}
        tail = {k: np.asarray(p[k])[-1:] for k in ("X_obs", "dt", "cog")}
        hist.append({**p, **cut})
        last.append({**p, **tail})
    if not hist:
        return float("nan")
    tr = est.transform(est._shift_patients(hist), _shifted=True)  # same placement as score()
    x = model_x(est, last, tr["beta"], tr["subtype"])
    c = est._final_c_or_zero(x.shape[1])
    resid = np.vstack([p["X_obs"] for p in last]) - (c + np.asarray(est.final_s, float) * x)
    ok = np.isfinite(resid)
    return float(np.sum(resid[ok] ** 2) / max(int(ok.sum()), 1))


def make_cv_scorer(groups: Dict[str, np.ndarray], *, forecast: bool):
    """Multi-metric scorer for ``cross_validate``: one ``transform`` per fold for the MSEs.

    CVIC is not here: the runner needs it per subject, not per fold.
    """
    def scorer(est, X, y=None):
        s = held_out_scores(est, X, groups)
        out = {k: v for k, v in s.items() if k not in ("beta", "subtype")}
        if forecast:
            out["forecast"] = forecast_mse(est, X)
        return out
    return scorer


def _packed(est, X, weights: Optional[np.ndarray]):
    packed = pack_patients(X, pad_rows=est.jax_pad_rows, pad_patients=est.jax_pad_patients)
    if weights is not None:  # the f-step's weighted loss
        data, meta = packed
        packed = (data._replace(W=data.W * jc.f64(np.sqrt(weights))[None, :]), meta)
    return packed


def fit_summary(est, X, groups: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Sparsity, KKT, s at bounds, x in [0, 1] coverage and BIC variants of a fitted model."""
    F = np.stack([np.ravel(f) for f in est.cluster_f]).astype(float)
    kappa = np.ravel(np.asarray(est.final_kappa, float))
    beta, z = est.final_beta_, np.asarray(est.final_assignments, int)
    s_par = np.ravel(est.final_s if getattr(est, "final_s_clip", None) is None else est.final_s_clip)
    weights = est.biomarker_weights
    packed = _packed(est, X, weights)
    if est._uses_shift():
        data, meta = packed
        packed = (jc.shift_data(data, jc.f64(np.asarray(est.final_c, float))), meta)
    pull = f_pull(packed, F, np.ravel(est.final_s), kappa, float(est.final_scalar_K), beta, z, est.K, est.t_span)
    n_obs_b = sum(np.isfinite(np.asarray(p["X_obs"], float)).sum(axis=0) for p in X)
    bic_gauss, _ = gaussian_bic(est._sse_per_biomarker, n_obs_b, int(est.bic_n_params_))
    x = model_x(est, X, beta, z)
    obs = np.vstack([np.isfinite(np.asarray(p["X_obs"], float)) for p in X])
    free = np.asarray(getattr(est, "s_mask_", np.ones(s_par.size, bool)), bool)
    lo, hi = est.s_lower_value, est.s_upper
    out: Dict[str, Any] = {
        "bic": float(est.bic_),
        "bic_gauss": float(bic_gauss),
        "lse_final": float(est.lse_final),
        "n_iter": int(est.n_iter_),
        "stop_reason": str(est.stop_reason_),
        "n_f_active": int(np.sum(F >= ACTIVE)),
        "n_f_zero": int(np.sum(F < ZERO)),
        "n_kappa_active": int(np.sum(kappa >= ACTIVE)),
        "final_scalar_K": float(est.final_scalar_K),
        "subtype_counts": np.bincount(z, minlength=int(est.n_subtypes)).tolist(),
        "beta_near_tmax_frac": float(np.mean(beta >= est.t_max - 0.5)),
        "f_kkt": kkt_report(F, float(est.lambda_f), pull),
        "s_param": s_par.tolist(),
    }
    for name, idx in {"all": np.arange(s_par.size), **groups}.items():
        v = s_par[idx][free[idx]]
        out[f"s_median_{name}"] = float(np.median(v)) if v.size else float("nan")
        out[f"s_frac_at_lower_{name}"] = float(np.mean(v <= lo + 1e-4)) if v.size and lo is not None else float("nan")
        out[f"s_frac_at_upper_{name}"] = float(np.mean(v >= hi - 1e-4)) if v.size and hi is not None else float("nan")
        xv = x[:, idx][obs[:, idx]]
        out[f"x_frac_below_half_{name}"] = float(np.mean(xv < 0.5)) if xv.size else float("nan")
        out[f"x_frac_above_half_{name}"] = float(np.mean(xv > 0.5)) if xv.size else float("nan")
    if getattr(est, "restarts_", None):
        out["restarts"] = est.restarts_
        out["chosen_restart"] = est.chosen_restart_
        out["failed_all_restarts"] = est.failed_all_restarts_
    return out


def lambda_f_max_report(est, X, search_range: List[float], seed: int) -> Dict[str, Any]:
    """lasso lambda_max for f at kappa = 0, scalar_K = 0.25, the MixedLM beta and random z.

    s = 1 for the scale model; zsym uses the starting clipping level s0 = 2.
    """
    B = np.asarray(est.K).shape[0]
    t_span = np.linspace(0.0, est.t_max, int(est.t_max / est.step))
    z = np.random.default_rng(seed).integers(0, est.n_subtypes, size=len(X))
    beta = np.array([p["initial_beta"] for p in X], dtype=float)
    packed = _packed(est, X, est.biomarker_weights)
    s = np.ones(B)
    if est._is_zsym():
        data, meta = packed
        packed, s = (jc.shift_data(data, jc.f64(np.full(B, -2.0))), meta), np.full(B, 4.0)
    lam, pull0 = lambda_f_max(packed, s, np.zeros(B), 0.25, beta, z, est.K, t_span, est.n_subtypes)
    lo, hi = search_range
    return {"lambda_f_max": float(lam), "search_range_over_lambda_f_max": [lo / lam, hi / lam],
            "pull0_quantiles": {q: float(np.quantile(pull0, float(q))) for q in ("0.1", "0.5", "0.9")}}
