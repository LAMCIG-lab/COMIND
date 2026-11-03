"""Write ``tests/data/golden_scale.npz`` from the current package.

Run once on the commit before the zsym / biomarker-weight changes:

    python tests/make_golden_scale.py

``test_zsym_weights.py::test_scale_matches_golden`` then checks that the
default ``obs_model="scale"`` with no weights reproduces these outputs exactly.
"""
import os
import subprocess

import numpy as np

from COMIND_transformer import jax_core as jc
from COMIND_transformer.subtyping_em_transformer import SubtypingEM

OUT = os.path.join(os.path.dirname(__file__), "data", "golden_scale.npz")


def golden_toy(seed=0, n_pat=40, B=4, t_max=20.0, step=0.05):
    rng = np.random.default_rng(seed)
    K = np.zeros((B, B)); K[1, 0] = K[0, 1] = 1.0; K[2, 1] = K[1, 2] = 1.0
    F = np.zeros((2, B)); F[0, 0] = 0.15; F[1, 3] = 0.15
    t_span = np.linspace(0, t_max, int(t_max / step))
    _, h, n = jc.grid_from_t_span(t_span)
    tr = np.asarray(jc._traj_jit(jc.f64(F), jc.f64(np.zeros(B)), jc.f64(0.2), jc.f64(K),
                                 jc.f64(h), n_steps=n))
    pats = []
    for i in range(n_pat):
        z = i % 2; b0 = rng.uniform(1, 12); dt = np.array([0.0, 1.0, 2.0])
        mean = np.array([[np.interp(b0 + d, t_span, tr[z, :, b]) for b in range(B)] for d in dt])
        X = 0.8 * mean + rng.normal(0, 0.05, mean.shape)
        if i % 7 == 0:
            X[1, 2] = np.nan
        pats.append({"id": i, "X_obs": X, "dt": dt, "cog": np.zeros((3, 1)),
                     "initial_beta": b0})
    return K, pats, t_max, step


def golden_em(K, t_max, step, **kw):
    B = K.shape[0]
    args = dict(K=K, initial_f=np.full(B, 0.02), initial_s=np.full(B, 0.9),
                initial_scalar_K=0.1, initial_kappa=np.zeros(B), n_subtypes=2, max_iter=6,
                t_max=t_max, step=step, fit_s=True, s_lower_value=0.2, s_upper=1.0,
                verbose=0, backend="jax", transform_backend="jax",
                rng=np.random.default_rng(0))
    args.update(kw)
    return SubtypingEM(**args)


def golden_outputs():
    K, pats, t_max, step = golden_toy()
    train, val = pats[:30], pats[30:]
    em = golden_em(K, t_max, step).fit(train)
    tr = em.transform(val)
    return dict(
        final_s=np.asarray(em.final_s), final_kappa=np.asarray(em.final_kappa),
        final_scalar_K=np.float64(em.final_scalar_K),
        cluster_f=np.array([np.ravel(f) for f in em.cluster_f]),
        beta=np.asarray(em.beta_history)[:, -1], assignments=np.asarray(em.final_assignments),
        lse_history=np.asarray(em.lse_history), bic=np.float64(em.bic_),
        val_beta=tr["beta"], val_subtype=tr["subtype"],
        score=np.float64(em.score(val)), score_total=np.float64(em.score(val, per_value=False)),
    )


if __name__ == "__main__":
    out = golden_outputs()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, commit=commit, **out)
    print(f"wrote {OUT} at {commit}")
    for k, v in out.items():
        print(k, np.round(v, 6) if np.ndim(v) < 2 else np.shape(v))
