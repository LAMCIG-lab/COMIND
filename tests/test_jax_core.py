"""Tests for the JAX backend (``jax_core``, ``optimizer_jax`` and ``SubtypingEM(backend="jax")``)."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from scipy.integrate import solve_ivp  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402

from COMIND_transformer import jax_core as jc  # noqa: E402
from COMIND_transformer import optimizer_jax as oj  # noqa: E402
from COMIND_transformer.kernel_jsd_multi import KernelJSDMulti  # noqa: E402
from COMIND_transformer.optimizer_beta import _vectorized_beta_loss_and_grad  # noqa: E402
from COMIND_transformer.optimizer_theta_cluster import theta_cluster_loss  # noqa: E402
from COMIND_transformer.optimizer_theta_globals import theta_s_loss_multi  # noqa: E402
from COMIND_transformer.subtyping_em_transformer import SubtypingEM  # noqa: E402
from COMIND_transformer.utils import solve_system, split_observed  # noqa: E402

T_MAX = 12.0
STEP = 0.1
FD_STEP = 1e-6


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def make_K(n_biomarkers: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    K = rng.random((n_biomarkers, n_biomarkers))
    np.fill_diagonal(K, 0.0)
    return K / K.sum(axis=1, keepdims=True)


def make_problem(
    seed: int = 0,
    n_biomarkers: int = 4,
    n_patients: int = 40,
    n_subtypes: int = 3,
    nan_frac: float = 0.1,
    n_cog: int = 2,
):
    """Random stacked data plus random parameters at an interior point."""
    rng = np.random.default_rng(seed)
    K = make_K(n_biomarkers, seed)
    visits = rng.integers(2, 5, size=n_patients)
    ids = np.repeat(np.arange(n_patients), visits)
    dt = np.concatenate([np.arange(v, dtype=float) for v in visits])
    n_rows = len(ids)
    X_raw = rng.uniform(0.05, 0.9, size=(n_rows, n_biomarkers))
    nan_idx = rng.choice(X_raw.size, size=int(nan_frac * X_raw.size), replace=False)
    X_raw.ravel()[nan_idx] = np.nan
    X_obs, obs_weight = split_observed(X_raw)
    cog = rng.normal(size=(n_rows, n_cog))
    return dict(
        K=K,
        ids=ids,
        dt=dt,
        X_raw=X_raw,
        X_obs=X_obs,
        obs_weight=obs_weight,
        cog=cog,
        n_patients=n_patients,
        n_subtypes=n_subtypes,
        beta=rng.uniform(0.5, 8.0, size=n_patients),
        assignments=rng.integers(0, n_subtypes, size=n_patients),
        s=rng.uniform(0.5, 1.5, size=n_biomarkers),
        kappa=rng.uniform(0.05, 0.8, size=n_biomarkers),
        scalar_K=0.7,
        F=rng.uniform(0.02, 0.3, size=(n_subtypes, n_biomarkers)),
        cog_a=rng.normal(size=(n_subtypes, n_cog)),
        cog_b=rng.normal(size=n_subtypes),
    )


def lsoda_trajectories(F, K, t_span, scalar_K, kappa):
    """Per-subtype (B, T) LSODA trajectories, as the NumPy path solves them."""
    return [
        solve_system(np.zeros(K.shape[0]), np.ravel(f), K, t_span, scalar_K, kappa)
        for f in F
    ]


def central_fd(fun, x: np.ndarray, step: float = FD_STEP) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    for i in range(x.size):
        e = np.zeros_like(x)
        e.flat[i] = step
        g.flat[i] = (fun(x + e) - fun(x - e)) / (2.0 * step)
    return g


def assert_grad_close(g, g_fd, rtol=1e-5):
    g = np.asarray(g, dtype=float)
    scale = max(np.max(np.abs(g_fd)), 1e-12)
    np.testing.assert_allclose(g, g_fd, rtol=rtol, atol=rtol * scale)


def grid():
    t_span, h, n_steps = jc.make_grid(T_MAX, STEP)
    return t_span, jc.f64(h), n_steps


def pack(prob, **kw):
    return jc.pack_data(
        prob["X_raw"], None, prob["dt"], prob["ids"], prob["cog"], prob["n_patients"], **kw
    )


# ---------------------------------------------------------------------------
# 1. Trajectories
# ---------------------------------------------------------------------------


def _tight_reference(f, K, t_span, scalar_K, kappa):
    K_eff = scalar_K * K + np.diag(kappa)

    def rhs(t, x):
        xc = np.clip(x, 0.0, 1.0 - 1e-9)
        return (1.0 - xc) * (K_eff @ xc + f)

    return solve_ivp(
        rhs, [0.0, t_span[-1]], np.zeros(K.shape[0]), t_eval=t_span,
        method="DOP853", rtol=1e-12, atol=1e-14,
    ).y


@pytest.mark.parametrize("scalar_K", [0.1, 1.0, 5.0])
def test_trajectory_matches_lsoda(scalar_K):
    """RK4 vs ``utils.solve_system`` (LSODA) and vs a tight-tolerance reference.

    ``solve_system`` uses solve_ivp's default ``rtol=1e-3``, so LSODA itself is
    only accurate to ~4e-4 here; the 1e-3 bound is set by LSODA, not RK4. The
    tight DOP853 reference measures the RK4 error itself.
    """
    rng = np.random.default_rng(int(scalar_K * 10))
    B = 5
    K = make_K(B, seed=1)
    t_span, h, n_steps = grid()
    kappa = rng.uniform(0.0, 1.0, B)
    F = rng.uniform(0.0, 0.3, size=(2, B))

    traj = np.asarray(
        jc._traj_jit(jc.f64(F), jc.f64(kappa), jc.f64(scalar_K), jc.f64(K), h, n_steps=n_steps)
    )
    assert traj.shape == (2, len(t_span), B)
    for z in range(2):
        lsoda = solve_system(np.zeros(B), F[z], K, t_span, scalar_K, kappa)
        ref = _tight_reference(F[z], K, t_span, scalar_K, kappa)
        assert np.max(np.abs(traj[z].T - lsoda)) < 1e-3
        rk4_tol = 1e-4 if scalar_K <= 1.0 else 5e-4
        assert np.max(np.abs(traj[z].T - ref)) < rk4_tol


def test_trajectory_stiff_report(capsys):
    """Report (not assert) the RK4 error in a stiff regime."""
    rng = np.random.default_rng(20)
    B = 5
    K = make_K(B, seed=1)
    t_span, h, n_steps = grid()
    kappa = rng.uniform(0.0, 5.0, B)
    f = rng.uniform(0.0, 0.3, B)
    traj = np.asarray(
        jc._traj_jit(jc.f64(f[None]), jc.f64(kappa), jc.f64(20.0), jc.f64(K), h, n_steps=n_steps)
    )[0].T
    lsoda = solve_system(np.zeros(B), f, K, t_span, 20.0, kappa)
    ref = _tight_reference(f, K, t_span, 20.0, kappa)
    with capsys.disabled():
        print(
            f"\n[stiff] scalar_K=20, kappa<=5, step={STEP}: "
            f"max|RK4-LSODA|={np.max(np.abs(traj - lsoda)):.3e}, "
            f"max|RK4-exact|={np.max(np.abs(traj - ref)):.3e}"
        )


# ---------------------------------------------------------------------------
# 2. Interpolation
# ---------------------------------------------------------------------------


def test_interp_rows_matches_np_interp():
    rng = np.random.default_rng(2)
    t_span, h, _ = grid()
    Z, T, B = 3, len(t_span), 4
    traj = rng.normal(size=(Z, T, B))
    t = np.concatenate([
        rng.uniform(-3.0, T_MAX + 3.0, size=200),
        [-1.0, 0.0, t_span[1], t_span[-2], T_MAX, T_MAX + 1.0],
    ])
    z = rng.integers(0, Z, size=len(t))
    out = np.asarray(jc.interp_rows(jc.f64(traj), jc.f64(t), jc.i32(z), h))
    ref = np.array([[np.interp(t[r], t_span, traj[z[r], :, b]) for b in range(B)]
                    for r in range(len(t))])
    np.testing.assert_allclose(out, ref, rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# 3. Loss equivalence given identical trajectories
# ---------------------------------------------------------------------------


@pytest.fixture
def fixed_traj(monkeypatch):
    """Make ``jax_core.solve_trajectories`` return given (Z, T, B) trajectories."""
    holder = {}

    def fake(F, kappa, scalar_K, K, h, n_steps):
        return holder["traj"]

    monkeypatch.setattr(jc, "solve_trajectories", fake)
    return holder


def test_globals_loss_matches_numpy(fixed_traj):
    prob = make_problem(seed=3)
    t_span, h, n_steps = grid()
    data, meta = pack(prob)
    P = meta["P"]
    lam = dict(lambda_s=0.3, lambda_scalar=0.7, lambda_kappa=0.2, scalar_K_center=0.4)
    s_mask = np.array([True, False, True, True])

    X_preds = lsoda_trajectories(prob["F"], prob["K"], t_span, prob["scalar_K"], prob["kappa"])
    fixed_traj["traj"] = jc.f64(np.stack([x.T for x in X_preds]))

    s = prob["s"].copy()
    s[~s_mask] = 1.0
    params = np.concatenate([s[s_mask], prob["kappa"], [prob["scalar_K"]]])
    ref = theta_s_loss_multi(
        params, prob["dt"] + prob["beta"][prob["ids"]], prob["X_obs"], prob["K"], t_span,
        list(prob["F"]), prob["assignments"][prob["ids"]],
        lambda_s=lam["lambda_s"], lambda_scalar=lam["lambda_scalar"],
        lambda_kappa=lam["lambda_kappa"], scalar_K_center=lam["scalar_K_center"],
        obs_weight=prob["obs_weight"], s_mask=s_mask,
    )
    out = jc.globals_loss(
        jc.f64(s), jc.f64(prob["kappa"]), jc.f64(prob["scalar_K"]), jc.f64(prob["F"]),
        jc.pad_beta(prob["beta"], P), jc.pad_assignments(prob["assignments"], P),
        data, jc.f64(prob["K"]), jc.make_hyp(**lam), h, n_steps,
        s_free=jc.f64(s_mask.astype(float)),
    )
    np.testing.assert_allclose(float(out), ref, rtol=1e-9)


def test_cluster_loss_matches_numpy(fixed_traj):
    prob = make_problem(seed=4)
    prob["assignments"][prob["assignments"] == 2] = 0  # subtype 2 empty
    t_span, h, n_steps = grid()
    data, meta = pack(prob)
    P = meta["P"]
    lambda_f = 0.05

    X_preds = lsoda_trajectories(prob["F"], prob["K"], t_span, prob["scalar_K"], prob["kappa"])
    fixed_traj["traj"] = jc.f64(np.stack([x.T for x in X_preds]))

    ref = 0.0
    for z in range(prob["n_subtypes"]):
        pids = np.where(prob["assignments"] == z)[0]
        if len(pids) == 0:
            ref += lambda_f * np.sum(prob["F"][z])  # constant in the joint loss
            continue
        rows = np.isin(prob["ids"], pids)
        t_obs = prob["dt"][rows] + prob["beta"][prob["ids"][rows]]
        ref += theta_cluster_loss(
            prob["F"][z], t_obs, prob["X_obs"][rows], prob["K"], t_span, prob["s"],
            prob["scalar_K"], lambda_f, kappa=prob["kappa"],
            obs_weight=prob["obs_weight"][rows],
        )
    out = jc.cluster_loss(
        jc.f64(prob["F"]), jc.f64(prob["s"]), jc.f64(prob["kappa"]), jc.f64(prob["scalar_K"]),
        jc.pad_beta(prob["beta"], P), jc.pad_assignments(prob["assignments"], P),
        data, jc.f64(prob["K"]), jc.make_hyp(lambda_f=lambda_f), h, n_steps,
    )
    np.testing.assert_allclose(float(out), ref, rtol=1e-9)


@pytest.mark.parametrize(
    "lambda_cog,lambda_beta,lambda_jsd",
    [(0.0, 0.0, 0.0), (0.3, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 2.0), (0.3, 0.5, 2.0)],
)
def test_beta_loss_matches_numpy(lambda_cog, lambda_beta, lambda_jsd):
    prob = make_problem(seed=5)
    t_span, h, _ = grid()
    data, meta = pack(prob)
    P = meta["P"]
    beta_mean, beta_var = 4.0, 3.5

    X_preds = lsoda_trajectories(prob["F"], prob["K"], t_span, prob["scalar_K"], prob["kappa"])
    Z = prob["n_subtypes"]
    ref, _ = _vectorized_beta_loss_and_grad(
        prob["beta"], prob["X_obs"], prob["dt"], prob["ids"], prob["cog"], t_span,
        list(prob["F"]), prob["scalar_K"], prob["s"], prob["assignments"], prob["K"],
        list(prob["cog_a"]), list(prob["cog_b"]), lambda_cog, lambda_jsd, T_MAX,
        {z: X_preds[z] for z in range(Z)}, {z: np.zeros_like(X_preds[z]) for z in range(Z)},
        lambda_beta=lambda_beta, beta_mean=beta_mean, beta_var=beta_var,
        obs_weight=prob["obs_weight"],
    )
    n_bins = max(20, int(T_MAX * 2))
    out = jc.beta_loss(
        jc.pad_beta(prob["beta"], P), jc.f64(np.stack([x.T for x in X_preds])),
        jc.f64(prob["s"]), jc.pad_assignments(prob["assignments"], P),
        jc.f64(prob["cog_a"]), jc.f64(prob["cog_b"]), data,
        jc.make_hyp(lambda_cog=lambda_cog, lambda_beta=lambda_beta, beta_mean=beta_mean,
                    beta_var=beta_var, lambda_jsd=lambda_jsd),
        h, jc.f64(0.0), jc.f64(T_MAX), jc.f64(1.0),
        n_bins=n_bins, use_jsd=lambda_jsd > 0, auto_bandwidth=True,
    )
    np.testing.assert_allclose(float(out), ref, rtol=1e-9)


def test_sse_matrix_matches_numpy():
    from COMIND_transformer.assignments import sse_matrix

    prob = make_problem(seed=6)
    t_span, h, _ = grid()
    data, meta = pack(prob)
    X_preds = lsoda_trajectories(prob["F"], prob["K"], t_span, prob["scalar_K"], prob["kappa"])
    for lam_cog in (0.0, 0.4):
        ref = sse_matrix(
            prob["X_obs"], prob["dt"], prob["ids"], prob["beta"], X_preds, prob["s"], t_span,
            cog=prob["cog"], cluster_cog_a=list(prob["cog_a"]),
            cluster_cog_b=list(prob["cog_b"]), lambda_cog=lam_cog,
            obs_weight=prob["obs_weight"],
        )
        out = np.asarray(jc._sse_matrix_jit(
            jc.f64(np.stack([x.T for x in X_preds])), jc.f64(prob["s"]),
            jc.pad_beta(prob["beta"], meta["P"]), data, jc.f64(prob["cog_a"]),
            jc.f64(prob["cog_b"]), jc.f64(lam_cog), h,
        ))
        np.testing.assert_allclose(out[: prob["n_patients"]], ref, rtol=1e-9)
        assert np.all(out[prob["n_patients"]:] == 0.0)


# ---------------------------------------------------------------------------
# 4. JSD equivalence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("Z,empty", [(2, None), (3, None), (4, None), (3, 1), (4, 2)])
@pytest.mark.parametrize("bandwidth", [None, 0.8])
def test_kernel_jsd_matches_numpy(Z, empty, bandwidth):
    rng = np.random.default_rng(10 + Z)
    n, P = 37, 64
    beta = rng.uniform(0.0, T_MAX, size=n)
    assignments = rng.integers(0, Z, size=n)
    if empty is not None:
        assignments[assignments == empty] = (empty + 1) % Z
    present = np.unique(assignments)
    ref = KernelJSDMulti(
        [beta[assignments == z] for z in present], value_range=(0, T_MAX), bandwidth=bandwidth,
    ).jsd()

    patient_valid = np.zeros(P)
    patient_valid[:n] = 1.0
    out = jc.kernel_jsd(
        jc.pad_beta(beta, P), jc.pad_assignments(assignments, P), jc.f64(patient_valid), Z,
        jc.f64(0.0), jc.f64(T_MAX), jc.f64(1.0 if bandwidth is None else bandwidth),
        max(20, int(T_MAX * 2)), bandwidth is None,
    )
    np.testing.assert_allclose(float(out), ref, rtol=1e-9)


def test_kernel_jsd_single_present_subtype_is_zero():
    P = 64
    beta = np.linspace(1, 9, 20)
    pv = np.zeros(P)
    pv[:20] = 1.0
    out = jc.kernel_jsd(
        jc.pad_beta(beta, P), jc.pad_assignments(np.ones(20, int), P), jc.f64(pv), 3,
        jc.f64(0.0), jc.f64(T_MAX), jc.f64(1.0), 24, True,
    )
    assert float(out) == 0.0


# ---------------------------------------------------------------------------
# 5. Gradients vs central finite differences (with NaNs and padding rows)
# ---------------------------------------------------------------------------


def _globals_inputs(prob, data, meta):
    P = meta["P"]
    return dict(
        F=jc.f64(prob["F"]),
        beta_full=jc.pad_beta(prob["beta"], P),
        assign_full=jc.pad_assignments(prob["assignments"], P),
        data=data,
        K=jc.f64(prob["K"]),
    )


def test_globals_gradient_fd():
    prob = make_problem(seed=7)
    _, h, n_steps = grid()
    data, meta = pack(prob)
    assert meta["R"] > meta["n_rows"]
    inp = _globals_inputs(prob, data, meta)
    hyp = jc.make_hyp(lambda_s=0.2, lambda_scalar=0.5, lambda_kappa=0.3)
    B = len(prob["s"])

    def call(p):
        return jc._globals_vg(
            jc.f64(p[:B]), jc.f64(p[B:2 * B]), jc.f64(p[-1]), inp["F"], inp["beta_full"],
            inp["assign_full"], inp["data"], inp["K"], hyp, h, n_steps=n_steps,
            s_free=jc.f64(np.ones(B)),
        )

    p0 = np.concatenate([prob["s"], prob["kappa"], [prob["scalar_K"]]])
    _, (gs, gk, gK) = call(p0)
    g = np.concatenate([np.asarray(gs), np.asarray(gk), [float(gK)]])
    assert_grad_close(g, central_fd(lambda p: float(call(p)[0]), p0))


def test_cluster_gradient_fd():
    prob = make_problem(seed=8)
    _, h, n_steps = grid()
    data, meta = pack(prob)
    inp = _globals_inputs(prob, data, meta)
    hyp = jc.make_hyp(lambda_f=0.1)
    shape = prob["F"].shape

    def call(f):
        return jc._cluster_vg(
            jc.f64(f.reshape(shape)), jc.f64(prob["s"]), jc.f64(prob["kappa"]),
            jc.f64(prob["scalar_K"]), inp["beta_full"], inp["assign_full"], inp["data"],
            inp["K"], hyp, h, n_steps=n_steps,
        )

    f0 = prob["F"].ravel()
    _, g = call(f0)
    assert_grad_close(np.asarray(g).ravel(), central_fd(lambda f: float(call(f)[0]), f0))


@pytest.mark.parametrize("use_jsd", [False, True])
def test_beta_gradient_fd(use_jsd):
    prob = make_problem(seed=9)
    t_span, h, n_steps = grid()
    data, meta = pack(prob)
    P, n = meta["P"], prob["n_patients"]
    traj = jc._traj_jit(
        jc.f64(prob["F"]), jc.f64(prob["kappa"]), jc.f64(prob["scalar_K"]),
        jc.f64(prob["K"]), h, n_steps=n_steps,
    )
    hyp = jc.make_hyp(lambda_cog=0.2, lambda_beta=0.3, beta_mean=4.0, beta_var=2.0,
                      lambda_jsd=3.0)
    assign_full = jc.pad_assignments(prob["assignments"], P)

    def call(b):
        return jc._beta_vg(
            jc.pad_beta(b, P), traj, jc.f64(prob["s"]), assign_full, jc.f64(prob["cog_a"]),
            jc.f64(prob["cog_b"]), data, hyp, h, jc.f64(0.0), jc.f64(T_MAX), jc.f64(1.0),
            n_bins=24, use_jsd=use_jsd, auto_bandwidth=True,
        )

    _, g = call(prob["beta"])
    g = np.asarray(g)
    assert np.all(g[n:] == 0.0), "padding patients must get zero gradient"
    assert_grad_close(g[:n], central_fd(lambda b: float(call(b)[0]), prob["beta"]))


# ---------------------------------------------------------------------------
# 6. Padding invariance / 7. Missing data
# ---------------------------------------------------------------------------


def _all_losses_and_grads(prob, data, meta):
    """Loss and real-entry gradients of the three value_and_grad functions."""
    _, h, n_steps = grid()
    inp = _globals_inputs(prob, data, meta)
    n, P = prob["n_patients"], meta["P"]
    hyp = jc.make_hyp(lambda_s=0.1, lambda_scalar=0.2, lambda_kappa=0.1, lambda_f=0.1,
                      lambda_cog=0.2, lambda_beta=0.3, beta_mean=4.0, beta_var=2.0,
                      lambda_jsd=1.0)
    s, kappa, sK = jc.f64(prob["s"]), jc.f64(prob["kappa"]), jc.f64(prob["scalar_K"])
    vg, (gs, gk, gK) = jc._globals_vg(
        s, kappa, sK, inp["F"], inp["beta_full"], inp["assign_full"], data, inp["K"], hyp, h,
        n_steps=n_steps, s_free=jc.f64(np.ones(len(prob["s"]))),
    )
    vc, gF = jc._cluster_vg(
        inp["F"], s, kappa, sK, inp["beta_full"], inp["assign_full"], data, inp["K"], hyp, h,
        n_steps=n_steps,
    )
    traj = jc._traj_jit(inp["F"], kappa, sK, inp["K"], h, n_steps=n_steps)
    vb, gb = jc._beta_vg(
        inp["beta_full"], traj, s, inp["assign_full"], jc.f64(prob["cog_a"]),
        jc.f64(prob["cog_b"]), data, hyp, h, jc.f64(0.0), jc.f64(T_MAX), jc.f64(1.0),
        n_bins=24, use_jsd=True, auto_bandwidth=True,
    )
    sse = jc._recon_sse_jit(traj, s, inp["beta_full"], inp["assign_full"], data, h)
    sse_mat = jc._sse_matrix_jit(
        traj, s, inp["beta_full"], data, jc.f64(prob["cog_a"]), jc.f64(prob["cog_b"]),
        jc.f64(0.2), h,
    )
    del P
    return [
        float(vg), np.asarray(gs), np.asarray(gk), float(gK), float(vc), np.asarray(gF),
        float(vb), np.asarray(gb)[:n], float(sse), np.asarray(sse_mat)[:n],
    ]


def test_padding_invariance():
    prob = make_problem(seed=11)
    a = _all_losses_and_grads(prob, *pack(prob))
    b = _all_losses_and_grads(prob, *pack(prob, pad_rows=700, pad_patients=150))
    for x, y in zip(a, b):
        np.testing.assert_allclose(x, y, rtol=1e-12, atol=1e-12)


def test_pack_data_rejects_small_pads():
    prob = make_problem(seed=11)
    with pytest.raises(ValueError):
        pack(prob, pad_rows=10)
    with pytest.raises(ValueError):
        pack(prob, pad_patients=5)


def test_missing_entry_value_is_ignored():
    prob = make_problem(seed=12)
    miss = np.argwhere(np.isnan(prob["X_raw"]))
    assert len(miss) > 0
    data_nan, meta = pack(prob)

    X_perturbed = prob["X_raw"].copy()
    X_perturbed[np.isnan(X_perturbed)] = 123.456
    data_pert, _ = jc.pack_data(
        X_perturbed, prob["obs_weight"], prob["dt"], prob["ids"], prob["cog"],
        prob["n_patients"],
    )
    a = _all_losses_and_grads(prob, data_nan, meta)
    b = _all_losses_and_grads(prob, data_pert, meta)
    for x, y in zip(a, b):
        np.testing.assert_allclose(x, y, rtol=1e-13, atol=1e-13)


# ---------------------------------------------------------------------------
# 8. Empty subtype
# ---------------------------------------------------------------------------


def test_empty_subtype_f_unchanged():
    prob = make_problem(seed=13)
    prob["assignments"][prob["assignments"] == 1] = 0
    t_span, _, _ = grid()
    F0 = [f.copy() for f in prob["F"]]
    F_fit = oj.fit_theta_cluster_all_jax(
        prob["X_obs"], prob["dt"], prob["ids"], prob["K"], t_span,
        cluster_f=F0, assignments=prob["assignments"], s=prob["s"],
        scalar_K=prob["scalar_K"], lambda_f=0.5, beta_pred=prob["beta"],
        kappa=prob["kappa"], obs_weight=prob["obs_weight"],
    )
    assert np.array_equal(F_fit[1], prob["F"][1])
    assert not np.allclose(F_fit[0], prob["F"][0])
    assert all(np.all(f >= 0) for f in F_fit)


def test_fit_theta_globals_jax_respects_s_mask():
    prob = make_problem(seed=14)
    t_span, _, _ = grid()
    s_mask = np.array([True, False, True, False])
    s_fit, kappa_fit, sK_fit, trajs = oj.fit_theta_globals_jax(
        prob["X_obs"], prob["dt"], prob["ids"], prob["K"], t_span, list(prob["F"]),
        prob["assignments"], beta_pred=prob["beta"], s_guess=prob["s"],
        scalar_K_guess=prob["scalar_K"], kappa_guess=prob["kappa"],
        obs_weight=prob["obs_weight"], s_mask=s_mask,
    )
    assert np.all(s_fit[~s_mask] == 1.0)
    assert np.all(kappa_fit >= 0) and sK_fit >= 0
    assert len(trajs) == prob["n_subtypes"]
    assert trajs[0].shape == (prob["K"].shape[0], len(t_span))

    s_fit2, *_ = oj.fit_theta_globals_jax(
        prob["X_obs"], prob["dt"], prob["ids"], prob["K"], t_span, list(prob["F"]),
        prob["assignments"], beta_pred=prob["beta"], s_guess=prob["s"],
        obs_weight=prob["obs_weight"], fit_s=False,
    )
    assert np.all(s_fit2 == 1.0)


def test_nonfinite_guard():
    prob = make_problem(seed=15)
    t_span, _, _ = grid()
    X_preds = lsoda_trajectories(prob["F"], prob["K"], t_span, prob["scalar_K"], prob["kappa"])
    X_preds[0][:, 5] = np.nan
    before = oj.nonfinite_events()
    with pytest.warns(RuntimeWarning, match="non-finite"):
        oj.estimate_beta_jax(
            prob["beta"], prob["X_obs"], prob["dt"], prob["ids"], prob["cog"], t_span,
            X_preds, prob["s"], np.zeros(prob["n_patients"], int),
            list(prob["cog_a"]), list(prob["cog_b"]), t_max=T_MAX,
            obs_weight=prob["obs_weight"],
        )
    assert oj.nonfinite_events() > before


# ---------------------------------------------------------------------------
# 9. No recompiles / 10. End to end
# ---------------------------------------------------------------------------


def make_patients(seed: int = 0, n_patients: int = 60, n_biomarkers: int = 5, noise: float = 0.02):
    """Synthetic patients from 2 ground-truth subtypes (forcing on different biomarkers).

    Returns the patient list, ``K``, and "clinically informed" initial betas
    (true beta + N(0, 1)), standing in for the mixed-model initialization the
    notebooks use.
    """
    rng = np.random.default_rng(seed)
    K = make_K(n_biomarkers, seed)
    t_span = np.linspace(0.0, T_MAX, int(T_MAX / STEP))
    F_true = np.zeros((2, n_biomarkers))
    F_true[0, :2] = [0.4, 0.2]
    F_true[1, -2:] = [0.2, 0.4]
    trajs = lsoda_trajectories(F_true, K, t_span, 0.3, np.full(n_biomarkers, 0.1))
    X, beta_true = [], []
    for i in range(n_patients):
        z = i % 2
        beta = rng.uniform(1.0, 7.0)
        n_visits = int(rng.integers(2, 5))
        dt = np.arange(n_visits, dtype=float)
        x_obs = np.array(
            [[np.interp(beta + d, t_span, trajs[z][b]) for b in range(n_biomarkers)] for d in dt]
        ) + rng.normal(0.0, noise, size=(n_visits, n_biomarkers))
        X.append(dict(id=i, X_obs=x_obs, dt=dt, cog=rng.normal(size=(n_visits, 1)),
                      subtype_true=z))
        beta_true.append(beta)
    initial_beta = np.clip(np.array(beta_true) + rng.normal(0.0, 1.0, n_patients), 0.0, T_MAX)
    return X, K, initial_beta


def test_no_recompiles():
    X, K, _ = make_patients(seed=21)
    jax.config.update("jax_log_compiles", True)
    jax.config.update("jax_explain_cache_misses", True)
    try:
        jax.clear_caches()
        rng = np.random.default_rng(0)
        common = dict(K=K, n_subtypes=2, t_max=T_MAX, step=STEP, backend="jax",
                      max_iter=3, verbose=0)

        em1 = SubtypingEM(**common, lambda_f=0.01, lambda_kappa=0.5, lambda_jsd=0.1,
                          lambda_cog=0.0, initial_assignments=rng.integers(0, 2, len(X)))
        em1.fit(X)
        R, P = em1.jax_padded_shape_

        em2 = SubtypingEM(**common, lambda_f=0.05, lambda_kappa=1.0, lambda_jsd=0.7,
                          lambda_cog=0.1, lambda_scalar=0.3, lambda_beta=0.2,
                          initial_assignments=rng.integers(0, 2, len(X)))
        em2.fit(X)

        fold = X[:41]
        em3 = SubtypingEM(**common, lambda_jsd=0.3, jax_pad_rows=R, jax_pad_patients=P)
        em3.fit(fold)
        assert em3.jax_padded_shape_ == (R, P)

        for lam_cog, lam_jsd in ((0.0, 0.1), (0.2, 0.6)):
            SubtypingEM(**common, lambda_cog=lam_cog, lambda_jsd=lam_jsd,
                        estep_mode="joint_grid", cap_beta_by_followup=True,
                        initial_assignments=rng.integers(0, 2, len(X))).fit(X)

        # Intercept-only jits (_globals_c_vg, _shift_data_jit) compile once on
        # the same padded shape; they must not bump the existing caches.
        B = K.shape[0]
        SubtypingEM(
            **common, fit_intercept=True, c_bounds=(0.0, np.inf),
            initial_c=np.zeros(B), jax_pad_rows=R, jax_pad_patients=P,
            lambda_jsd=0.1, lambda_f=0.01, lambda_kappa=0.5,
            initial_assignments=rng.integers(0, 2, len(X)),
        ).fit(X)

        def cache_sizes():
            return {name: fn._cache_size() for name, fn in jc.JITTED_FUNCTIONS.items()}

        sizes = cache_sizes()
        expected = {name: 0 if name in jc.TRANSFORM_ONLY_JITTED else 1
                    for name in jc.JITTED_FUNCTIONS}
        assert sizes == expected, sizes

        # transform shapes differ from the fit's, so the first round may add one
        # entry per function; a second round with the same padded shape adds none.
        em1.transform_backend = "jax"
        em1.transform_pad_rows, em1.transform_pad_patients = 256, 64
        em1.transform(X[:30])
        em1.transform_soft(X[:30], per_timepoint=True)
        after_first = cache_sizes()
        em1.transform(X[30:], use_cognitive_prior=False)
        em1.transform_soft(X[30:], temperature=2.0, per_timepoint=True)
        assert cache_sizes() == after_first
        assert all(after_first[name] == 1 for name in jc.TRANSFORM_ONLY_JITTED), after_first
    finally:
        jax.config.update("jax_log_compiles", False)
        jax.config.update("jax_explain_cache_misses", False)


def test_end_to_end_vs_numpy():
    """JAX backend vs NumPy ``lbfgs_approx`` from the same informed start.

    From fully random initial betas, exact-gradient globals steps (JAX and
    NumPy ``lbfgs_exact`` alike) can fall into a saturated basin (large
    ``scalar_K`` or ``kappa``, near-flat trajectories, vanishing beta
    gradients). Informed initial betas, as used in practice, avoid it.
    """
    X, K, initial_beta = make_patients(seed=0)
    true = np.array([p["subtype_true"] for p in X])
    common = dict(K=K, n_subtypes=2, t_max=T_MAX, step=STEP, max_iter=20, verbose=0,
                  lambda_kappa=0.2, lambda_scalar=0.2, initial_beta=initial_beta)

    em_np = SubtypingEM(**common, theta_solver_stages=("lbfgs_approx",),
                        rng=np.random.default_rng(1))
    em_np.fit(X)
    em_jax = SubtypingEM(**common, backend="jax", rng=np.random.default_rng(1))
    em_jax.fit(X)

    assert em_jax.lse_final <= 1.05 * em_np.lse_final, (em_jax.lse_final, em_np.lse_final)
    assert adjusted_rand_score(true, em_jax.final_assignments) >= 0.8
    assert em_jax.jax_nonfinite_events_ == 0
    assert set(em_jax.accepted_solver_stages) == {"jax"}
    assert np.isfinite(em_jax.bic_)
    out = em_jax.transform(X[:5])
    assert np.all(np.isfinite(out["beta"]))


def test_invalid_backend():
    with pytest.raises(ValueError):
        SubtypingEM(backend="torch")
