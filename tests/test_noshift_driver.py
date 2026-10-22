"""Tests for the 2026-10-20 W-score / CN A+ ADNI driver (synthetic data)."""

import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("jax")
pytest.importorskip("optuna")

EXP_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "experiments",
    "2026_10_20_ADNI_optuna_connectome_experiment",
)
DRIVER_PATH = os.path.join(EXP_DIR, "run_comind_optuna_bic_noshift.py")
if not os.path.exists(DRIVER_PATH):
    pytest.skip(f"driver not found: {DRIVER_PATH}", allow_module_level=True)

sys.path.insert(0, EXP_DIR)
_spec = importlib.util.spec_from_file_location("noshift_driver", DRIVER_PATH)
drv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(drv)

from comind_init import (  # noqa: E402
    data_driven_f_and_assignments,
    init_ratio,
    random_f_and_assignments,
    restart_is_failed,
    select_restart,
)
from COMIND_transformer.model_selection import count_bic_params  # noqa: E402
from COMIND_transformer.preprocessing import build_connectome  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402

IMG = ["ctx-lh-a_wscore", "ctx-rh-b_wscore", "Left-Hippocampus_wscore"]


def _row(pid, t, dx_bl, img, *, dx=None, moca=26.0, adas=15.0, mmse=28.0, amyloid="A?",
         apoe=0.0, wscore_ref=False, **extra):
    row = dict(PTID=pid, dt=float(t), DX_bl=dx_bl, DX=dx or dx_bl, MOCA=moca, ADAS13=adas,
               MMSE=mmse, APOE4=apoe, amyloid=amyloid, wscore_ref=wscore_ref)
    row.update({c: v for c, v in zip(IMG, img)})
    row.update(extra)
    return row


def _cn_rows(n_subjects=20, seed=0, amyloid="A-", wscore_ref=True):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_subjects):
        for t in (0.0, 1.0):
            rows.append(_row(
                f"CN{i}", t, "CN", rng.normal(0, 0.2, 3),
                dx="CN", moca=28.0, adas=8.0, amyloid=amyloid, wscore_ref=wscore_ref,
            ))
    return rows


def test_no_minshift_cn_reference_centered():
    rows = [_row(f"CN{i}", t, "CN", [0.0, 0.0, 0.0]) for i in range(8) for t in (0.0, 1.0)]
    rows += [_row("AD0", t, "AD", [-2.0, -2.0, -2.0]) for t in (0, 1)]
    out, info = drv.preprocess_imaging(pd.DataFrame(rows), IMG, zero_mode="cnmean")
    assert info["zscore_applied"] is False
    assert info["n_outlier_cells"] == 0
    cn_mean = out.loc[out["DX_bl"] == "CN", IMG].mean(axis=0).to_numpy()
    np.testing.assert_array_less(np.abs(cn_mean), 1e-12)
    assert (out.loc[out["PTID"] == "AD0", IMG].to_numpy() > 0).all()
    np.testing.assert_array_equal(info["per_biomarker_min"], np.zeros(len(IMG)))
    assert info["imaging_scale"] == drv.Z_OUTLIER_SD
    assert info["zero_mode"] == "cnmean"


def test_wscore_preprocess_flip_mask_no_zscore():
    df = pd.DataFrame([_row("S0", 0, "AD", [-2.0, 6.0, 0.5])])
    out, info = drv.preprocess_imaging(df, IMG, zero_mode="cnmean")
    assert info["zscore_applied"] is False
    np.testing.assert_allclose(out.loc[0, IMG[0]], 0.4, rtol=1e-12)
    assert np.isnan(out.loc[0, IMG[1]])
    np.testing.assert_allclose(out.loc[0, IMG[2]], -0.1, rtol=1e-12)
    assert info["n_outlier_cells"] == 1


def test_flip_direction():
    out, _ = drv.preprocess_imaging(
        pd.DataFrame([_row("AD0", 0, "AD", [-3.0, -3.0, -3.0])]), IMG, zero_mode="cnmean"
    )
    vals = out.loc[out["PTID"] == "AD0", IMG].to_numpy().ravel()
    assert np.all(vals > 0)
    np.testing.assert_allclose(vals, 3.0 / drv.Z_OUTLIER_SD, rtol=1e-12)


def test_scale_and_range():
    extra = [
        _row("AD0", 0, "AD", [-4.9] * 3),
        _row("AD0", 1, "AD", [-12.0] * 3),
        _row("AD1", 0, "AD", [7.0] * 3),
    ]
    out, info = drv.preprocess_imaging(pd.DataFrame(extra), IMG, zero_mode="cnmean")
    vals = out[IMG].to_numpy()
    finite = vals[np.isfinite(vals)]
    assert finite.min() >= -1.0 and finite.max() <= 1.0
    assert info["n_outlier_cells"] == 6
    assert info["n_rows_with_masked_cells"] == 2


def test_cell_masking():
    rows = _cn_rows()
    bad = [-1.0, -1.0, -1.0]
    bad[0] = 8.0
    ad = [
        _row("AD0", 0, "AD", [-1.0] * 3, moca=22.0, adas=30.0, amyloid="A+"),
        _row("AD0", 1, "AD", bad, moca=20.0, adas=33.0, amyloid="A+"),
        _row("AD0", 2, "AD", [np.nan] * 3, moca=np.nan, adas=np.nan, mmse=np.nan, amyloid="A+"),
        _row("AD0", 3, "AD", [-1.5] * 3, moca=18.0, adas=36.0, amyloid="A+"),
    ]
    df = pd.concat([pd.DataFrame(rows), pd.DataFrame(ad)], ignore_index=True)
    pre = drv.preprocess_frame(df, zero_mode="cnmean")
    out = pre["df"]
    g = out[out["PTID"] == "AD0"].sort_values("dt")
    assert g["dt"].tolist() == [0.0, 1.0, 3.0]
    masked = g[g["dt"] == 1.0].iloc[0]
    assert np.isnan(masked[IMG[0]])
    np.testing.assert_allclose(masked[IMG[1:]].to_numpy(dtype=float), 1.0 / drv.Z_OUTLIER_SD,
                               rtol=1e-12)
    cn_moca = (30.0 - 28.0) / 30.0
    cn_adas = 8.0 / 85.0
    assert masked["MOCA"] == pytest.approx(((30.0 - 20.0) / 30.0 - cn_moca) / (1.0 - cn_moca))
    assert masked["ADAS13"] == pytest.approx((33.0 / 85.0 - cn_adas) / (1.0 - cn_adas))
    info = pre["imaging_info"]
    assert info["zscore_applied"] is False
    assert info["n_outlier_cells"] == 1
    assert info["n_rows_with_masked_cells"] == 1
    assert info["outlier_cells_per_region"].tolist() == [1, 0, 0]


def test_cohort_drops_cn_and_smc():
    rows = _cn_rows(12)
    img = [-2.0, -2.0, -2.0]
    extra = []
    for t in (0, 1, 2):
        extra.append(_row("SMC0", t, "SMC", img, dx="Dementia" if t == 2 else "CN",
                          amyloid="A+"))
        extra.append(_row("CNdem", t, "CN", img, dx="Dementia" if t == 2 else "CN",
                          amyloid="A+"))
        extra.append(_row("AD0", t, "AD", img, amyloid="A+"))
    pre = drv.preprocess_frame(
        pd.concat([pd.DataFrame(rows), pd.DataFrame(extra)], ignore_index=True),
        zero_mode="cnmean",
    )
    kept = pre["df"]
    assert set(kept["PTID"]) == {"AD0"}
    assert not kept["DX_bl"].isin(["CN", "SMC"]).any()
    stats = pre["cohort_filter_stats"]
    assert stats["n_dropped_smc"] == 1
    assert stats["n_dropped_cn_bl"] == 12 + 1
    assert stats["n_cn_amyloid_pos_added"] == 0


def _processed_frame(rows):
    df = pd.DataFrame(rows)
    df["dt"] = df["dt"].astype(float)
    return df


def test_min_timepoints_counts_mri_rows():
    rows = [_row("one_mri", 0, "AD", [0.2, 0.2, 0.2])]
    rows += [_row("one_mri", t, "AD", [np.nan] * 3) for t in (1, 2, 3)]
    rows += [_row("two_mri", t, "AD", [0.2, 0.3, 0.1]) for t in (0, 1)]
    rows += [_row("two_mri", 2, "AD", [np.nan] * 3)]
    out, stats = drv.filter_cohort_subjects(
        _processed_frame(rows), "PTID", "dt", "DX", "DX_bl", IMG, min_timepoints=2
    )
    assert set(out["PTID"]) == {"two_mri"}
    assert len(out) == 3  # clinical-only row kept
    assert stats["n_dropped_lt_min_timepoints"] == 1


def test_amyloid_column_used():
    rows = []
    for t in (0, 1):
        rows.append(_row("mci_pos", t, "LMCI", [0.2] * 3, dx="MCI", amyloid="A+"))
        # high AV45 must not override the amyloid column
        rows.append(_row("mci_neg", t, "EMCI", [0.2] * 3, dx="MCI", amyloid="A-", AV45=1.5))
    df = _processed_frame(rows)
    out, stats = drv.filter_cohort_subjects(df, "PTID", "dt", "DX", "DX_bl", IMG)
    assert set(out["PTID"]) == {"mci_pos"}
    assert stats["amyloid_rule"] == "amyloid_column"
    assert stats["n_dropped_mci_amyloid_neg"] == 1
    with pytest.raises(ValueError, match="amyloid"):
        drv.filter_cohort_subjects(
            df.drop(columns=["amyloid"]), "PTID", "dt", "DX", "DX_bl", IMG
        )


def test_clinical_direction():
    df = pd.DataFrame({"MOCA": [30.0, 25.0, 10.0], "ADAS13": [5.0, 20.0, 60.0],
                       "MMSE": [30.0, 28.0, 20.0]})
    drv.apply_clinical_biomarker_transforms(df)
    assert np.all(np.diff(df["MOCA"].to_numpy()) > 0)  # raw MoCA falls -> transformed rises
    assert np.all(np.diff(df["ADAS13"].to_numpy()) > 0)
    np.testing.assert_allclose(df["MOCA"], [0.0, 5.0 / 30.0, 20.0 / 30.0])
    np.testing.assert_allclose(df["ADAS13"], np.array([5.0, 20.0, 60.0]) / 85.0)
    assert df["MOCA"].iloc[0] < df["MOCA"].iloc[1]


@pytest.fixture(scope="module")
def tiny_fit():
    rng = np.random.default_rng(3)
    x_cols = ["ctx-lh-a_harmonized", "ctx-lh-b_harmonized", "Left-Putamen_harmonized",
              "MOCA", "ADAS13"]
    B = len(x_cols)
    K = rng.uniform(0, 1, (B, B))
    K[:, 3:] = 0.0
    K[3:, :] = 0.0
    np.fill_diagonal(K, 0.0)
    X = []
    for i in range(24):
        n = int(rng.integers(2, 4))
        dt = np.arange(n, dtype=float)
        base = rng.uniform(0.0, 0.4, B)
        X_obs = base + 0.03 * dt[:, None] + rng.normal(0, 0.02, (n, B))
        X.append(dict(id=i, X_obs=X_obs, dt=dt, cog=np.zeros((n, 1))))
    params = drv.sample_random_params(np.random.default_rng(0), n_subtypes=2)
    params.update(lambda_f=0.05, lambda_cog=0.0, lambda_beta=0.0)
    f_init = np.full(B, 0.1)
    em = drv.build_estimator(
        params, K, f_init, 10.0, max_iter=2, fit_s=False, x_cols=x_cols,
        initial_beta=rng.uniform(0, 5, len(X)), step=0.1, backend="jax", verbose=0,
    )
    em.fit(X)
    return em, K, X, x_cols, params


def test_s_fixed(tiny_fit):
    em, K, X, x_cols, _ = tiny_fit
    np.testing.assert_array_equal(np.asarray(em.final_s), np.ones(len(x_cols)))
    F = np.stack([np.ravel(f) for f in em.cluster_f])
    expected = int(np.sum(np.abs(em.final_kappa) >= 0.01)) + 1 + int(np.sum(F >= 0.01))
    assert em.bic_n_params_ == expected
    s_mask = drv.clinical_s_mask(x_cols)
    k_no_s = count_bic_params(em.final_s, em.final_kappa, em.cluster_f, em.n_subtypes, 0.0,
                              em.cluster_cog_a, fit_s=False, s_mask=s_mask)
    k_s = count_bic_params(em.final_s, em.final_kappa, em.cluster_f, em.n_subtypes, 0.0,
                           em.cluster_cog_a, fit_s=True, s_mask=s_mask)
    assert k_no_s == expected
    assert k_s == expected + int(s_mask.sum())


def test_post_fit_diagnostics_run(tiny_fit):
    em, K, X, _, params = tiny_fit
    packed = drv.pack_patients(X)
    diag, convex = drv.compute_post_fit_diagnostics(em, K, params["lambda_f"], packed=packed)
    assert convex.shape == (em.n_subtypes, K.shape[0])
    assert diag["f_kkt"] is not None and diag["f_kkt_error"] is None
    assert diag["n_f_active"] + diag["n_f_zero"] <= convex.size
    assert sum(diag["subtype_counts"]) == len(X)
    assert diag["n_em_iters_accepted"] == len(em.lse_history) - 1


CONNECTOME_ROW = os.path.join(drv.CONNECTOME_DIR, "K_82_log_knn8_row.csv")


@pytest.mark.skipif(not os.path.exists(CONNECTOME_ROW), reason="connectome CSV not available")
def test_connectome_not_symmetrized():
    K_df = pd.read_csv(CONNECTOME_ROW, index_col=0)
    names = list(np.random.default_rng(0).permutation(K_df.index.tolist()))
    x_cols = names + ["MOCA", "ADAS13"]
    K, disconnected = build_connectome(CONNECTOME_ROW, x_cols, pad_missing=True)
    n = len(names)
    np.testing.assert_array_equal(K[:n, :n], K_df.loc[names, names].to_numpy(dtype=float))
    np.testing.assert_allclose(K[:n, :n].sum(axis=1), 1.0, atol=1e-10)
    assert not np.allclose(K, K.T)
    assert np.all(K[n:, :] == 0) and np.all(K[:, n:] == 0)
    assert disconnected == ["MOCA", "ADAS13"]


def test_search_space_log_bounds():
    rng = np.random.default_rng(1)
    draws = [drv.sample_random_params(rng, n_subtypes=3) for _ in range(200)]
    for name, (lo, hi, use_log) in drv.SEARCH_SPACE.items():
        assert use_log
        vals = np.array([d[name] for d in draws])
        assert np.all(vals >= lo) and np.all(vals <= hi)
        width = np.log10(hi) - np.log10(lo)
        mid = 0.5 * (np.log10(lo) + np.log10(hi))
        assert abs(np.median(np.log10(vals)) - mid) < 0.1 * width
    expected = {
        "lambda_f": (1e-2, 1e6), "lambda_kappa": (1e-2, 1e4), "lambda_scalar": (1e-2, 1e3),
        "scalar_K_center": (1e-2, 1.0), "lambda_jsd": (1e-1, 1e3),
    }
    assert {k: v[:2] for k, v in drv.SEARCH_SPACE.items()} == expected


def test_study_names_unique():
    csv = os.path.join(drv.CONNECTOME_DIR, "K_82_log_knn8_row.csv")
    four = []
    for z in (3, 4):
        for cnapos in (False, True):
            four.append(
                drv.study_paths(
                    "/tmp/exp", csv, z, zero_mode="cnmean", intercept=True,
                    include_cn_amyloid_pos=cnapos,
                )
            )
    for key in ("study_name", "out_dir", "journal_path"):
        assert len({p[key] for p in four}) == 4, key
    p = drv.study_paths(
        "/tmp/exp", csv, 3, zero_mode="cnmean", intercept=True,
        include_cn_amyloid_pos=True,
    )
    assert p["study_name"] == (
        "comind_adni_bic_wscore_K_82_log_knn8_row_cnmean_int_cnapos_k3"
    )
    combos = [
        (os.path.join(drv.CONNECTOME_DIR, f"{k}.csv"), z, fit_s, zero, icept, cnapos)
        for k in ("K_82_top10_median", "K_82_log_knn8_row")
        for z in (3, 4)
        for fit_s in (False, True)
        for zero in ("minshift", "cnmean")
        for icept in (True, False)
        for cnapos in (False, True)
    ]
    paths = [
        drv.study_paths(
            "/tmp/exp", c, z, fit_s=f, zero_mode=zero, intercept=icept,
            include_cn_amyloid_pos=cnapos,
        )
        for c, z, f, zero, icept, cnapos in combos
    ]
    for key in ("study_name", "out_dir", "journal_path"):
        assert len({p[key] for p in paths}) == len(combos), key


def test_convex_start_formula():
    F = np.array([[0.1, 0.0]])
    K = np.array([[0.0, 0.0], [1.0, 0.0]])
    convex = drv.convex_start_mask(F, np.zeros(2), 1.0, K)
    assert convex.tolist() == [[False, True]]


def test_clinical_cn_mean_rescale():
    df = pd.DataFrame({
        "DX_bl": ["CN", "CN", "AD"],
        "MOCA": [(30.0 - 28.0) / 30.0] * 2 + [(30.0 - 20.0) / 30.0],
        "ADAS13": [8.0 / 85.0] * 2 + [33.0 / 85.0],
    })
    means = drv.rescale_clinical_to_cn_mean(df)
    assert means["MOCA"] == pytest.approx((30.0 - 28.0) / 30.0)
    assert df["MOCA"].iloc[0] == pytest.approx(0.0)
    assert df["MOCA"].iloc[2] > df["MOCA"].iloc[0]


def test_minshift_initial_c_matches_cn_mean():
    rows = [_row(f"CN{i}", t, "CN", [0.0, 0.0, 0.0]) for i in range(8) for t in (0.0, 1.0)]
    rows += [_row("AD0", t, "AD", [-2.0, -2.0, -2.0]) for t in (0, 1)]
    df = pd.DataFrame(rows)
    out, info = drv.preprocess_imaging(df, IMG, zero_mode="minshift")
    cn_mean = out.loc[out["DX_bl"] == "CN", IMG].mean(axis=0).to_numpy()
    np.testing.assert_allclose(info["cn_imaging_mean"], cn_mean, atol=1e-8)
    x_cols = IMG + ["MOCA", "ADAS13"]
    icept = drv.intercept_config(x_cols, IMG, "minshift", info["cn_imaging_mean"], True)
    np.testing.assert_allclose(icept["initial_c"][: len(IMG)], cn_mean, atol=1e-8)
    assert icept["initial_c"][-2:].tolist() == [0.0, 0.0]
    assert icept["c_bounds"].shape == (len(x_cols), 2)
    np.testing.assert_allclose(icept["c_bounds"][: len(IMG), 0], 0.0)
    assert np.all(np.isinf(icept["c_bounds"][: len(IMG), 1]))
    icept_cn = drv.intercept_config(x_cols, IMG, "cnmean", info["cn_imaging_mean"], True)
    np.testing.assert_allclose(icept_cn["c_bounds"][: len(IMG), 0], -1.0)
    assert icept["c_mask"][: len(IMG)].all()
    assert not icept["c_mask"][-2:].any()


def test_no_intercept_estimator_uses_new_init():
    x_cols = ["a", "b", "MOCA", "ADAS13"]
    params = drv.sample_random_params(np.random.default_rng(0), n_subtypes=2)
    B = len(x_cols)
    em = drv.build_estimator(
        params, np.eye(B), np.full(B, 0.01), 10.0, max_iter=1, fit_s=False, x_cols=x_cols,
        step=0.1, backend="numpy", verbose=0, fit_intercept=False,
    )
    assert em.fit_intercept is False
    np.testing.assert_array_equal(em.initial_kappa, np.zeros(B))
    assert em.initial_scalar_K == pytest.approx(params["scalar_K_center"])
    np.testing.assert_array_equal(em.initial_s, np.ones(B))
    assert em.initial_c is None


def test_restart_selection_skips_failed():
    records = [
        {"restart": 0, "objective": 1.0, "failed": True},
        {"restart": 1, "objective": 5.0, "failed": False},
        {"restart": 2, "objective": 2.0, "failed": False},
    ]
    chosen, failed_all = select_restart(records)
    assert chosen["restart"] == 2
    assert not failed_all
    all_failed = [{"restart": i, "objective": 10.0 - i, "failed": True} for i in range(3)]
    chosen, failed_all = select_restart(all_failed)
    assert failed_all
    assert chosen["restart"] == 2


def _toy_patients(n=40, B=4, seed=0):
    rng = np.random.default_rng(seed)
    pats = []
    truth = []
    for i in range(n):
        z = i % 2
        truth.append(z)
        profile = np.array([0.4, 0.3, 0.0, 0.0]) if z == 0 else np.array([0.0, 0.0, 0.4, 0.3])
        nvis = 3
        dt = np.arange(nvis, dtype=float)
        X = profile + rng.normal(0, 0.02, (nvis, B))
        pats.append({
            "X_obs": X, "dt": dt, "cog": np.zeros((nvis, 1)), "id": i,
            "initial_beta": 12.0,
        })
    return pats, np.array(truth), rng


def test_data_driven_f_and_assignments():
    pats, truth, rng = _toy_patients()
    Z, B = 2, 4
    f0, z0 = data_driven_f_and_assignments(pats, Z, np.zeros(B), rng, pattern_cols=[0, 1, 2, 3])
    assert f0.shape == (Z, B) and z0.shape == (len(pats),)
    assert np.all(f0 > 0)
    assert not np.allclose(f0[0], f0[1])
    assert set(z0.tolist()) == {0, 1}
    assert adjusted_rand_score(truth, z0) > 0.8


def test_random_f_and_assignments_range():
    rng = np.random.default_rng(1)
    f0, z0 = random_f_and_assignments(20, 5, 3, rng)
    assert f0.shape == (3, 5)
    assert np.all((f0 >= 0.005) & (f0 <= 0.02))
    assert not np.allclose(f0[0], f0[1])
    assert not np.allclose(f0[0], f0[2])
    assert z0.shape == (20,)


def test_old_start_init_ratio_exceeds_one_new_does_not():
    from COMIND_transformer.subtyping_em_transformer import SubtypingEM

    pats, _, _ = _toy_patients(n=30, B=4, seed=1)
    B = 4
    K = np.zeros((B, B))
    c0 = np.zeros(B)

    def _fit(f, kappa, scalar_K):
        em = SubtypingEM(
            K=K, initial_f=f, initial_s=np.ones(B), initial_scalar_K=scalar_K,
            initial_kappa=kappa, n_subtypes=2, max_iter=1, t_max=10.0, step=0.1,
            fit_s=False, s_mask=np.zeros(B, dtype=bool), verbose=0, backend="jax",
            rng=np.random.default_rng(0),
        )
        em.fit(pats)
        return init_ratio(em, pats, c0)

    rng = np.random.default_rng(0)
    old = _fit(rng.uniform(0.05, 0.15, B), rng.uniform(0, 1, B), 1.0)
    f_new, z_new = data_driven_f_and_assignments(pats, 2, c0, np.random.default_rng(2))
    em_new = SubtypingEM(
        K=K, initial_f=f_new, initial_s=np.ones(B), initial_scalar_K=0.25,
        initial_kappa=np.zeros(B), initial_assignments=z_new, n_subtypes=2,
        max_iter=1, t_max=10.0, step=0.1, fit_s=False, s_mask=np.zeros(B, dtype=bool),
        verbose=0, backend="jax", rng=np.random.default_rng(2),
    )
    em_new.fit(pats)
    new = init_ratio(em_new, pats, c0)
    assert old > 1.0
    assert new < 1.0


def test_zero_f_restart_is_failed():
    from COMIND_transformer.subtyping_em_transformer import SubtypingEM

    pats, _, _ = _toy_patients(n=40, B=4, seed=2)
    B = 4
    K = np.zeros((B, B))
    em = SubtypingEM(
        K=K, initial_f=np.zeros((2, B)), initial_s=np.ones(B), initial_scalar_K=0.25,
        initial_kappa=np.zeros(B), n_subtypes=2, max_iter=3, t_max=10.0, step=0.1,
        fit_s=False, s_mask=np.zeros(B, dtype=bool), verbose=0, backend="jax",
        rng=np.random.default_rng(0),
    )
    em.fit(pats)
    ratio = init_ratio(em, pats, np.zeros(B))
    assert restart_is_failed(em, init_ratio_value=ratio, n_patients=len(pats))


def _two_visits(pid, dx_bl, amyloid, **kw):
    img = [-1.0, -1.0, -1.0]
    return [_row(pid, t, dx_bl, img, amyloid=amyloid, **kw) for t in (0.0, 1.0)]


def test_cohort_filter_table_and_ref_leak():
    rows = []
    rows += _two_visits("AD_pos", "AD", "A+")
    rows += _two_visits("AD_neg", "AD", "A-")
    rows += _two_visits("AD_unk", "AD", "A?")
    rows += _two_visits("MCI_pos", "MCI", "A+")
    rows += _two_visits("MCI_neg", "MCI", "A-")
    rows += _two_visits("MCI_unk", "MCI", "A?", dx="MCI")
    rows += _two_visits("MCI_unk_dem", "EMCI", "A?", dx="Dementia")
    rows += _two_visits("LMCI_pos", "LMCI", "A+")
    rows += _two_visits("CN_pos", "CN", "A+")
    rows += _two_visits("CN_neg", "CN", "A-", wscore_ref=True)
    rows += _two_visits("CN_unk", "CN", "A?")
    rows += _two_visits("SMC_pos", "SMC", "A+")
    rows += [_row("NO_DX", t, "", [-1.0] * 3, amyloid="A+") for t in (0.0, 1.0)]
    df = _processed_frame(rows)
    ref_ids = {"CN_neg"}

    out_off, stats_off = drv.filter_cohort_subjects(
        df, "PTID", "dt", "DX", "DX_bl", IMG, include_cn_amyloid_pos=False, ref_ids=ref_ids,
    )
    assert set(out_off["PTID"]) == {"AD_pos", "AD_unk", "MCI_pos", "MCI_unk_dem", "LMCI_pos"}
    assert stats_off["n_dropped_ad_amyloid_neg"] == 1
    assert stats_off["n_dropped_mci_amyloid_neg"] == 1
    assert stats_off["n_dropped_mci_no_amy_no_dem"] == 1
    assert stats_off["n_dropped_smc"] == 1
    assert stats_off["n_cn_amyloid_pos_added"] == 0
    assert stats_off["n_cn_amyloid_pos_eligible"] == 1
    assert "CN_neg" not in set(out_off["PTID"])

    out_on, stats_on = drv.filter_cohort_subjects(
        df, "PTID", "dt", "DX", "DX_bl", IMG, include_cn_amyloid_pos=True, ref_ids=ref_ids,
    )
    assert set(out_on["PTID"]) == set(out_off["PTID"]) | {"CN_pos"}
    assert stats_on["n_cn_amyloid_pos_added"] == 1

    with pytest.raises(RuntimeError, match="wscore_ref"):
        drv.filter_cohort_subjects(
            df, "PTID", "dt", "DX", "DX_bl", IMG, include_cn_amyloid_pos=False,
            ref_ids={"AD_pos"},
        )


def test_clinical_cn_mean_from_ref_ids():
    rows = [
        _row("REF", 0, "CN", [0.0] * 3, moca=28.0, adas=8.5, amyloid="A-", wscore_ref=True),
        _row("REF", 1, "CN", [0.0] * 3, moca=27.0, adas=9.0, amyloid="A-", wscore_ref=True),
        _row("AD0", 0, "AD", [-1.0] * 3, moca=20.0, adas=33.0, amyloid="A+"),
    ]
    df = pd.DataFrame(rows)
    drv.apply_clinical_biomarker_transforms(df)
    means = drv.rescale_clinical_to_cn_mean(df, ref_ids={"REF"})
    ref = df[df["PTID"] == "REF"]
    assert float(np.nanmean(ref["MOCA"])) == pytest.approx(0.0, abs=1e-12)
    assert float(np.nanmean(ref["ADAS13"])) == pytest.approx(0.0, abs=1e-12)
    assert means["MOCA"] == pytest.approx(np.mean([(30.0 - 28.0) / 30.0, (30.0 - 27.0) / 30.0]))


@pytest.mark.skipif(
    not os.path.exists(drv.CSV_PATH) or not os.path.exists(CONNECTOME_ROW),
    reason="W-score CSV or connectome not available",
)
def test_wscore_imaging_maps_to_connectome():
    cols = pd.read_csv(drv.CSV_PATH, nrows=0).columns
    img = drv.select_imaging_cols(cols)
    assert len(img) == drv.N_IMAGING_EXPECTED
    names = [drv.adni_col_to_connectome_name(c, drv.FEATURE_SUFFIX) for c in img]
    assert len(names) == len(set(names))
    K_df = pd.read_csv(CONNECTOME_ROW, index_col=0)
    missing = [n for n in names if n not in K_df.index]
    assert missing == []
    x_cols = img + ["MOCA", "ADAS13"]
    x_to_k = {c: drv.adni_col_to_connectome_name(c, drv.FEATURE_SUFFIX) for c in img}
    K, disconnected = build_connectome(CONNECTOME_ROW, x_cols, x_to_k=x_to_k, pad_missing=True)
    assert [c for c in disconnected if c in set(img)] == []
    assert set(disconnected) == {"MOCA", "ADAS13"}
    assert K.shape == (len(x_cols), len(x_cols))


def test_base_val_ids_identical_with_and_without_cnapos(tmp_path, monkeypatch):
    monkeypatch.setattr(drv, "N_IMAGING_EXPECTED", 3)
    names = [drv.adni_col_to_connectome_name(c, drv.FEATURE_SUFFIX) for c in IMG]
    K = pd.DataFrame(
        [[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]],
        index=names, columns=names,
    )
    kpath = str(tmp_path / "K_tiny.csv")
    K.to_csv(kpath)

    rows = []
    for i in range(12):
        apoe = float(i % 2)
        rows += _two_visits(f"AD{i}", "AD", "A+", apoe=apoe, moca=22.0, adas=30.0)
        rows += _two_visits(f"MCI{i}", "MCI", "A+", apoe=apoe, moca=24.0, adas=20.0)
        rows += _two_visits(
            f"CNp{i}", "CN", "A+", apoe=apoe, moca=27.0, adas=10.0, wscore_ref=False,
        )
        rows += _two_visits(
            f"REF{i}", "CN", "A-", apoe=apoe, moca=28.0, adas=8.0, wscore_ref=True,
        )
    df = pd.DataFrame(rows)
    ref_ids = {f"REF{i}" for i in range(12)}
    off = drv.prepare_experiment_data(
        df, kpath, zero_mode="cnmean", fit_intercept=True,
        include_cn_amyloid_pos=False, ref_ids=ref_ids,
    )
    on = drv.prepare_experiment_data(
        df, kpath, zero_mode="cnmean", fit_intercept=True,
        include_cn_amyloid_pos=True, ref_ids=ref_ids,
    )
    val_off = [p["id"] for p in off["x_val"]]
    val_on = [p["id"] for p in on["x_val"]]
    assert val_off == val_on
    assert val_off
    assert not on["x_val_cn_apos"] or all(str(p["id"]).startswith("CNp") for p in on["x_val_cn_apos"])
    assert len(on["x_train"]) > len(off["x_train"])
    all_ids = {str(p["id"]) for p in on["x_train"] + on["x_val"] + on["x_val_cn_apos"]}
    assert all_ids.isdisjoint(ref_ids)
