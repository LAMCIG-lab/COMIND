"""Unit tests for 2026-10-27 min-shift / z-score / Gaussian-BIC flags."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("optuna")

from optuna.trial import TrialState

EXP_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "experiments",
    "2026_10_27_ADNI_minshift_rerun",
)
sys.path.insert(0, EXP_DIR)

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "noshift_driver", os.path.join(EXP_DIR, "run_comind_optuna_bic_noshift.py")
)
drv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(drv)

from comind_bic import gaussian_bic  # noqa: E402

IMG = ["ctx-lh-a_wscore", "ctx-rh-b_wscore", "Left-Hippocampus_wscore"]
COMBAT = [drv.combat_col_for_wscore(c) for c in IMG]


def _row(pid, t, dx_bl, img, *, dx=None, moca=26.0, adas=15.0, mmse=28.0, amyloid="A+",
         apoe=0.0, wscore_ref=False, **extra):
    row = dict(PTID=pid, dt=float(t), DX_bl=dx_bl, DX=dx or dx_bl, MOCA=moca, ADAS13=adas,
               MMSE=mmse, APOE4=apoe, amyloid=amyloid, wscore_ref=wscore_ref)
    row.update({c: v for c, v in zip(IMG, img)})
    row.update(extra)
    return row


def test_imaging_scale_splits_from_mask():
    df = pd.DataFrame([_row("AD0", 0, "AD", [-2.0, -2.0, 6.0])])
    out5, info5 = drv.preprocess_imaging(df.copy(), IMG, zero_mode="cnmean")
    out10, info10 = drv.preprocess_imaging(
        df.copy(), IMG, zero_mode="cnmean", imaging_scale=10.0
    )
    assert info5["imaging_scale"] == pytest.approx(drv.Z_OUTLIER_SD)
    assert info10["imaging_scale"] == pytest.approx(10.0)
    assert info5["n_outlier_cells"] == info10["n_outlier_cells"] == 1
    assert np.isnan(out5.loc[0, IMG[2]]) and np.isnan(out10.loc[0, IMG[2]])
    np.testing.assert_allclose(
        out10.loc[0, IMG[:2]].to_numpy(dtype=float),
        out5.loc[0, IMG[:2]].to_numpy(dtype=float) * 5.0 / 10.0,
    )


def test_shift_from_train_min_uses_train_rows_only():
    rows = []
    for t in (0.0, 1.0):
        rows.append(_row("TR", t, "AD", [-1.0, -1.0, -1.0], amyloid="A+"))
        rows.append(_row("VA", t, "AD", [-4.0, -4.0, -4.0], amyloid="A+"))
    df = pd.DataFrame(rows)
    out, info = drv.preprocess_imaging(
        df, IMG, zero_mode="minshift", shift_from_train=True, train_ids=["TR"],
        imaging_scale=10.0,
    )
    # flipped TR = 1, VA = 4; min from train = 1; VA scaled (4-1)/10 = 0.3
    # healthiest (train) goes to 0; VA stays positive
    tr = out.loc[out["PTID"] == "TR", IMG].to_numpy(dtype=float)
    va = out.loc[out["PTID"] == "VA", IMG].to_numpy(dtype=float)
    np.testing.assert_allclose(tr, 0.0, atol=1e-12)
    np.testing.assert_allclose(va, 0.3, atol=1e-12)
    np.testing.assert_allclose(info["per_biomarker_min"], 1.0)
    np.testing.assert_allclose(info["per_biomarker_min_sd_units"], -1.0)
    assert info["median_zero_offset_sd"] == pytest.approx(-1.0)


def test_shift_from_train_can_go_slightly_negative_on_held_out():
    rows = []
    for t in (0.0, 1.0):
        rows.append(_row("TR", t, "AD", [-2.0, -2.0, -2.0]))
        rows.append(_row("VA", t, "AD", [1.0, 1.0, 1.0]))  # flipped = -1 < train min 2
    df = pd.DataFrame(rows)
    out, _ = drv.preprocess_imaging(
        df, IMG, zero_mode="minshift", shift_from_train=True, train_ids=["TR"]
    )
    va = out.loc[out["PTID"] == "VA", IMG].to_numpy(dtype=float)
    assert np.all(va < 0)


def test_clinical_scaling_instrument_skips_cn_mean():
    rows = [
        _row("REF", 0, "CN", [0.0] * 3, moca=28.0, adas=8.5, amyloid="A-", wscore_ref=True),
        _row("REF", 1, "CN", [0.0] * 3, moca=28.0, adas=8.5, amyloid="A-", wscore_ref=True),
        _row("AD0", 0, "AD", [-1.0] * 3, moca=20.0, adas=33.0, amyloid="A+"),
        _row("AD0", 1, "AD", [-1.0] * 3, moca=20.0, adas=33.0, amyloid="A+"),
    ]
    df = pd.DataFrame(rows)
    pre = drv.preprocess_frame(
        df, zero_mode="cnmean", clinical_scaling="instrument", include_cn_amyloid_pos=False,
        ref_ids={"REF"},
    )
    assert pre["clinical_cn_mean"] == {}
    ad = pre["df"][pre["df"]["PTID"] == "AD0"].iloc[0]
    assert ad["MOCA"] == pytest.approx((30.0 - 20.0) / 30.0)
    assert ad["ADAS13"] == pytest.approx(33.0 / 85.0)


def test_zscore_reference_mean0_sd1():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(20):
        raw = rng.normal(5.0, 2.0, 3)
        for t in (0.0, 1.0):
            row = _row(
                f"REF{i}", t, "CN", [0.0, 0.0, 0.0], amyloid="A-", wscore_ref=True,
            )
            for c, v in zip(COMBAT, raw):
                row[c] = float(v)
            rows.append(row)
    for t in (0.0, 1.0):
        row = _row("AD0", t, "AD", [0.0, 0.0, 0.0], amyloid="A+")
        for c in COMBAT:
            row[c] = 9.0
        rows.append(row)
    df = pd.DataFrame(rows)
    out, info = drv.apply_zscore_imaging(df, IMG)
    assert info["zscore_applied"] is True
    ref = out["wscore_ref"].astype(str).isin(["True", "true", "1"])
    z = out.loc[ref, IMG].to_numpy(dtype=float)
    np.testing.assert_allclose(np.nanmean(z, axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.nanstd(z, axis=0, ddof=0), 1.0, atol=1e-12)


def test_combat_col_mapping():
    assert drv.combat_col_for_wscore("Left-Hippocampus_wscore") == "log_Left-Hippocampus.combat"
    assert drv.combat_col_for_wscore("ctx-lh-entorhinal_wscore") == "log_ctx-lh-entorhinal.combat"


def test_cn_apos_in_main_split_uses_dx_amyloid_labels():
    rows = []
    for i in range(8):
        for t in (0.0, 1.0):
            rows.append(_row(f"AD{i}", t, "AD", [-1.0] * 3, amyloid="A+", apoe=i % 2))
            rows.append(_row(f"MCI{i}", t, "MCI", [-1.0] * 3, amyloid="A+", apoe=i % 2))
            rows.append(_row(f"CN{i}", t, "CN", [-0.2] * 3, amyloid="A+", apoe=i % 2))
    df = pd.DataFrame(rows)
    tr, va, va_cn = drv.split_subject_ids_from_frame(
        df, include_cn_amyloid_pos=True, cn_apos_in_main_split=True,
        cn_apos_ids=[f"CN{i}" for i in range(8)],
    )
    assert va_cn == []
    assert set(tr + va) == {f"AD{i}" for i in range(8)} | {f"MCI{i}" for i in range(8)} | {f"CN{i}" for i in range(8)}
    assert any(i.startswith("CN") for i in tr)
    assert any(i.startswith("CN") for i in va)


def test_gaussian_bic_known_values():
    sse = np.array([10.0, 20.0])
    n = np.array([5.0, 5.0])
    bic, neg2 = gaussian_bic(sse, n, k_model=3)
    sigma2 = np.array([2.0, 4.0])
    hand_neg2 = float(np.sum(n * (np.log(2 * np.pi) + np.log(sigma2) + 1.0)))
    hand_bic = hand_neg2 + 3 * np.log(10.0)
    assert neg2 == pytest.approx(hand_neg2)
    assert bic == pytest.approx(hand_bic)


def test_budget_reached_counts_complete():
    import optuna

    study = optuna.create_study()

    def obj(trial):
        return 1.0

    study.optimize(obj, n_trials=3)
    n_complete = sum(t.state == TrialState.COMPLETE for t in study.trials)
    assert n_complete == 3
    assert drv.budget_reached(study, 3) is True
    assert drv.budget_reached(study, 2) is True
    assert drv.budget_reached(study, 4) is False
    assert drv.budget_reached(study, None) is False


def test_n_obs_per_biomarker():
    pats = [
        {"X_obs": np.array([[1.0, np.nan], [2.0, 3.0]])},
        {"X_obs": np.array([[np.nan, 4.0]])},
    ]
    np.testing.assert_array_equal(drv.n_obs_per_biomarker(pats, 2), [2, 2])
