"""Unit tests for the 2026-10-28 ADNI z-score / zsym / reweighting driver helpers."""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("optuna")

EXP_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "experiments",
    "2026_10_28_ADNI_lse_rescale_reweighting",
)
if not os.path.exists(os.path.join(EXP_DIR, "run_comind_optuna_zsym.py")):
    pytest.skip("experiment driver not present (experiments/ is untracked)", allow_module_level=True)
# Other experiment folders ship helpers with the same module names.
for _m in ("comind_init", "comind_bic", "comind_identifiability"):
    sys.modules.pop(_m, None)
sys.path.insert(0, EXP_DIR)

_spec = importlib.util.spec_from_file_location(
    "zsym_driver", os.path.join(EXP_DIR, "run_comind_optuna_zsym.py")
)
drv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(drv)
ci = sys.modules["comind_init"]

X_COLS = ["ctx-lh-a_wscore", "ctx-rh-b_wscore", "Left-Hippocampus_wscore", "MOCA", "ADAS13"]


def test_clinical_weight_vector_share():
    assert drv.clinical_weight_vector(X_COLS, None) is None
    for rho in (0.25, 0.5):
        w = drv.clinical_weight_vector(X_COLS, rho)
        assert np.allclose(w[:3], 1.0)
        assert np.isclose(w[3:].sum() / w.sum(), rho)
        assert np.isclose(w[3], (rho / (1 - rho)) * (3 / 2))
    with pytest.raises(ValueError):
        drv.clinical_weight_vector(X_COLS, 1.0)


def test_clinical_to_z_uses_training_first_visits():
    rows = []
    for pid, (moca0, adas0) in enumerate([(28, 10), (24, 20), (20, 30), (10, 60)]):
        for t, d in ((0.0, 0), (1.0, -2)):
            rows.append({"PTID": f"S{pid}", "dt": t, "MOCA__raw": moca0 + d,
                         "ADAS13__raw": adas0 - d, "MOCA": np.nan, "ADAS13": np.nan})
    df = pd.DataFrame(rows)
    train = ["S0", "S1", "S2"]
    out, info = drv.clinical_to_z(df, train)
    first = df[(df.dt == 0) & df.PTID.isin(train)]
    mu, sd = (-first.MOCA__raw).mean(), (-first.MOCA__raw).std(ddof=0)
    assert np.isclose(info["clinical_z_mean_MOCA"], mu)
    assert np.allclose(out.MOCA, (-df.MOCA__raw - mu) / sd)
    # Worse is positive for both scores.
    assert out.loc[out.PTID == "S3", "MOCA"].min() > 0
    assert out.loc[out.PTID == "S3", "ADAS13"].min() > 0
    t0 = out[(out.dt == 0) & out.PTID.isin(train)]
    assert np.isclose(t0.ADAS13.mean(), 0.0) and np.isclose(t0.ADAS13.std(ddof=0), 1.0)


def test_zsym_init_patterns_and_null_sse():
    p = {"X_obs": np.array([[-2.0, 0.0, 4.0, np.nan, 1.0]])}
    prof = ci._patient_mean_profile(p, np.zeros(5), zsym_s0=2.0)
    assert np.allclose(prof, [0.0, 0.5, 1.0, 0.0, 0.75])
    w = np.array([1.0, 1.0, 1.0, 1.0, 4.0])
    assert np.isclose(ci.null_sse([p], np.zeros(5)), 4 + 0 + 16 + 1)
    assert np.isclose(ci.null_sse([p], np.zeros(5), w), 4 + 0 + 16 + 4)
    init = ci.initial_values(1, [p] * 4, 2, np.zeros(5), [0, 1, 2], 0.1, 0, zsym_s0=2.0)
    assert np.allclose(init["initial_s"], 2.0)


def test_subtype_by_apoe4_cramers_v():
    pats = [{"APOE4": np.array([a])} for a in [0, 0, 1, 1, 2, 2, np.nan]]
    z = np.array([0, 0, 1, 1, 1, 1, 0])
    tab = drv.subtype_by_apoe4(z, pats, 2)
    assert tab["apoe4_levels"] == ["0", "1", "2"]
    assert tab["table"] == [[2, 0, 0], [0, 2, 2]]
    assert np.isclose(tab["cramers_v"], 1.0)
    assert tab["n_unknown"] == 1


def test_study_tag_distinguishes_arms():
    tags = {
        drv.connectome_tag("K.csv", fit_s=True, intercept=False, input_mode="zscore",
                           obs_model=om, clinical_weight_share=rho)
        for om, rho in (("scale", None), ("zsym", None), ("zsym", 0.25), ("zsym", 0.5))
    }
    assert len(tags) == 4
    assert drv.connectome_tag("K.csv") == "K_cnmean_int"
