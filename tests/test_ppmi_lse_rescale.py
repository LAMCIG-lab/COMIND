"""Unit tests for the 2026-11-02 PPMI z-score / zsym / reweighting driver helpers."""
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
    "2026_11_02_PPMI_lse_rescale_reweighting",
)
if not os.path.exists(os.path.join(EXP_DIR, "run_comind_ppmi_zsym.py")):
    pytest.skip("experiment driver not present (experiments/ is untracked)", allow_module_level=True)
# Other experiment folders ship helpers with the same module names.
for _m in ("comind_init", "comind_bic", "comind_identifiability", "zsym_core"):
    sys.modules.pop(_m, None)
sys.path.insert(0, EXP_DIR)

_spec = importlib.util.spec_from_file_location(
    "ppmi_zsym_driver", os.path.join(EXP_DIR, "run_comind_ppmi_zsym.py")
)
drv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(drv)


def test_core_uses_ppmi_clinical_columns():
    x_cols = ["L_a_Z_predict", "R_b_Z_predict", "MCATOT", "TD_score", "PIGD_score"]
    g = drv.zc.biomarker_groups(x_cols)
    assert g["clinical"].tolist() == [2, 3, 4]
    w = drv.zc.clinical_weight_vector(x_cols, 0.5)
    assert np.isclose(w[2:].sum() / w.sum(), 0.5)


def test_clinical_to_z_flips_moca_and_uses_training_first_visits():
    rows = []
    for pid, (m, td, pg) in enumerate([(29, 2, 0), (27, 5, 1), (25, 8, 2), (15, 20, 8)]):
        for t, d in ((0.0, 0), (1.0, 1)):
            rows.append({"subj_id": f"P{pid}", "time": t, "MCATOT__raw": m - d,
                         "TD_score__raw": td + d, "PIGD_score__raw": pg + d})
    df = pd.DataFrame(rows)
    out, info = drv.clinical_to_z(df, ["P0", "P1", "P2"])
    t0 = out[(out.time == 0) & out.subj_id.isin(["P0", "P1", "P2"])]
    for col in ("MCATOT", "TD_score", "PIGD_score"):
        assert np.isclose(t0[col].mean(), 0.0) and np.isclose(t0[col].std(ddof=0), 1.0)
        assert out.loc[out.subj_id == "P3", col].min() > 0  # worse is positive
    assert info["clinical_z_n_ref_MCATOT"] == 3


def test_genetics_table_and_carrier_v():
    labs = ["Sporadic PD"] * 4 + ["GBA", "GBA", "PRKN", "LRRK2"]
    pats = [{"subgroup": np.array([lab, lab], dtype=object)} for lab in labs] + [{"subgroup": None}]
    z = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0])
    tab = drv.subtype_by_genetics(z, pats, 2)
    assert tab["levels"] == ["Sporadic PD", "GBA", "PRKN", "LRRK2"]
    assert tab["table"] == [[4, 0, 0, 0], [0, 2, 1, 1]]
    assert tab["carrier_table"] == [[4, 0], [0, 4]]
    assert np.isclose(tab["carrier_cramers_v"], 1.0)
    assert tab["n_unknown"] == 1


def test_staging_spearman_visit_rule():
    pats = [{"dt": np.array([0.0, 2.0]), "NSD_STAGE": np.array(["2b", str(s)], dtype=object),
             "NHY": np.array([1.0, h])} for s, h in ((3, 1.0), (4, 2.0), (5, 3.0), (6, 3.0))]
    betas = [1.0, 2.0, 3.0, 4.0]
    tr = drv.staging_spearman(pats, betas, split="train")
    assert np.isclose(tr["nsd_train"]["rho"], 1.0) and tr["nsd_train"]["n"] == 4
    va = drv.staging_spearman(pats, betas, split="val")  # first visit: all NSD 2 -> undefined
    assert np.isnan(va["nsd_val"]["rho"])
