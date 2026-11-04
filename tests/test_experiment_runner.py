"""Config loading and one end-to-end runner trial on synthetic data (no ADNI / PPMI files)."""
import json
import os
import types

import numpy as np
import pytest

optuna = pytest.importorskip("optuna")

from COMIND_transformer.experiment import runner
from COMIND_transformer.experiment.config import load_config
from COMIND_transformer.experiment.data import Dataset, biomarker_weights
from COMIND_transformer.experiment.selection import select_z
from COMIND_transformer.subtyping_em_transformer import SubtypingEM

from test_zsym_weights import _zsym_toy

CONFIG = """
dataset: adni
data_csv: $COMIND_TEST_DATA/fake.csv
connectome: $COMIND_TEST_DATA/K.csv
n_subtypes: 2
estimator: {max_iter: 3, t_max: 30.0, step: 0.05, fit_s: true, s_lower_value: 1.0, s_upper: 3.0,
            obs_model: zsym, stop_rel_tol: 1e-4}
n_init: 2
objective: cv_lse
search_space: {lambda_f: [1e-3, 1e-1, true]}
n_trials: 1
n_startup_trials: 1
arms:
  S0: {estimator: {obs_model: scale}}
  S2: {clinical_weight_share: 0.25}
"""


@pytest.fixture
def config_path(tmp_path, monkeypatch):
    monkeypatch.setenv("COMIND_TEST_DATA", str(tmp_path))
    path = tmp_path / "exp" / "config.yaml"
    path.parent.mkdir()
    path.write_text(CONFIG)
    return str(path)


def test_arms_merge_and_coerce(config_path, tmp_path):
    with pytest.raises(ValueError, match="arms"):
        load_config(config_path)
    s0 = load_config(config_path, arm="S0", n_subtypes=3)
    s2 = load_config(config_path, arm="S2")
    assert s0.estimator["obs_model"] == "scale" and s0.estimator["max_iter"] == 3
    assert s0.n_subtypes == 3 and s0.arm == "S0"
    assert s2.estimator["obs_model"] == "zsym" and s2.clinical_weight_share == 0.25
    assert s2.estimator["stop_rel_tol"] == 1e-4  # "1e-4" is a string in YAML
    assert s2.data_csv == str(tmp_path / "fake.csv")
    assert s2.estimator["backend"] == "jax"  # ESTIMATOR_DEFAULTS


def test_biomarker_weights_share():
    groups = {"imaging": np.arange(82), "clinical": np.array([82, 83])}
    w = biomarker_weights(groups, 0.25)
    assert w[groups["clinical"]].sum() / w.sum() == pytest.approx(0.25)
    assert biomarker_weights(groups, None) is None


@pytest.fixture
def data(monkeypatch):
    toy = _zsym_toy([2.0] * 4, n_pat=30)
    pats = toy["pats"]
    for p in pats:
        p["id"] = f"S{p['id']:03d}"
    data = Dataset(x_train=pats[:24], x_val=pats[24:], K=toy["K"], x_cols=["a", "b", "c", "clin"],
                   groups={"imaging": np.arange(3), "clinical": np.array([3])}, info={"n_train": 24})
    fake = types.SimpleNamespace(load=lambda *a, **k: data,
                                 external_checks=lambda est, d, b, z: {"n_val_placed": int(len(z))})
    monkeypatch.setitem(runner.LOADERS, "adni", fake)
    return data


def test_one_trial_end_to_end(config_path, data):
    cfg = load_config(config_path, arm="S2")
    exp_dir = os.path.dirname(config_path)
    study = runner.run_worker(cfg, exp_dir, config_path=config_path)
    (trial,) = study.trials
    assert trial.state == optuna.trial.TrialState.COMPLETE
    assert trial.value == pytest.approx(trial.user_attrs["cv_lse"])
    for key in ("cv_cvic", "cv_forecast", "val_lse", "val_mse_clinical", "bic_gauss", "f_kkt",
                "restarts", "s_median_all", "n_val_placed"):
        assert key in trial.user_attrs, key
    assert len(trial.user_attrs["cv_lse_folds"]) == 3
    json.dumps(trial.user_attrs)

    out = os.path.join(exp_dir, "S2", "k2")
    manifest = json.load(open(os.path.join(out, "manifest.json")))
    assert manifest["resolved_config"]["clinical_weight_share"] == 0.25
    assert manifest["uses_cv"] and len(manifest["git"]["commit"]) == 40
    assert "lambda_f_max" in json.load(open(os.path.join(out, "lambda_f_max.json")))

    npz = os.path.join(out, "trials", "0000.npz")
    est = SubtypingEM.load(npz)
    saved = np.load(npz)
    assert saved["val_ids"].tolist() == [p["id"] for p in data.x_val]
    np.testing.assert_allclose(est.transform(data.x_val)["beta"], saved["val_beta"])
    assert est.lambda_f == pytest.approx(trial.params["lambda_f"])
    assert saved["cv_loglik"].shape == (24,) and sorted(set(saved["cv_fold"])) == [0, 1, 2]
    assert trial.user_attrs["cv_cvic"] == pytest.approx(-2 * saved["cv_loglik"].sum())
    assert sum(trial.user_attrs["cv_cvic_folds"]) == pytest.approx(trial.user_attrs["cv_cvic"])


def test_select_z_across_studies(config_path, data):
    exp_dir = os.path.dirname(config_path)
    for z in (1, 2):
        runner.run_worker(load_config(config_path, arm="S0", n_subtypes=z), exp_dir, config_path=config_path)
    res = select_z(exp_dir, "S0", n_draw=2000)
    assert res["candidates"] == [1, 2] and res["selected_n_subtypes"] in (1, 2)
    assert res["n_subjects"] == 24 and res["n_folds"] == 3 and res["best_by"] == "cv_cvic"
    for z in (1, 2):
        assert res["cvic"][z] == pytest.approx(res["best_trials"][z]["cv_cvic"])
    assert len(res["tests"]) == 1
    json.dumps(res)

    from COMIND_transformer.experiment.rescore import rescore_fold
    with pytest.raises(FileNotFoundError):
        select_z(exp_dir, "S0", rescored=True, n_draw=2000)
    for z in (1, 2):
        for fold in range(3):
            rescore_fold(os.path.join(exp_dir, "S0", f"k{z}"), fold)
    fresh = select_z(exp_dir, "S0", rescored=True, n_draw=2000)
    assert fresh["rescored"] and fresh["selected_n_subtypes"] in (1, 2)
    for z in (1, 2):
        assert fresh["cvic"][z] == pytest.approx(fresh["best_trials"][z]["rescored_cvic"])
        assert fresh["cvic"][z] != pytest.approx(res["cvic"][z])  # different folds, different score
    json.dumps(fresh)


def test_results_loader(config_path, data):
    from COMIND_transformer.experiment import results as R
    exp_dir = os.path.dirname(config_path)
    runner.run_worker(load_config(config_path, arm="S2"), exp_dir, config_path=config_path)
    out = os.path.join(exp_dir, "S2", "k2")
    df = R.trials_table(out)
    assert list(df.index) == [0] and {"bic", "cv_cvic", "lambda_f"} <= set(df.columns)
    t = R.load_trial(out, 0)
    assert t.subjects.split.value_counts().to_dict() == {"train": 24, "val": 6}
    np.testing.assert_allclose(t.subjects.query("split == 'val'").beta, t.npz["val_beta"])
    t_span, traj = R.trajectories(t.est)
    assert traj.shape == (2, len(t_span), 4)
    assert len(R.visits(t)) == sum(np.isfinite(p["X_obs"]).sum() for p in data.x_train + data.x_val)
    assert len(R.visit_metadata(t)) == sum(len(p["dt"]) for p in data.x_train + data.x_val)


def test_lock_error_is_retried(monkeypatch):
    calls = []
    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("Error: did not possess lock")
        return "ok"
    monkeypatch.setattr(runner.time, "sleep", lambda s: None)
    assert runner._with_lock_retry(flaky) == "ok" and len(calls) == 3
    with pytest.raises(ValueError):
        runner._with_lock_retry(lambda: (_ for _ in ()).throw(ValueError("not a lock error")))
