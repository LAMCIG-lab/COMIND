"""Guard against committing subject-level data (ADNI / PPMI data use agreements)."""
import os
import shutil
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALLOWED_IN_EXPERIMENTS = ("config.yaml", "NOTES.md")
# Tracked before this guard existed; they hold per-subject beta histories and are
# listed here only so the guard can run. Review whether they belong in git.
LEGACY_DATA = {
    "results/em_best_model_68biom_full_fine.npz",
    "results/ppmi_gridsearch_results.npz",
    "results/ppmi_single_fit.npz",
    "notebooks/synthetic-experiments/synthetic_opt_sens_plot_cache.pkl",
}


def _git(*args):
    return subprocess.run(("git",) + args, cwd=ROOT, capture_output=True, text=True, check=True).stdout


@pytest.fixture(scope="module")
def tracked():
    if not os.path.isdir(os.path.join(ROOT, ".git")):
        pytest.skip("not a git checkout")
    return _git("ls-files").split()


def test_experiments_track_only_config_and_notes(tracked):
    bad = [f for f in tracked if f.startswith("experiments/")
           and not (f.count("/") == 2 and f.rsplit("/", 1)[1] in ALLOWED_IN_EXPERIMENTS)]
    assert not bad, f"tracked experiment outputs: {bad[:10]}"


def test_no_new_data_files_tracked(tracked):
    data = [f for f in tracked if f.endswith((".csv", ".npz", ".pkl", ".parquet", ".xlsx"))
            and not f.startswith("tests/data/") and f not in LEGACY_DATA]
    assert not data, f"data files tracked: {data[:10]}"


def test_gitignore_allows_only_config_and_notes(tracked):
    probe = os.path.join(ROOT, "experiments", f"_privacy_probe_{os.getpid()}")
    files = ["config.yaml", "NOTES.md", "jobs.txt", "analyze.ipynb", "S0/k3/journal.log",
             "S0/k3/trials/0000.npz", "S0/k3/manifest.json", "_smoke/S0/k3/trials/0000.npz"]
    try:
        for f in files:
            os.makedirs(os.path.dirname(os.path.join(probe, f)), exist_ok=True)
            open(os.path.join(probe, f), "w").close()
        rel = os.path.relpath(probe, ROOT)
        visible = _git("ls-files", "--others", "--exclude-standard", rel).split()
    finally:
        shutil.rmtree(probe, ignore_errors=True)
    assert sorted(os.path.basename(f) for f in visible) == sorted(ALLOWED_IN_EXPERIMENTS)
