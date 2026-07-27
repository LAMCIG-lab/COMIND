# Example COMIND experiment template

Generic grid-search driver for **already cleaned** longitudinal data. Dataset-
specific cleaning (winsorize, sign flips, lognorm, etc.) stays upstream — this
folder only runs `parse_data` → `build_connectome` → `SubtypingEM`.

## Files

| File | Role |
|------|------|
| `run_comind.py` | Edit the CONFIG block; run one grid candidate |
| `submit_comind.pbs` | Torque/PBS array wrapper |

Outputs go under `results/` and PBS stdout under `logs/` (both gitignored).

## Required CSV

Long format, one row per visit:

| Column role | Config knobs | Rules |
|-------------|--------------|--------|
| Subject id | `ID_COL` (default `subj_id`) | No missing values |
| Time | `TIME_COL` (default `time`) | **Already in years**; no missing values |
| Biomarkers | `X_COLS` | **Non-negative**, no missing values; higher = more severe |
| Clinical (optional) | `CLINICAL_COLS` | Used only for beta init / cog prior; leave `[]` if unused |
| Metadata (optional) | `METADATA_COLS` | Passthrough; NaN allowed |

The connectome CSV (`CONNECTOME_PATH`) must be square with matching row/column
names. Names that appear in `X_COLS` but not in the connectome are padded as
disconnected nodes (`pad_missing=True`).

## Setup

1. Install COMIND in your env (`pip install -e /path/to/COMIND` or set `PYTHONPATH`).
2. Edit `run_comind.py` CONFIG: `CSV_PATH`, `CONNECTOME_PATH`, `X_COLS`, grid.
3. Edit `submit_comind.pbs`: `EXP_DIR`, `#PBS -o` logs path, `conda activate YOUR_ENV`.
4. Align the PBS array with the grid (see below).

## Confirm grid size before submitting

```bash
cd experiment_pipeline/example_experiment
python run_comind.py
# prints: Grid: <n_hyper> hyper combos x <K> n_subtypes = <N> total candidates
```

Set `#PBS -t 0-$((N-1))%60` to match. The **default** CONFIG in this repo is:

- 108 hyperparameter combinations × 3 subtype counts = **324** candidates  
- PBS array: `#PBS -t 0-323%60`

If you change `param_grid_hyper` or `n_subtypes_list`, recompute and update `#PBS -t`.

## Local smoke test

```bash
PBS_ARRAYID=0 python run_comind.py
```

Uses placeholder paths until you point CONFIG at real files — expect a file-not-found until then.
