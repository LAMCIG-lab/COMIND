# Handoff: ENIGMA-PD loader and SGE runs (zsym, CVIC, s in [2, 4])

Written 2026-11-04 on the PBS cluster (orpheus) for a new Claude Code session on LONI (SGE).
Branch `optuna_grid`. Read this whole file before writing code. Ask the user before deviating
from a decision below.

## Goal

Run the same experiment we now run on ADNI and PPMI, on ENIGMA-PD, with two arms:

- **W0**: equal biomarker weights.
- **W50**: the clinical scores together carry half of the loss (`clinical_weight_share: 0.5`).

Each arm: Z = 1..5, one Optuna study per Z, **200 trials per study in one go**, then fresh-fold
CVIC re-scoring and the ordered Tukey test to choose Z.

That needs three pieces of work: an `enigma_pd` loader in the package (the main job), SGE
submission scripts, and an experiment folder with `config.yaml` + `NOTES.md`.

## Decisions already made (by the user)

| Topic | Decision |
|---|---|
| Sites | `ppmi`, `singapore`, `stanford`, **and `oxford`** (oxford is new; the past runs used 3 sites) |
| Missing data | `missing="mask"`: out-of-range cells become NaN, the visit stays; imaging-only and clinical-only visits allowed |
| Split | NSD split with `split_reference="shared"`: val = DPMoSt's NSD-stratified PPMI val subjects; everyone else is train |
| Granularity | `full` (no lobe averaging) |
| Imaging | the ENIGMA `*_Z_predict` z-scores as they are: **no division by 5, no min-shift**, `|z| > 5` masked |
| Ventricles | **kept**, as disconnected nodes (zero rows / columns in K): 82 connected regions + 2 lateral ventricles = 84 imaging biomarkers |
| Clinical | MoCA, TD_mean, PIGD_mean (state "either": OFF, else ON), **z-scored** like our ADNI / PPMI loaders |
| Observation model | `obs_model: zsym`, **no intercept** (`fit_intercept` off; zsym ties c = -s), `fit_s: true` for every biomarker (clinical included), **s bounds [2, 4]** (zsym now starts at the middle, s = 3) |
| Budget | 200 trials per study straight away, Z = 1..5, both arms |

Everything else copies `experiments/2026_11_04_PPMI_zsym_cvic_s2to4/config.yaml`: step 0.05,
inner_ftol 1e-7, max_iter 80, min_iter 25, stop_rel_tol 1e-4, stop_patience 5,
init_scalar_k_cap 0.02, lambda_jsd 0, scalar_K_center 0.25, n_init 2, objective cv_lse,
cv_folds 3, cvic true, polish_f true, n_startup_trials 15, searched lambda_f [1e-2, 1e6],
lambda_kappa [1e-2, 1e4], lambda_scalar [1e-2, 1e3] (all log), connectome
`K_82_log_knn8_row.csv`.

## Read first (in this repo)

1. `COMIND_transformer/experiment/__init__.py`: the workflow (config -> smoke -> submit -> rescore -> select_z -> notebooks).
2. `COMIND_transformer/experiment/ppmi.py` and `adni.py`: the loader pattern to copy. A loader is
   `load(data_csv, connectome, *, t_max, seed, **cfg.data) -> Dataset` plus
   `external_checks(est, data, val_beta, val_subtypes) -> dict`.
3. `COMIND_transformer/experiment/data.py`: `Dataset`, `biomarker_groups`, `clinical_to_z`, `shift_beta`.
4. `COMIND_transformer/experiment/runner.py` (`LOADERS`) and `config.py` (`DATASETS`): register the new loader in both.
5. `experiments/2026_11_04_PPMI_zsym_cvic_s2to4/config.yaml` and `NOTES.md`: the settings to copy.
6. `scripts/submit_arms.py`, `scripts/submit.pbs`, `scripts/rescore_cvic.py`, `scripts/rescore.pbs`: the PBS versions of the SGE scripts you will write.
7. `scripts/build_analysis_notebook.py`: builds the results notebooks (ADNI / PPMI branches; add an ENIGMA-PD branch).

## Past ENIGMA-PD runs to reference (on LONI, not in git)

Folder `/ifs/loni/faculty/thompson/four_d/dsemchin/comind/experiment_pipeline/pd_all/2026_10_27_comind_minshift/`:

- `prepare_data_comind.py`: our copy of Emile's `prepare_data` (DPMoSt). **This is the data logic to port.**
  It reads `<DPMOST_BASE>/analysis_<site>/data_<site>_pd.csv`, with
  `DPMOST_BASE = /ifs/loni/faculty/thompson/four_d/edangremont/dpmost_updated/dpmost` (Emile's folder: read only).
- `run_comind_enigma_optuna_v2.py`: the old driver (min-shift, /10 scale, [0, 1] clinical, intercept options). Reference only:
  do not reuse its scaling, intercepts or saving.
- `submit_comind_v2.sge`, `submit_cvic_selection.sge`: the SGE settings that worked on LONI.
- `K_82_log_knn8_row.csv`: the connectome. Its labels already match the ENIGMA columns
  (`L_precuneus_Z_predict`, `Left-Thalamus-Proper_Z_predict`, ...; 82 regions, no ventricles).

## 1. The loader: `COMIND_transformer/experiment/enigma_pd.py`

Port the needed parts of `prepare_data_comind.py` into the package (mask mode, NSD split with the
shared val set, `_select_state`, `mask_outliers`, `_emit_unpaired_clinical`, `_coalesce_same_time`,
`_parse_nsd_stage`). The shared val set needs a second pass in `missing="drop"` mode, so keep
`drop_outliers` too. Drop the lobe-averaging and crossval code. Credit Emile's original in the docstring.

Behaviour to keep from `prepare_data` (check each one in a test):

- **Signs.** `prepare_data` multiplies every biomarker by -1 **except** names containing
  `Ventricle`, `TD` or `PIGD`. After it: cortical / subcortical z are flipped (atrophy positive),
  ventricles are left as they are (enlargement positive), MoCA arrives as **-MoCA**, TD / PIGD
  means are unchanged (higher is worse). So every biomarker is already "worse = positive".
  **Do not call `data.mask_and_flip`** (it would flip a second time); masking is already done.
- **Time.** `time` is float years (months / 12); clinical-only visits sit at MRI time +
  timedif / 12, so a subject's first visit can be a clinical-only visit with an earlier time.
  `parse_data` stores the time column as `dt` **without** subtracting the first visit, so subtract
  each subject's minimum time first (the old driver did `t - t.min()`).
- **Pairing.** Clinical stays on the MRI row when `|timedif_clinical| <= 3` months; otherwise it
  becomes its own clinical-only visit.
- **QC.** Rows with `fail == True` dropped; duplicates on (subject, time) dropped.

What the loader adds, following `ppmi.py`:

- **IDs as strings.** PPMI ids are ints, other sites strings. Check that ids are unique across
  sites; if any collide, prefix with the site (`"singapore:123"`) and say so.
- **Clinical z-scores.** Keep the instrument values in `*__raw` columns first: `MoCA__raw = -MoCA`
  (undo the flip, so 0-30), `TD_mean__raw`, `PIGD_mean__raw`. Then call
  `clinical_to_z(df, ID_COL, TIME_COL, {"MoCA": ("MoCA__raw", -1.0), "TD_mean": ("TD_mean__raw", 1.0),
  "PIGD_mean": ("PIGD_mean__raw", 1.0)}, train_ids)`: worse is positive, z against the training
  subjects' first finite value. This is the same placeholder as ADNI / PPMI (clinical z = 0 is the
  training-patient mean, not a healthy reference).
- **Biomarker order.** `x_cols` = 68 cortical + 14 subcortical (the 82 connectome names) + the two
  `*-Lateral-Ventricle_Z_predict` + `["MoCA", "TD_mean", "PIGD_mean"]`. Groups via
  `biomarker_groups(x_cols, ["MoCA", "TD_mean", "PIGD_mean"])`, so the ventricles count as imaging.
- **Connectome.** `build_connectome(connectome, x_cols, pad_missing=True)`. Unlike `ppmi.py`,
  disconnected **imaging** nodes are expected here: allow exactly the two ventricles and the three
  clinical scores to be disconnected, and raise if anything else is.
- **Beta start.** MixedLM on the training subjects' visits that have all three clinical values,
  via `fit_mixedlm_beta_from_clinical(df=...)` with columns renamed to what `build_severity_index`
  expects: `MCATOT` = raw MoCA (higher is better), `TD_score` = raw TD_mean, `PIGD_score` = raw
  PIGD_mean. Training subjects with no complete clinical visit get the median MixedLM beta (the old
  driver did the same); then `shift_beta`. Only `x_train` gets `initial_beta`.
- **Metadata** on each patient (for validation, never fitted): `SITE`, `Sex`, `Age`, `NSD_STAGE`,
  `HY`, `Classification` (TD / PIGD / Indeterminate), `UPDRSIII`, and the `*__raw` clinical values
  (pass them as `metadata_cols`, or as `clinical_cols` so they land in `p["cog"]`; say which).
- **Dataset.info** (aggregates only, goes into the manifest): train / val subject counts, visit
  counts (paired / imaging-only / clinical-only), cells masked, per-site subject counts, the
  `prepare_data` meta, the clinical z means / SDs.
- **external_checks** (never used for selection), mirroring `ppmi.py`:
  - Spearman of disease time with H&Y and NSD stage (NSD exists for PPMI only): train uses beta + last
    follow-up against the last visit's stage, val the first visit, like `ppmi.staging_spearman`
    (that one reads `NHY`; ENIGMA calls it `HY`).
  - Subtype x site and subtype x Classification tables with Cramér's V (`adni._cramers_v`).

Data location in the config: `data_csv` is a **directory** for this dataset (the DPMoSt base with
the `analysis_<site>/` folders). Keep paths under `$COMIND_DATA`: e.g. on LONI symlink
`$COMIND_DATA/enigma_pd -> <DPMOST_BASE>` and copy the connectome to
`$COMIND_DATA/connectomes/K_82_log_knn8_row.csv`. Loader options go under `data:` in the config
(`sites`, `missing`, `state`, `split_seed: 75`).

**Oxford is new.** Before anything else, check that `analysis_oxford/data_oxford_pd.csv` exists
and has the columns the others have (`participant_id`, `session_id`, `time` in months,
`timedif_clinical`, `fail`, the 84 `*_Z_predict` columns, MoCA, TD / PIGD ON / OFF, HY, Sex, Age).
If it does not, stop and ask the user.

**Show the user before fitting anything** (aggregates only, no subject IDs): train / val counts
per site, visit counts by type, fraction of cells missing per block, the clinical z means / SDs,
the input range, and which columns were chosen. `scripts/run_experiment.py <config> --arm W0
--dry-run` prints most of this; extend `Dataset.info` if needed. For reference, the old driver
expected 332 train / 96 val / 87 biomarkers for 3 sites in **drop** mode; mask mode with Oxford
will be larger, and nobody knows the exact numbers yet.

**Tests** (`tests/test_enigma_pd_loader.py`): `prepare_data` already takes a `data_base`, so
build tiny fake site CSVs in `tmp_path` and check signs (imaging flipped, ventricles not, MoCA
back to raw in `*__raw`, clinical z signs), `|z| > 5` masking, unpaired clinical visits, dt
starting at 0, string ids, ventricles disconnected in K, and that the val set comes from PPMI
NSD-staged subjects only. Then `pytest tests/` must still pass.

## 2. SGE submission

Write `scripts/submit.sge` and `scripts/rescore.sge` from the PBS versions plus what worked in
`submit_comind_v2.sge`:

- `#$ -S /bin/bash`, `#$ -q compute.q`, `#$ -l h_vmem=6G` (the old runs peaked at 2.4 GB; h_vmem
  counts virtual memory, and JAX reserves a lot, so smoke-test first), `#$ -l h_rt=24:00:00`,
  `#$ -j y`, `#$ -m n`.
- `source /usr/local/bin/.anaconda3-bashrc && conda activate comind`, the same thread / XLA
  environment as `submit.pbs`, plus `JAX_PLATFORMS=cpu`.
- `SGE_TASK_ID` starts at 1: stagger with `$(( 3 * (SGE_TASK_ID - 1) ))`.
- Pass `CONFIG`, `ARM`, `NSUB`, `COMIND_ROOT`, `COMIND_DATA` and optionally `N_TRIALS` with `-v`, as
  `submit_arms.py` already does for PBS.

Then add `--scheduler {pbs,sge}` to `scripts/submit_arms.py` and `scripts/rescore_cvic.py`:
- PBS: `-t 0-{n-1}` (`-t 0` for one task) with `submit.pbs` / `rescore.pbs`.
- SGE: `-t 1-{n}` with `submit.sge` / `rescore.sge`. The rescore script turns `SGE_TASK_ID - 1` into
  the fold number.
- SGE's `qsub` prints `Your job-array 1234.1-20:1 ("name") has been submitted`. Parse the job ID
  from that for `jobs.txt`.

Also:
- Set `worker_hours` in the config below the wall clock: 23.5 with `h_rt=24:00:00`.
  `runner.STALE_SECONDS` (26 h) then reaps trials from killed workers on the next submission.
- Ask the user how many concurrent tasks LONI allows. The old run used 66 workers per study (198 jobs).
  A reasonable split per arm is `--workers 6 12 20 28 30` for Z = 1..5.
- Monitoring differs from PBS. `qstat -u $USER` lists running and queued tasks. `qacct -j <id>`
  shows finished ones (`maxvmem`, `ru_wallclock`, `exit_status`).

## 3. Experiment folder

`experiments/<YYYY_MM_DD>_ENIGMAPD_zsym_cvic_s2to4/config.yaml`:
- `dataset: enigma_pd`, with `data_csv` and `connectome` under `$COMIND_DATA`.
- `data:` holds the loader options. `estimator:` and the search settings are as in the PPMI s2to4 config.
- `n_trials: 200`, `worker_hours: 23.5`.
- `arms: {W0: {}, W50: {clinical_weight_share: 0.5}}`. With 84 imaging and 3 clinical biomarkers, each clinical score in W50 weighs 28× an imaging region.

Write `NOTES.md` covering why the run exists, the decisions in the table above, and the cohort counts from the dry run.

## 4. Order of work

1. Use a **separate clone** on LONI (e.g. `.../dsemchin/comind_optuna_grid`), at the latest `optuna_grid`.
   The existing `.../dsemchin/comind` checkout carries local patches (`comind_intercept.patch`,
   `comind_score.patch`) for the old runs, and `pip install -e` the new clone into its own conda env or venv,
   so the old drivers keep working.
2. Check the Oxford file, then write the loader and its tests.
3. Dry run and show the user the cohort summary. Wait for an OK.
4. Write the SGE scripts. Smoke-test one trial per arm through `qsub` (`run_experiment.py --smoke`), then check `qacct` for memory.
5. Commit (`config.yaml` and `NOTES.md` only under `experiments/`) and push. Then submit W0 and W50 with `submit_arms.py --scheduler sge`.
6. When all 10 studies have 200 trials, run `rescore_cvic.py --scheduler sge` for each arm. When the 15 fold files per arm exist, run `select_z.py`.
7. Notebooks: add an ENIGMA-PD branch to `scripts/build_analysis_notebook.py`, with H&Y / NSD staging and site / Classification tables like the PPMI branch. Build `analyze_results.ipynb` (W0) and `analyze_results_W50.ipynb`.

## Ground rules

- Patient data never goes into git. Under `experiments/` only `config.yaml` and `NOTES.md` are tracked; `tests/test_repo_privacy.py` enforces it.
- Data paths stay under `$COMIND_DATA`; `ExperimentConfig` expands it, and the jobs get it through `-v`.
- No blanket `try/except`. Let a broken trial fail loudly; the runner only retries journal-lock errors.
- Commit messages end with the co-author line the session gives you. Commit or push only when the user asks.
