"""
Config-driven Optuna experiments for SubtypingEM (ADNI and PPMI).

New experiment:

1. Make ``experiments/<YYYY_MM_DD>_<name>/config.yaml``; copy the newest
   one and edit it. Write ``NOTES.md`` (why you ran it, what you found).
   Both are tracked in git; nothing else in the folder is.
2. ``export COMIND_DATA=/home/dsemchin/data``
3. ``python scripts/run_experiment.py <config> --arm S1 --dry-run``: data counts.
4. ``python scripts/run_experiment.py <config> --arm S1 --smoke``: one short
   trial in ``_smoke/``.
5. ``python scripts/submit_arms.py <config> --n-subtypes 1 2 3 4 5 --workers 16``
6. ``python scripts/summarize_study.py experiments/<dated>``: best trial per
   arm and Z.
7. ``python scripts/select_z.py experiments/<dated>``: ordered Tukey test on
   held-out CVIC across the Z studies of each arm (include Z=1 as the
   baseline candidate).

Each study's ``manifest.json`` records the git commit, the resolved config
and the SubtypingEM defaults, so ``git show <commit>`` gives the exact code.
Per-trial models load with ``SubtypingEM.load(<arm>/k<Z>/trials/NNNN.npz)``;
scores and diagnostics are in the Optuna trial ``user_attrs``.
"""
from .config import ExperimentConfig, load_config  # noqa: F401
