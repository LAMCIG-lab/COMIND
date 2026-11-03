"""
Experiment config: one ``config.yaml`` per dated experiment folder.

Top-level keys are the fields of ``ExperimentConfig``. An ``arms:`` mapping
holds per-arm overrides, merged into the top level (nested dicts merge key by
key). ``$VARS`` in ``data_csv`` / ``connectome`` are expanded from the
environment, so configs never hold absolute paths into the data store.
"""
from __future__ import annotations

import copy
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

import yaml

DATASETS = ("adni", "ppmi")
OBJECTIVES = ("cv_lse", "cv_forecast", "cvic", "bic", "bic_gauss")

# Passed to SubtypingEM unless the config's ``estimator`` overrides them.
ESTIMATOR_DEFAULTS: Dict[str, Any] = {
    "t_max": 40.0,
    "step": 0.01,
    "epsilon": 1e-2,
    "backend": "jax",
    "lambda_cog": 0.0,
    "lambda_beta": 0.0,
}

# name: [low, high, log]
DEFAULT_SEARCH_SPACE: Dict[str, list] = {
    "lambda_f": [1e-2, 1e6, True],
    "lambda_kappa": [1e-2, 1e4, True],
    "lambda_scalar": [1e-2, 1e3, True],
    "scalar_K_center": [1e-2, 1.0, True],
    "lambda_jsd": [1e-1, 1e3, True],
}


@dataclass
class ExperimentConfig:
    dataset: str
    data_csv: str
    connectome: str
    n_subtypes: int = 3
    data: Dict[str, Any] = field(default_factory=dict)  # loader options
    estimator: Dict[str, Any] = field(default_factory=dict)  # SubtypingEM kwargs
    n_init: int = 3  # restarts for the full fit; CV folds use restart 0 only
    clinical_weight_share: Optional[float] = None  # None = equal biomarker weights
    objective: str = "cv_lse"
    cv_folds: int = 3
    cvic: bool = True  # also score held-out CVIC in every CV fold (used to compare Z)
    polish_f: bool = False
    search_space: Dict[str, list] = field(default_factory=lambda: copy.deepcopy(DEFAULT_SEARCH_SPACE))
    n_trials: int = 200
    n_startup_trials: int = 20
    worker_hours: float = 23.5
    seed: int = 75
    save_histories: bool = False
    arm: str = "main"

    def __post_init__(self):
        if self.dataset not in DATASETS:
            raise ValueError(f"dataset must be one of {DATASETS}, got {self.dataset!r}")
        if self.objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}, got {self.objective!r}")
        if self.clinical_weight_share is not None and not 0.0 < self.clinical_weight_share < 1.0:
            raise ValueError(f"clinical_weight_share must be in (0, 1), got {self.clinical_weight_share}")
        # PyYAML reads "1e-2" as a string, so coerce the numbers here.
        self.search_space = {
            name: [float(lo), float(hi), bool(log)] for name, (lo, hi, log) in self.search_space.items()
        }
        self.estimator = {**ESTIMATOR_DEFAULTS, **_floats(self.estimator)}
        self.data_csv = _expand(self.data_csv)
        self.connectome = _expand(self.connectome)

    @property
    def uses_cv(self) -> bool:
        return self.objective.startswith("cv_") or self.objective == "cvic"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _expand(path: str) -> str:
    out = os.path.expanduser(os.path.expandvars(str(path)))
    if "$" in out:
        raise ValueError(f"unset environment variable in path {path!r}")
    return out


def _floats(d: Dict[str, Any]) -> Dict[str, Any]:
    """Turn numeric-looking strings ("1e-4") into floats."""
    out = {}
    for k, v in d.items():
        if isinstance(v, str):
            try:
                v = float(v)
            except ValueError:
                pass
        out[k] = v
    return out


def _merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str, arm: Optional[str] = None, **overrides) -> ExperimentConfig:
    """Read ``path``, apply arm ``arm`` and then ``overrides`` (e.g. ``n_subtypes=4``)."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    arms = raw.pop("arms", None) or {}
    if arm == "main" and not arms:
        arm = None
    if arm is not None:
        if arm not in arms:
            raise KeyError(f"arm {arm!r} not in {path} (arms: {sorted(arms)})")
        raw = _merge(raw, arms[arm])
        raw["arm"] = arm
    elif arms:
        raise ValueError(f"{path} defines arms {sorted(arms)}; pass one with --arm")
    raw.update({k: v for k, v in overrides.items() if v is not None})
    return ExperimentConfig(**raw)


def arm_names(path: str) -> list:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return list((raw.get("arms") or {}).keys()) or ["main"]
