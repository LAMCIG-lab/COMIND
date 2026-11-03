"""``manifest.json``: what code and config produced a study."""
from __future__ import annotations

import datetime as dt
import inspect
import json
import os
import platform
import subprocess
from typing import Any, Dict

PATCH_PATHS = ("*.py", "*.yaml", "*.pbs")  # never notebooks or data


def _git(root: str, *args: str) -> str:
    return subprocess.run(("git",) + args, cwd=root, capture_output=True, text=True, check=True).stdout.strip()


def git_state(root: str) -> Dict[str, Any]:
    status = _git(root, "status", "--porcelain", "--untracked-files=no")
    return {"commit": _git(root, "rev-parse", "HEAD"),
            "branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(status)}


def _versions() -> Dict[str, str]:
    import numpy, optuna, scipy, sklearn  # noqa: E401
    out = {"python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__,
           "sklearn": sklearn.__version__, "optuna": optuna.__version__}
    try:
        import jax
        out["jax"] = jax.__version__
    except ImportError:
        pass
    return out


def estimator_defaults() -> Dict[str, Any]:
    """SubtypingEM constructor defaults, so a changed default shows up between manifests."""
    from ..subtyping_em_transformer import SubtypingEM
    from .diagnostics import json_safe

    sig = inspect.signature(SubtypingEM.__init__)
    return json_safe({k: p.default for k, p in sig.parameters.items()
                      if k != "self" and isinstance(p.default, (int, float, str, bool, type(None), tuple))})


def write_manifest(path: str, cfg, config_path: str, data_info: Dict[str, Any]) -> None:
    """Write ``path`` once per study. A later worker on another commit adds ``manifest.<commit>.json``.

    With uncommitted changes, ``uncommitted.patch`` (``*.py`` / ``*.yaml`` / ``*.pbs``
    only) is written next to it.
    """
    from .diagnostics import json_safe

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    git = git_state(root)
    if os.path.exists(path):
        with open(path) as f:
            if json.load(f)["git"]["commit"] == git["commit"]:
                return
        path = os.path.join(os.path.dirname(path), f"manifest.{git['commit'][:8]}.json")
        if os.path.exists(path):
            return
    with open(config_path) as f:
        config_text = f.read()
    manifest = {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "git": git,
        "config_file": os.path.abspath(config_path),
        "config_text": config_text,
        "resolved_config": cfg.to_dict(),
        "objective": cfg.objective,
        "uses_cv": cfg.uses_cv,
        "versions": _versions(),
        "estimator_defaults": estimator_defaults(),
        "data_info": data_info,
    }
    if git["dirty"]:
        patch = _git(root, "diff", "HEAD", "--", *PATCH_PATHS)
        with open(os.path.join(os.path.dirname(path), "uncommitted.patch"), "w") as f:
            f.write(patch + "\n")
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(json_safe(manifest), f, indent=2)
    os.replace(tmp, path)
