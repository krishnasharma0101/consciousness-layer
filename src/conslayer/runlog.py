"""Run logging: every experiment writes results/<run_id>/ with config.json (config, seed, git hash,
package versions, models, start/end time IST), CSVs and plots, so every number in LAB_NOTEBOOK.md
traces to a file.

Rule: refuse to start on a dirty git tree (tracked files modified, or untracked files under src/,
experiments/, tests/). Override with allow_dirty=True, in which case the full `git diff` is saved to
results/<run_id>/dirty.diff and config.json records dirty=true.

Workflow (see LAB_NOTEBOOK.md): write the pre-registration -> commit -> run -> log results -> commit.
"""
from __future__ import annotations

import datetime as _dt
import importlib.metadata as _md
import json
import os
import platform
import subprocess
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
IST = _dt.timezone(_dt.timedelta(hours=5, minutes=30), "IST")
_PKGS = ["numpy", "scipy", "pandas", "networkx", "torch", "transformers", "sentence-transformers",
         "rank-bm25", "scikit-learn", "matplotlib"]


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


def now_ist() -> str:
    return _dt.datetime.now(IST).isoformat(timespec="seconds")


def git_state() -> dict:
    status = _git("status", "--porcelain")
    dirty_lines = []
    for line in status.splitlines():
        path = line[3:]
        if line.startswith("??"):
            if path.startswith(("src/", "experiments/", "tests/")) and not path.endswith("/__pycache__/"):
                dirty_lines.append(line)
        else:
            dirty_lines.append(line)
    return {"commit": _git("rev-parse", "HEAD"), "short": _git("rev-parse", "--short", "HEAD"),
            "dirty": bool(dirty_lines), "dirty_files": dirty_lines}


def package_versions() -> dict:
    out = {}
    for p in _PKGS:
        try:
            out[p] = _md.version(p)
        except _md.PackageNotFoundError:
            out[p] = None
    return out


class Run:
    def __init__(self, run_id: str, config: dict, seed, models: dict | None = None,
                 allow_dirty: bool = False, results_root: str | None = None, check_git: bool = True):
        self.run_id = run_id
        self.dir = os.path.join(results_root or os.path.join(REPO, "results"), run_id)
        gs = git_state() if check_git else {"commit": None, "short": None, "dirty": False, "dirty_files": []}
        if gs["dirty"] and not allow_dirty:
            raise RuntimeError(f"git tree is dirty, refusing to run {run_id}: {gs['dirty_files']}. "
                               "Commit first, or pass allow_dirty=True (diff will be saved).")
        if os.path.exists(os.path.join(self.dir, "config.json")):
            raise RuntimeError(f"{self.dir} already has a config.json; run IDs are never reused.")
        os.makedirs(self.dir, exist_ok=True)
        if gs["dirty"]:
            with open(os.path.join(self.dir, "dirty.diff"), "w") as f:
                f.write(_git("diff", "HEAD"))
        self._t0 = time.time()
        self.meta = {"run_id": run_id, "start_ist": now_ist(), "end_ist": None, "wall_clock_s": None,
                     "status": "running", "git": gs, "command": " ".join(sys.argv),
                     "python": platform.python_version(), "platform": platform.platform(),
                     "cpu_count": os.cpu_count(), "packages": package_versions(), "seed": seed,
                     "models": models or {}, "config": config, "files": [], "notes": []}
        self._write()

    def path(self, name: str) -> str:
        return os.path.join(self.dir, name)

    def _register(self, name: str) -> str:
        if name not in self.meta["files"]:
            self.meta["files"].append(name)
        self._write()
        return self.path(name)

    def save_csv(self, name: str, df) -> str:
        import pandas as pd
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        df.to_csv(self.path(name), index=False)
        return self._register(name)

    def save_json(self, name: str, obj) -> str:
        with open(self.path(name), "w") as f:
            json.dump(obj, f, indent=2, default=_default)
        return self._register(name)

    def savefig(self, name: str, fig) -> str:
        fig.savefig(self.path(name), dpi=130, bbox_inches="tight")
        return self._register(name)

    def note(self, msg: str) -> None:
        self.meta["notes"].append(f"{now_ist()} {msg}")
        self._write()

    def finish(self, status: str = "done", summary: dict | None = None) -> None:
        self.meta.update(end_ist=now_ist(), wall_clock_s=round(time.time() - self._t0, 1), status=status)
        if summary is not None:
            self.meta["summary"] = summary
        self._write()

    def _write(self) -> None:
        with open(os.path.join(self.dir, "config.json"), "w") as f:
            json.dump(self.meta, f, indent=2, default=_default)


def _default(o):
    try:
        import numpy as np
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    if hasattr(o, "__dict__"):
        return o.__dict__
    return str(o)
