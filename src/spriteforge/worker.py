"""Watch-folder job worker — the 3080 box's remote-trigger mode.

Point it at a directory (typically an SMB share the Mac can write to, or a
Syncthing folder) and it executes whatever lands there:

    jobs/slime.json    a plan, as emitted by `spriteforge plan` (an optional
                       top-level "seed" key is honoured)
    jobs/tree.txt      a bare prompt — planned via the director on arrival

Results land in jobs/done/<name>/ (sprite, animations, sheet, status.json);
failures in jobs/failed/<name>/ with the traceback. Drop a file named STOP in
the jobs dir to shut the worker down cleanly after the current job.

The SD pipe is built once and kept warm across jobs — model load is the slow
part (the handover's cold-start concern), generation is seconds.
"""

from __future__ import annotations

import json
import shutil
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

from .director import Plan, execute_plan, plan_asset

JOB_SUFFIXES = (".json", ".txt")


@dataclass
class JobResult:
    name: str
    status: str          # "done" | "failed"
    out_dir: Path
    error: str = ""


def _load_job(path: Path, offline: bool) -> tuple[Plan, int]:
    """A job file is a plan JSON (optional extra "seed" key) or a prompt txt."""
    if path.suffix == ".txt":
        prompt = path.read_text().strip()
        if not prompt:
            raise ValueError("empty prompt file")
        return plan_asset(prompt, offline=offline), 0
    data = json.loads(path.read_text())
    seed = int(data.pop("seed", 0))
    return Plan(**data), seed


def process_job(
    path: Path,
    jobs_dir: Path,
    pipe,
    palette=None,
    offline: bool = False,
) -> JobResult:
    """Execute one job file; move it and its outputs to done/ or failed/."""
    name = path.stem
    try:
        plan, seed = _load_job(path, offline)
        out_dir = jobs_dir / "done" / name
        results = execute_plan(plan, out_dir, palette=palette, seed=seed, pipe=pipe)
        (out_dir / "status.json").write_text(json.dumps(
            {"status": "done", "plan": json.loads(plan.to_json()),
             "seed": seed,
             "outputs": {k: str(v) for k, v in results.items()}},
            indent=2) + "\n")
        shutil.move(str(path), out_dir / path.name)
        return JobResult(name=name, status="done", out_dir=out_dir)
    except Exception as e:  # noqa: BLE001 — one bad job must not kill the worker
        fail_dir = jobs_dir / "failed" / name
        fail_dir.mkdir(parents=True, exist_ok=True)
        (fail_dir / "error.txt").write_text(traceback.format_exc())
        if path.exists():
            shutil.move(str(path), fail_dir / path.name)
        return JobResult(name=name, status="failed", out_dir=fail_dir, error=str(e))


def pending_jobs(jobs_dir: Path) -> list[Path]:
    """Job files waiting in the top level of the jobs dir, oldest first."""
    jobs = [p for p in jobs_dir.iterdir()
            if p.is_file() and p.suffix in JOB_SUFFIXES]
    return sorted(jobs, key=lambda p: (p.stat().st_mtime, p.name))


def run_worker(
    jobs_dir: str | Path,
    poll: float = 2.0,
    palette=None,
    offline: bool = False,
    pipe_factory=None,
    once: bool = False,
) -> list[JobResult]:
    """The worker loop. `pipe_factory` is injectable for tests; by default the
    SD pipeline is built lazily on the first job and reused (kept warm)."""
    jobs_dir = Path(jobs_dir)
    jobs_dir.mkdir(parents=True, exist_ok=True)
    stop_file = jobs_dir / "STOP"

    pipe = None
    results: list[JobResult] = []
    print(f"worker: watching {jobs_dir} (drop a STOP file to shut down)")
    while True:
        if stop_file.exists():
            print("worker: STOP file found — shutting down")
            stop_file.unlink()
            break
        batch = pending_jobs(jobs_dir)
        for path in batch:
            if pipe is None:
                if pipe_factory is not None:
                    pipe = pipe_factory()
                else:
                    from .generate import build_pipe

                    pipe = build_pipe()  # built once, kept warm across jobs
            result = process_job(path, jobs_dir, pipe,
                                 palette=palette, offline=offline)
            print(f"worker: {result.name} -> {result.status}"
                  + (f" ({result.error})" if result.error else ""))
            results.append(result)
        if once:
            break
        if not batch:
            time.sleep(poll)
    return results
