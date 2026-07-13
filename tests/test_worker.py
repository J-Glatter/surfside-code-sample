"""Worker loop with the diffusion pipe stubbed; jobs/palette/failure paths real."""

from __future__ import annotations

import json
import sys
import types
from unittest.mock import MagicMock

import pytest
from PIL import Image

from spriteforge.director import heuristic_decider
from spriteforge.worker import pending_jobs, run_worker

from .tools import fake_torch_module


@pytest.fixture
def worker_env(monkeypatch, tmp_path):
    fake_torch = fake_torch_module()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class FakePipe:
        calls = 0

        def __call__(self, **kwargs):
            FakePipe.calls += 1
            return types.SimpleNamespace(
                images=[Image.new("RGB", (128, 128), (90, 190, 110))])

    pipe = FakePipe()
    factory = MagicMock(side_effect=lambda: pipe)
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    return jobs, factory, pipe


def _write_plan_job(jobs, name, prompt="a treasure chest", seed=None, **overrides):
    plan = heuristic_decider(prompt)
    data = json.loads(plan.to_json())
    data.update(overrides)
    if seed is not None:
        data["seed"] = seed
    (jobs / f"{name}.json").write_text(json.dumps(data))


def test_worker_processes_plan_and_prompt_jobs(worker_env):
    jobs, factory, _ = worker_env
    _write_plan_job(jobs, "chest", seed=7)
    (jobs / "tree.txt").write_text("a tree swaying in the wind\n")

    results = run_worker(jobs, pipe_factory=factory, once=True, offline=True)

    assert {r.name: r.status for r in results} == {"chest": "done", "tree": "done"}
    assert (jobs / "done" / "chest" / "sprite.png").exists()
    status = json.loads((jobs / "done" / "chest" / "status.json").read_text())
    assert status["seed"] == 7
    # prompt job went through the director: tree -> static_prop + sway
    assert (jobs / "done" / "tree" / "sway.gif").exists()
    # job files were consumed from the inbox
    assert pending_jobs(jobs) == []
    # the pipe was built exactly once and stayed warm across both jobs
    assert factory.call_count == 1


def test_worker_survives_bad_jobs(worker_env):
    jobs, factory, _ = worker_env
    (jobs / "broken.json").write_text("{not json")
    (jobs / "empty.txt").write_text("   \n")
    _write_plan_job(jobs, "ok")

    results = run_worker(jobs, pipe_factory=factory, once=True, offline=True)

    by_name = {r.name: r for r in results}
    assert by_name["broken"].status == "failed"
    assert by_name["empty"].status == "failed"
    assert by_name["ok"].status == "done"
    assert (jobs / "failed" / "broken" / "error.txt").exists()
    assert (jobs / "failed" / "broken" / "broken.json").exists()  # moved, kept


def test_worker_stop_file(worker_env):
    jobs, factory, _ = worker_env
    (jobs / "STOP").touch()
    _write_plan_job(jobs, "never")

    results = run_worker(jobs, pipe_factory=factory, poll=0.01)

    assert results == []                       # stopped before processing
    assert not (jobs / "STOP").exists()        # consumed the stop file
    factory.assert_not_called()


def test_tile_job_restores_the_shared_pipe(tmp_path):
    """After a tile job the conv patch must be rolled back for the next job.

    Uses the real torch (no fake-module fixture): the tiling patch needs real
    Conv2d layers, and torch.Generator works as-is.
    """
    torch = pytest.importorskip("torch")
    jobs = tmp_path / "jobs"
    jobs.mkdir()

    class ConvPipe:
        def __init__(self):
            self.unet = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3))

        def __call__(self, **kwargs):
            return types.SimpleNamespace(
                images=[Image.new("RGB", (64, 64), (100, 160, 90))])

    pipe = ConvPipe()
    _write_plan_job(jobs, "grass", prompt="grass texture")
    _write_plan_job(jobs, "chest", prompt="a treasure chest")

    results = run_worker(jobs, pipe_factory=lambda: pipe, once=True, offline=True)

    assert all(r.status == "done" for r in results)
    assert pipe.unet[0].padding_mode == "zeros"   # tiling rolled back
