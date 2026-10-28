"""Shared kube-job submission for the pipeline step scripts in ``scripts/``.

Requires the `job` extra (kube-jobs):
    uv sync --extra job

Edit the TODOs below (repository URL, image), then run any of the per-step
scripts, or ``submit_pipeline.py`` for the whole chain.

The cluster username is taken from the kube config (as ``submit_job``
resolves it) — it becomes the ``JOB_USER``/``MLFLOW_TRACKING_USERNAME`` env
vars inside the job, so no username needs to be configured here.
"""

from __future__ import annotations

import time

from kube_jobs import storage, submit_job
from kube_jobs.cli.job import delete_job

from pipeline.common import kube_username


REPO_URL = "https://github.com/jiribuchta/prov-ml-template.git"
BRANCH = "feature/provenance-pipeline"

IMAGE = "cerit.io/jiri_buchta/base-test-cuda:0.0.7"  # TODO: your image

# The experiment config the jobs run: configs/experiment/<EXPERIMENT>.yaml
EXPERIMENT = "mammaprint"


def submit_steps(steps: list[str], cpu: int = 2, memory: str = "4Gi") -> None:
    """Submit one kube-job running the given pipeline steps sequentially."""
    # k8s names are DNS-1123 — no underscores (the step itself keeps them)
    step = steps[0].replace("_", "-")
    job_name = "provenance-pipeline" if len(steps) > 1 else f"provenance-{step}"
    # rerunnable: remove the previous run's job (submit_job 409s on existing
    # names; it suffixes the job name with the username itself)
    delete_job(f"{job_name}-{kube_username()}", public=False)
    time.sleep(5)
    submit_job(
        job_name=job_name,
        public=False,
        image=IMAGE,
        cpu=cpu,
        memory=memory,
        script=[
            f"git clone {REPO_URL} workdir",
            "cd workdir",
            f"git checkout {BRANCH}",
            # the job runs the pipeline, not job submission — never refresh
            # the kube-jobs git source (private GitLab, unreachable in-cluster)
            "export UV_FROZEN=1",
            "uv sync",
            "export PYTHONUNBUFFERED=1",
            *_chain_commands(steps),
        ],
        storage=[storage.secure.DATA, storage.secure.PROJECTS],
    )


def _chain_commands(steps: list[str]) -> list[str]:
    """One command per step; each step's printed pins feed the next.

    Steps print their pins as indented 'dataset.*=...' hint lines — extracted
    from the job log and passed as overrides, so a fresh chain runs without
    a pre-pinned config and every run is still explicitly defined.
    """
    pins = "$(grep -E '^[[:space:]]+dataset\\.' job.log | tr -d ' ')"
    if len(steps) == 1:
        return [
            f"uv run python -m pipeline pipeline={steps[0]} +experiment={EXPERIMENT}"
        ]
    commands = ["set -eo pipefail"]
    for i, step in enumerate(steps):
        overrides = "" if i == 0 else f" {pins}"
        tee = "2>&1 | tee job.log" if i == 0 else "2>&1 | tee -a job.log"
        commands.append(
            f"uv run python -m pipeline pipeline={step} "
            f"+experiment={EXPERIMENT}{overrides} {tee}"
        )
    return commands
