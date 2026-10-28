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
from kube_jobs.constants import CLUSTER
from kube_jobs.resolve_kube_config import load_kube_config
from kubernetes import client


REPO_URL = "https://github.com/jiribuchta/prov-ml-template.git"
BRANCH = "feature/provenance-pipeline"

IMAGE = "cerit.io/jiri_buchta/base-test-cuda:0.0.7"  # TODO: your image


def _username() -> str:
    """Cluster username from the kube config (as submit_job resolves it)."""
    load_kube_config(context=CLUSTER)
    review = client.AuthenticationV1Api().create_self_subject_review(
        body=client.V1SelfSubjectReview()
    )
    assert review.status and review.status.user_info and review.status.user_info.extra
    username = str(review.status.user_info.extra["username"][0]).split("@")[0]
    return username.lower().replace(" ", "-").replace("_", "-")


def submit_steps(steps: list[str], cpu: int = 2, memory: str = "4Gi") -> None:
    """Submit one kube-job running the given pipeline steps sequentially."""
    job_name = "provenance-pipeline" if len(steps) > 1 else f"provenance-{steps[0]}"
    # rerunnable: remove the previous run's job (submit_job 409s on existing
    # names; it suffixes the job name with the username itself)
    delete_job(f"{job_name}-{_username()}", public=False)
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
            *[
                f"uv run python -m pipeline pipeline={step} +experiment=provenance_pipeline"
                for step in steps
            ],
        ],
        storage=[storage.secure.DATA, storage.secure.PROJECTS],
    )
