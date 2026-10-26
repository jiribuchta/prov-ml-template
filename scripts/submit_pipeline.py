"""Submit the whole provenance pipeline as one k8s job.

Requires the `job` extra (kube-jobs from the feature/job-env-metadata branch):
    uv sync --extra job

Edit the TODOs below (repository URL, username, image, MLflow server), then:
    uv run python scripts/submit_pipeline.py

The job runs all four steps sequentially; every run in MLflow gets the
job_* tags (cluster, image, CPU/memory limits, ...) injected by kube-jobs.
"""

import time

from kube_jobs import storage, submit_job
from kube_jobs.cli.job import delete_job


JOB_NAME = "provenance-pipeline-demo"

# TODO: your repository once you create it
REPO_URL = "https://github.com/<org>/<repo>.git"
BRANCH = "feature/provenance-pipeline"

# rerunnable: remove the previous run's job (submit_job 409s on existing names)
delete_job(JOB_NAME, public=False)
time.sleep(5)

steps = ["create", "split", "embed", "cluster"]
submit_job(
    job_name=JOB_NAME,
    username="jiribuchta",  # TODO: your cluster username
    image="cerit.io/jiri_buchta/base-test-cuda:0.0.7",  # TODO: your image
    public=False,
    cpu=2,
    memory="4Gi",
    script=[
        f"git clone {REPO_URL} workdir",
        "cd workdir",
        f"git checkout {BRANCH}",
        "uv sync",
        "export PYTHONUNBUFFERED=1",
        # TODO: your MLflow server (or rely on the cluster-wide default)
        # "export MLFLOW_TRACKING_URI=http://mlflow-s3.rationai-mlflow",
        "export MLFLOW_USER=jiribuchta",  # TODO: your username
        *[
            f"uv run python -m pipeline pipeline={step} +experiment=provenance_pipeline"
            for step in steps
        ],
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
