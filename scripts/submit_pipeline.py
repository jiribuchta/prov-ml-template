"""Submit the whole provenance pipeline as one k8s job.

Requires the `job` extra (kube-jobs):
    uv sync --extra job

Edit the TODOs in scripts/_submit.py (repository URL, image), then:
    uv run python scripts/submit_pipeline.py

The job runs all steps sequentially; every run in MLflow gets the
job_* tags (cluster, image, CPU/memory limits, ...) injected by kube-jobs.
Register yourself once first: ``uv run python scripts/create_user.py``.
"""

from _submit import submit_steps


STEPS = ["register_dataset", "split_dataset", "embed_dataset", "learn"]

if __name__ == "__main__":
    submit_steps(STEPS)
