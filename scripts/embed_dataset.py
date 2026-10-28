"""Submit the dataset-embedding step as one k8s job.

Edit the TODOs in scripts/_submit.py (repository URL, image), then:
    uv run python scripts/embed_dataset.py
"""

from _submit import submit_steps


if __name__ == "__main__":
    submit_steps(["embed_dataset"])
