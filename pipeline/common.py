"""Shared helpers for the pipeline steps (see ``__main__.py``)."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import mlflow
import pandas as pd
from omegaconf import DictConfig, OmegaConf


if TYPE_CHECKING:
    from collections.abc import Iterator


# Column names required by the provenance library's manifest handling.
PATH_COLUMN = "wsi_path"
LABEL_COLUMN = "cancer"


def kube_username() -> str:
    """The username kube-jobs puts in ``JOB_USER``.

    In-cluster: the env var. Locally: the same kube-config SelfSubjectReview
    normalization. rationai-provenance looks runs up in ``User_Registry`` by
    this name, so ``create_user`` must register exactly this value.
    """
    if user := os.environ.get("JOB_USER"):
        return user
    try:  # local run: same resolution kube-jobs uses to set JOB_USER
        from kube_jobs.constants import CLUSTER
        from kube_jobs.resolve_kube_config import load_kube_config
        from kubernetes import client

        load_kube_config(context=CLUSTER)
        review = client.AuthenticationV1Api().create_self_subject_review(
            body=client.V1SelfSubjectReview()
        )
        assert review.status and review.status.user_info and review.status.user_info.extra
        raw = str(review.status.user_info.extra["username"][0]).split("@")[0]
        return raw.lower().replace(" ", "-").replace("_", "-")
    except Exception as error:
        if user := os.environ.get("MLFLOW_USER"):
            print(f"[kube_username] kube config unavailable ({error}); using MLFLOW_USER")
            return user
        raise RuntimeError(
            "cannot determine the username: no JOB_USER, no kube config, "
            "no MLFLOW_USER — set pipeline.username"
        ) from error


def data_root(config: DictConfig) -> Path:
    return Path(str(config.data_path)).resolve()


def resolve_upstream(config: DictConfig, key: str, step: str, artifact: str) -> str:
    """Return the upstream artifact URI from the config or the latest run.

    ``dataset.mlflow_uris.<key>`` pins a specific run; when null, the latest
    finished run tagged ``step=<step>`` in the experiment is used.
    """
    pinned = OmegaConf.select(config, f"dataset.mlflow_uris.{key}")
    if pinned:
        return str(pinned)

    experiment = mlflow.get_experiment_by_name(str(config.metadata.experiment_name))
    if experiment is None:
        raise RuntimeError(
            f"MLflow experiment '{config.metadata.experiment_name}' not found — "
            "run the upstream step first."
        )
    runs = mlflow.search_runs(
        [experiment.experiment_id],
        filter_string=f"tags.step = '{step}' and attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise RuntimeError(
            f"No finished '{step}' run found — run it first, or pin "
            f"dataset.mlflow_uris.{key} in the experiment config."
        )
    run_id = str(runs.iloc[0]["run_id"])
    return f"runs:/{run_id}/{artifact}"


def download(artifact_uri: str) -> str:
    """Download an mlflow artifact URI (runs:/...) to a local path."""
    return str(mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri))


def pin_hint(key: str, run_id: str, artifact: str) -> None:
    print(
        f"[hint] to pin this run for downstream steps, set in the experiment config:\n"
        f"         dataset.mlflow_uris.{key}=runs:/{run_id}/{artifact}"
    )


def read_source(source: str) -> pd.DataFrame:
    """Read the source table with sample paths (csv or xlsx)."""
    if source.endswith(".csv"):
        return pd.read_csv(source)
    return pd.read_excel(source)  # openpyxl handles .xlsx


@contextmanager
def suspended_run() -> Iterator[None]:
    """Pause the step's active MLflow run.

    Needed while a provenance registration (``register_new_user`` /
    ``register_dataset``) starts its own run — MLflow refuses to start a
    run while another is active.
    """
    active = mlflow.active_run()
    if active is None:
        yield
        return
    mlflow.end_run()
    try:
        yield
    finally:
        # the library call may have switched the active experiment
        # (set_experiment); restore it before resuming the step run
        mlflow.set_experiment(experiment_id=active.info.experiment_id)
        mlflow.start_run(active.info.run_id)  # ended by mlflow's atexit
