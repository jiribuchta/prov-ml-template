"""Step ``create_user`` — register the user in the provenance ``User_Registry``.

One-time per person: links every subsequent run to a real-world user.
"""

from __future__ import annotations

import os

import mlflow
from omegaconf import DictConfig
from rationai.mlkit.lightning.loggers import MLFlowLogger
from rationai.provenance import log_provenance, register_new_user

from pipeline.common import kube_username, suspended_run


def run(config: DictConfig, logger: MLFlowLogger) -> None:
    del logger  # the registration run is managed by register_new_user
    step = config.pipeline
    username = str(step.username) if step.username else kube_username()
    if (mlflow_user := os.environ.get("MLFLOW_USER")) and mlflow_user != username:
        print(  # provenance looks runs up by JOB_USER / MLFLOW_USER
            f"[create_user] WARNING: MLFLOW_USER={mlflow_user} differs from the "
            f"registry username '{username}' — export MLFLOW_USER={username} "
            "so local runs link to you"
        )
    log_provenance(config=config)
    with suspended_run():
        run_id = register_new_user(
            username=username,
            real_name=str(step.real_name),
            email=str(step.email),
            organization=str(step.organization),
            lead_name=str(step.lead_name),
            lead_email=str(step.lead_email),
        )
    mlflow.set_tag("user_run_id", run_id)
    print(f"[create_user] registered '{username}' → User_Registry run {run_id}")
