"""Step ``create_user`` — register the user in the provenance ``User_Registry``.

One-time per person: links every subsequent run to a real-world user.
"""

from __future__ import annotations

import mlflow
from omegaconf import DictConfig
from rationai.mlkit.lightning.loggers import MLFlowLogger
from rationai.provenance import log_provenance, register_new_user

from pipeline.common import suspended_run


def run(config: DictConfig, logger: MLFlowLogger) -> None:
    del logger  # the registration run is managed by register_new_user
    step = config.pipeline
    log_provenance(config=config)
    with suspended_run():
        run_id = register_new_user(
            username=str(step.username),
            real_name=str(step.real_name),
            email=str(step.email),
            organization=str(step.organization),
            lead_name=str(step.lead_name),
            lead_email=str(step.lead_email),
        )
    mlflow.set_tag("user_run_id", run_id)
    print(f"[create_user] registered '{step.username}' → User_Registry run {run_id}")
