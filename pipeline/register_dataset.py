"""Step ``register_dataset`` — register existing files as a dataset.

The data itself is never copied or modified — it stays where it is on the
cluster. ``pipeline.source`` is the cohort csv (usually the one that ships
with the data): the column with the sample paths is auto-detected
(``wsi_path`` / ``path`` / ``slide`` / ``slide_path`` / ``file``).
Labels are not registered — they stay in the cohort csv itself.
The registration — reading the csv, writing the paths-only
``manifest.csv`` next to it, fingerprinting the files — is
done by ``rationai-provenance``; this step only adds the config plumbing.
"""

from __future__ import annotations

from pathlib import Path

import mlflow
from omegaconf import DictConfig
from rationai.mlkit.lightning.loggers import MLFlowLogger
from rationai.provenance import log_provenance, register_dataset

from pipeline.common import data_root, suspended_run


def run(config: DictConfig, logger: MLFlowLogger) -> None:
    del logger  # the registration run is managed by rationai-provenance
    step = config.pipeline

    if not step.source:
        raise ValueError("pipeline.source — the csv with the slide paths — is required")

    source = Path(str(step.source))
    if not source.is_absolute():
        source = data_root(config) / source

    dataset = config.dataset
    log_provenance(config=config)
    with suspended_run():
        dataset_run_id = register_dataset(
            csv=str(source),
            dataset_name=str(dataset.name),
            version=str(dataset.version),
            # no dedicated registry — register into the project's experiment
            experiment_name=str(config.metadata.experiment_name),
        )
    mlflow.set_tag("dataset_run_id", dataset_run_id)
    print(f"[register_dataset] registered {source.name} → run in '{config.metadata.experiment_name}'")
    print(
        "[hint] to pin this dataset for downstream steps, set in the "
        f"experiment config:\n         dataset.run_id={dataset_run_id}"
    )
