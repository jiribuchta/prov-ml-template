"""Step ``split_dataset`` — verify the dataset, then split the FILES.

Each sample file goes wholly into one side (no file is split across train
and test). Only the lists become run artifacts — the data itself stays on
disk in ``data_path`` (too big to copy for real datasets).
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import mlflow
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit.lightning.loggers import MLFlowLogger
from rationai.provenance import log_provenance
from rationai.provenance.dataset import verify_manifest
from sklearn.model_selection import train_test_split

from pipeline.common import (
    LABEL_COLUMN,
    PATH_COLUMN,
    data_root,
    dataset_run_id,
    pin_hint,
)


def run(config: DictConfig, logger: MLFlowLogger) -> None:
    step = config.pipeline
    root = data_root(config)
    manifest_path = root / "manifest.csv"

    verification: dict[str, Any] = verify_manifest(
        manifest_path=str(manifest_path),
        data_root=str(root),
        dataset_run_id=dataset_run_id(config),  # pinned or by name/version
    )
    mlflow.set_tags(
        {
            "dataset_verified": str(verification["verified"]),
            "dataset_run_id": str(verification.get("dataset_run_id")),
        }
    )

    manifest = pd.read_csv(manifest_path)
    stratify = manifest[LABEL_COLUMN] if manifest[LABEL_COLUMN].nunique() > 1 else None
    train, test = train_test_split(
        manifest,
        test_size=float(step.test_size),
        stratify=stratify,
        random_state=int(step.seed),
    )
    with TemporaryDirectory() as tmp:
        train.to_csv(Path(tmp) / "train.csv", index=False)
        test.to_csv(Path(tmp) / "test.csv", index=False)
        logger.log_artifacts(tmp)

    log_provenance(
        config=config,
        output={"train": train, "test": test},
        path_column=PATH_COLUMN,
        label_column=LABEL_COLUMN,
    )
    run = mlflow.active_run()
    assert run is not None  # @autolog guarantees an active run
    print(f"[split_dataset] dataset_verified={verification['verified']}")
    print(f"[split_dataset] {len(train)} train / {len(test)} test files")
    pin_hint("split_dataset", run.info.run_id, "train.csv")
