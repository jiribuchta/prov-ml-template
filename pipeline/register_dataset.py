"""Step ``register_dataset`` — register existing files as a dataset.

The data itself is never copied or modified — it stays where it is on the
cluster; only ``manifest.csv`` (the file list) is written into ``data_path``.

``pipeline.source`` selects the input shape:

* a table (csv/xlsx) whose ``path_column`` lists the sample files
  (absolute, or relative to ``data_path``); the optional
  ``label_column`` holds integer labels (``0`` when absent) —
  e.g. a slide folder + an xlsx cohort table,
* ``null``: the data folder itself, one class subfolder per label,
  ``<data_path>/<class>/<file>``.
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit.lightning.loggers import MLFlowLogger
from rationai.provenance import log_provenance, register_dataset

from pipeline.common import (
    LABEL_COLUMN,
    PATH_COLUMN,
    data_root,
    read_source,
    suspended_run,
)


def run(config: DictConfig, logger: MLFlowLogger) -> None:
    del logger  # the registration run is managed by rationai-provenance
    step = config.pipeline
    root = data_root(config)

    if step.source:
        if not step.path_column:
            raise ValueError("pipeline.path_column is required with pipeline.source")
        table = read_source(str(step.source))
        paths = [Path(str(value).strip()) for value in table[str(step.path_column)]]
        paths = [p if p.is_absolute() else root / p for p in paths]
        labels = (
            table[str(step.label_column)].astype(int).tolist()
            if step.label_column
            else [0] * len(paths)
        )
    else:
        files = sorted(root.glob("*/*.csv"))
        if not files:
            raise FileNotFoundError(
                f"No sample files in '{root}' — expected the layout "
                "<data_path>/<class>/<file>, or set pipeline.source to a "
                "table with paths."
            )
        classes = sorted({f.parent.name for f in files})
        class_index = {name: i for i, name in enumerate(classes)}
        paths = files
        labels = [class_index[f.parent.name] for f in files]

    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {len(paths)} sample files not found, "
            f"e.g. {missing[:3]} — check data_path / pipeline.path_column."
        )
    manifest = pd.DataFrame(
        {
            PATH_COLUMN: [
                str(p.resolve().relative_to(root))
                if root in p.parents
                else str(p)  # outside data_path: keep the absolute path
                for p in paths
            ],
            LABEL_COLUMN: labels,
        }
    )
    manifest.to_csv(root / "manifest.csv", index=False)

    dataset = config.dataset
    log_provenance(config=config)
    with suspended_run():
        dataset_run_id = register_dataset(
            dataset_dir=str(root),
            dataset_name=str(dataset.name),
            version=str(dataset.version),
        )
    mlflow.set_tag("dataset_run_id", dataset_run_id)
    print(f"[register_dataset] {len(manifest)} samples → registered")
    print(f"[register_dataset] labels: {dict(manifest[LABEL_COLUMN].value_counts())}")
