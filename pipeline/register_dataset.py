"""Step ``register_dataset`` — register existing files as a dataset.

The data itself is never copied or modified — it stays where it is on the
cluster; only ``manifest.csv`` (the file list) is written into ``data_path``.

``pipeline.source`` is the one csv listing the sample files — e.g. the
cohort table that ships with the data. A column named ``path`` / ``slide``
/ ``file`` / ``wsi_path`` (or the table's only column) holds the paths —
absolute, or relative to ``data_path`` — and an optional ``label`` /
``cancer`` column holds integer labels (``0`` when absent).
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

PATH_COLUMNS = ("path", "slide", "file", "wsi_path", "slide_path")
LABEL_COLUMNS = ("label", "cancer")


def _pick_column(table: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """Column whose (stripped, lowercased) name is in *candidates*."""
    columns = {str(c).strip().lower(): c for c in table.columns}
    return next((columns[c] for c in candidates if c in columns), None)


def run(config: DictConfig, logger: MLFlowLogger) -> None:
    del logger  # the registration run is managed by rationai-provenance
    root = data_root(config)
    step = config.pipeline

    if not step.source:
        raise ValueError("pipeline.source — the csv with the slide paths — is required")

    source = Path(str(step.source))
    if not source.is_absolute():
        source = root / source
    table = read_source(str(source))

    path_col = _pick_column(table, PATH_COLUMNS) or (
        table.columns[0] if len(table.columns) == 1 else None
    )
    if path_col is None:
        raise ValueError(
            f"'{source}' has no path column — name one of {PATH_COLUMNS} "
            f"(or make it the only column); found {list(table.columns)}"
        )
    label_col = _pick_column(table, LABEL_COLUMNS)

    paths = [Path(str(value).strip()) for value in table[path_col]]
    paths = [p if p.is_absolute() else root / p for p in paths]
    labels = (
        table[label_col].astype(int).tolist()
        if label_col
        else [0] * len(paths)
    )
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {len(paths)} sample files not found, "
            f"e.g. {missing[:3]} — check data_path / pipeline.source."
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
    print(
        "[hint] to pin this dataset for downstream steps, set in the "
        f"experiment config:\n         dataset.run_id={dataset_run_id}"
    )
