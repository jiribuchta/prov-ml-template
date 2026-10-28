"""Step ``register_dataset`` — register existing files as a dataset.

The data itself is never copied or modified — it stays where it is on the
cluster; only ``manifest.csv`` (the file list) is written into ``data_path``.

``pipeline.source`` / ``pipeline.pattern`` select the input shape (first
match wins):

* ``source``: a table (csv/xlsx) whose ``path_column`` lists the sample
  files (absolute, or relative to ``data_path``); the optional
  ``label_column`` holds integer labels (``0`` when absent) — e.g. a slide
  folder + the cohort xlsx that usually ships with the data,
* ``pattern``: a regex on paths relative to ``data_path``; a named
  ``label`` group assigns classes, otherwise the parent folder is the class
  (files directly in ``data_path`` get label ``0``),
* both ``null``: the data folder itself, one class subfolder per label,
  ``<data_path>/<class>/<file>`` (any file type).
"""

from __future__ import annotations

import re
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


def _from_folder(root: Path, pattern: str | None) -> tuple[list[Path], list[int]]:
    """Collect samples from the folder itself (no source table)."""
    files: list[Path]
    label_strs: list[str]
    if pattern is None:
        # one class subfolder per label, any file type
        files = sorted(p for p in root.glob("*/*") if p.is_file())
        label_strs = [f.parent.name for f in files]
    else:
        regex = re.compile(pattern)
        files, label_strs = [], []
        for f in sorted(root.rglob("*")):
            rel = f.relative_to(root)
            if (
                not f.is_file()
                or f.name == "manifest.csv"  # ours, written into data_path
                or any(part.startswith(".") for part in rel.parts)
            ):
                continue
            m = regex.search(rel.as_posix())
            if m is None:
                continue
            files.append(f)
            if "label" in regex.groupindex:
                label_strs.append(str(m.group("label")))
            elif f.parent != root:
                label_strs.append(f.parent.name)
            else:  # directly in data_path, no class folder
                label_strs.append("0")
    classes = sorted(set(label_strs))
    class_index = {name: i for i, name in enumerate(classes)}
    return files, [class_index[s] for s in label_strs]


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
        paths, labels = _from_folder(
            root, str(step.pattern) if step.pattern else None
        )
        if not paths:
            raise FileNotFoundError(
                f"No sample files in '{root}' — expected the layout "
                "<data_path>/<class>/<file>, or set pipeline.source / "
                "pipeline.pattern."
            )

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
    print(
        "[hint] to pin this dataset for downstream steps, set in the "
        f"experiment config:\n         dataset.run_id={dataset_run_id}"
    )
