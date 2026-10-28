"""Step ``embed_dataset`` — reduce the train samples to PCA embeddings.

Reads the sample feature files from ``data_path`` (one feature row per
file — for WSI data, replace this step with your featurization); consumes
the split run's artifact.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit.lightning.loggers import MLFlowLogger
from rationai.provenance import log_provenance
from sklearn.decomposition import PCA

from pipeline.common import (
    LABEL_COLUMN,
    PATH_COLUMN,
    data_root,
    download,
    pin_hint,
    resolve_upstream,
)


def run(config: DictConfig, logger: MLFlowLogger) -> None:
    step = config.pipeline
    split_uri = resolve_upstream(config, "split_dataset", "split_dataset", "train.csv")
    train = pd.read_csv(download(split_uri))

    root = data_root(config)
    features = np.vstack(
        [np.loadtxt(root / rel, delimiter=",") for rel in train[PATH_COLUMN]]
    )
    embeddings = pd.DataFrame(
        PCA(
            n_components=int(step.n_components), random_state=int(step.seed)
        ).fit_transform(features),
        columns=[f"pc{i + 1}" for i in range(int(step.n_components))],
    )
    embeddings.insert(0, PATH_COLUMN, train[PATH_COLUMN].to_numpy())
    embeddings.insert(1, LABEL_COLUMN, train[LABEL_COLUMN].to_numpy())

    with TemporaryDirectory() as tmp:
        out = Path(tmp) / "embeddings"
        out.mkdir()
        embeddings.to_csv(out / "train.csv", index=False)
        logger.log_artifacts(tmp)

    log_provenance(
        config=config,
        output={"embeddings": embeddings},
        inputs=[split_uri],
        label_column=LABEL_COLUMN,
    )
    print(f"[embed_dataset] {len(embeddings)} samples → {int(step.n_components)} dims")
    run = mlflow.active_run()
    assert run is not None  # @autolog guarantees an active run
    pin_hint("embed_dataset", run.info.run_id, "embeddings/train.csv")
