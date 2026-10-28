"""Step ``learn`` — k-means the embeddings, score against ground truth.

Consumes the embed run's artifact and logs the adjusted Rand index.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit.lightning.loggers import MLFlowLogger
from rationai.provenance import log_provenance
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from pipeline.common import (
    LABEL_COLUMN,
    PATH_COLUMN,
    download,
    resolve_upstream,
)


def run(config: DictConfig, logger: MLFlowLogger) -> None:
    step = config.pipeline
    embed_uri = resolve_upstream(config, "embed_dataset")
    embeddings = pd.read_csv(download(embed_uri))
    feature_cols = [c for c in embeddings.columns if c.startswith("pc")]

    labels = KMeans(
        n_clusters=int(step.n_clusters), random_state=int(step.seed), n_init="auto"
    ).fit_predict(embeddings[feature_cols].to_numpy())
    ari = adjusted_rand_score(embeddings[LABEL_COLUMN].to_numpy(), labels)

    assignments = embeddings[[PATH_COLUMN, LABEL_COLUMN]].assign(cluster=labels)
    with TemporaryDirectory() as tmp:
        out = Path(tmp) / "cluster"
        out.mkdir()
        assignments.to_csv(out / "assignments.csv", index=False)
        logger.log_artifacts(tmp)

    log_provenance(
        config=config,
        output={"assignments": assignments},
        inputs=[embed_uri],
        label_column=LABEL_COLUMN,
    )
    mlflow.log_metric("adjusted_rand_index", float(ari))
    print(f"[learn] ARI vs ground truth = {ari:.3f}")
