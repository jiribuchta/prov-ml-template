"""End-to-end provenance pipeline: create → split → embed → cluster.

Every step runs as an MLflow run (mlkit ``@autolog``) and records its
provenance via [rationai-provenance](https://github.com/RationAI/provenance):

* **create**  — build a manifest from the sample files in ``data_path``
  (``<data_path>/<class>/<sample>.csv``) and register it
  (``register_dataset``: file sizes/mtimes → ``Dataset_Registry``).
* **split**   — verify the data on disk against the registration
  (``verify_dataset``), then split the FILES into train/test folders
  (copied into the run's ``split/`` artifact, class layout preserved).
* **embed**   — reduce the split run's train FILES to PCA embeddings;
  consumes the split run's artifact.
* **cluster** — k-means on the embeddings + adjusted Rand index vs the
  ground truth; consumes the embed run's artifact.

Downstream steps locate their upstream via ``dataset.mlflow_uris.<step>``
in the Hydra config — pinned to a specific run, or ``null`` to automatically
use the latest finished run of that step (tag ``step``).

Run with::

    uv run python -m pipeline pipeline=create +experiment=provenance_pipeline
    uv run python -m pipeline pipeline=split  +experiment=provenance_pipeline
    uv run python -m pipeline pipeline=embed  +experiment=provenance_pipeline
    uv run python -m pipeline pipeline=cluster +experiment=provenance_pipeline
"""

from __future__ import annotations

from pathlib import Path
from shutil import copy2
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

import hydra
import mlflow
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger
from rationai.provenance import log_provenance, register_dataset, verify_dataset
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split


if TYPE_CHECKING:
    from collections.abc import Callable


# Column names required by the provenance library's manifest handling.
PATH_COLUMN = "wsi_path"
LABEL_COLUMN = "cancer"


def _data_root(config: DictConfig) -> Path:
    return Path(str(config.data_path)).resolve()


def _resolve_upstream(config: DictConfig, key: str, step: str, artifact: str) -> str:
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


def _download(artifact_uri: str) -> str:
    """Download an mlflow artifact URI (runs:/...) to a local path."""
    return str(mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri))


def _pin_hint(key: str, run_id: str, artifact: str) -> None:
    print(
        f"[hint] to pin this run for downstream steps, set in the experiment config:\n"
        f"         dataset.mlflow_uris.{key}=runs:/{run_id}/{artifact}"
    )


# ── Steps ───────────────────────────────────────────────────────────────


def _create(config: DictConfig, logger: MLFlowLogger) -> None:
    """Build a manifest from a data folder and register the dataset.

    Expected layout (one feature vector per CSV file, one class per
    subfolder — sorted subfolder names map to the integer labels the
    provenance manifest requires)::

        <data_path>/<class>/<sample>.csv
    """
    del logger  # the registration run is managed by register_dataset

    data_root = _data_root(config)
    files = sorted(data_root.glob("*/*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No sample files in '{data_root}' — expected the layout "
            "<data_path>/<class>/<sample>.csv (one feature row per file)."
        )
    classes = sorted({f.parent.name for f in files})
    class_index = {name: i for i, name in enumerate(classes)}

    manifest = pd.DataFrame(
        {
            PATH_COLUMN: [str(f.relative_to(data_root)) for f in files],
            LABEL_COLUMN: [class_index[f.parent.name] for f in files],
        }
    )
    manifest.to_csv(data_root / "manifest.csv", index=False)

    dataset = config.dataset
    log_provenance(config=config)
    register_dataset(
        dataset_dir=str(data_root),
        dataset_name=str(dataset.name),
        version=str(dataset.version),
    )
    print(f"[create] {len(files)} samples in {len(classes)} classes → registered")
    print(f"[create] classes: {class_index}")


def _split(config: DictConfig, logger: MLFlowLogger) -> None:
    """Verify the dataset, then split the FILES into train/test folders.

    Copies each sample file into ``split/train/`` or ``split/test/``
    (keeping the class-subfolder layout) and writes the corresponding
    ``train.csv``/``test.csv`` manifests — all logged as run artifacts, so
    downstream steps no longer need the original data folder.
    """
    step = config.pipeline
    data_root = _data_root(config)
    manifest_path = data_root / "manifest.csv"

    verification: dict[str, Any] = verify_dataset(
        manifest_path=str(manifest_path), data_root=str(data_root)
    )
    mlflow.set_tags(
        {
            "dataset_verified": str(verification["verified"]),
            "dataset_run_id": str(verification.get("dataset_run_id")),
        }
    )

    manifest = pd.read_csv(manifest_path)
    train, test = train_test_split(
        manifest,
        test_size=float(step.test_size),
        stratify=manifest[LABEL_COLUMN],
        random_state=int(step.seed),
    )
    with TemporaryDirectory() as tmp:
        for name, frame in [("train", train), ("test", test)]:
            frame.to_csv(Path(tmp) / f"{name}.csv", index=False)
            for rel in frame[PATH_COLUMN]:
                destination = Path(tmp) / name / rel
                destination.parent.mkdir(parents=True, exist_ok=True)
                copy2(data_root / rel, destination)
        logger.log_artifacts(tmp, artifact_path="split")

    log_provenance(
        config=config,
        output={"train": train, "test": test},
        path_column=PATH_COLUMN,
        label_column=LABEL_COLUMN,
    )
    run = mlflow.active_run()
    assert run is not None  # @autolog guarantees an active run
    print(f"[split] dataset_verified={verification['verified']}")
    print(f"[split] {len(train)} train / {len(test)} test files → artifacts/split/")
    _pin_hint("split", run.info.run_id, "split")


def _embed(config: DictConfig, logger: MLFlowLogger) -> None:
    """Load the split run's train files and reduce them to PCA embeddings."""
    step = config.pipeline
    split_uri = _resolve_upstream(config, "split", "split", "split")
    split_dir = Path(_download(split_uri))
    train = pd.read_csv(split_dir / "train.csv")

    features = np.vstack(
        [
            np.loadtxt(split_dir / "train" / rel, delimiter=",")
            for rel in train[PATH_COLUMN]
        ]
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
    print(f"[embed] {len(embeddings)} samples → {int(step.n_components)} dims")
    run = mlflow.active_run()
    assert run is not None  # @autolog guarantees an active run
    _pin_hint("embed", run.info.run_id, "embeddings/train.csv")


def _cluster(config: DictConfig, logger: MLFlowLogger) -> None:
    """K-means the embeddings and score against ground-truth labels."""
    step = config.pipeline
    embed_uri = _resolve_upstream(config, "embed", "embed", "embeddings/train.csv")
    embeddings = pd.read_csv(_download(embed_uri))
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
    print(f"[cluster] ARI vs ground truth = {ari:.3f}")


_STEPS: dict[str, Callable[[DictConfig, MLFlowLogger], None]] = {
    "create": _create,
    "split": _split,
    "embed": _embed,
    "cluster": _cluster,
}


@hydra.main(config_path="../configs", config_name="pipeline", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    mlflow.set_tag("step", str(config.pipeline.step))
    _STEPS[str(config.pipeline.step)](config, logger)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
