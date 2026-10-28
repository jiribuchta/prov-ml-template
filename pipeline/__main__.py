"""End-to-end provenance pipeline.

Prerequisites (each run once by the person responsible): create_user →
register_dataset. The pipeline itself: split_dataset → embed_dataset → learn.

Every step runs as an MLflow run (mlkit ``@autolog``) and records its
provenance via [rationai-provenance](https://github.com/RationAI/provenance).
Each step lives in its own module next to this file:

* ``create_user``    — prerequisite, once per user: register yourself in the
  ``User_Registry``.
* ``register_dataset`` — prerequisite, once per dataset version (data
  manager): register the files already in ``data_path`` as a dataset
  (in place — nothing is copied or created; the provenance
  ``register_dataset`` call records file sizes/mtimes → ``Dataset_Registry``).
  Samples come from a source table (csv/xlsx), a ``pipeline.pattern`` regex
  on the folder, or one class subfolder per label.
* ``split_dataset``  — verify the data on disk against the registration
  (``verify_dataset``), then split the FILES into train/test lists
  (each sample file wholly on one side; only the lists are logged).
* ``embed_dataset``  — reduce the train samples to PCA embeddings; consumes
  the split run's artifact.
* ``learn``          — k-means on the embeddings + adjusted Rand index vs
  the ground truth; consumes the embed run's artifact.

Downstream steps locate their upstream via ``dataset.mlflow_uris.<step>``
in the Hydra config — required, the ``[hint]`` line each step prints;
no implicit latest-run resolution.

Run with::

    uv run python -m pipeline pipeline=create_user     +experiment=provenance_pipeline
    uv run python -m pipeline pipeline=register_dataset  +experiment=provenance_pipeline
    uv run python -m pipeline pipeline=split_dataset   +experiment=provenance_pipeline
    uv run python -m pipeline pipeline=embed_dataset   +experiment=provenance_pipeline
    uv run python -m pipeline pipeline=learn           +experiment=provenance_pipeline
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import hydra
import mlflow
from omegaconf import DictConfig
from rationai.mlkit import autolog

from pipeline import create_user, embed_dataset, learn, register_dataset, split_dataset


if TYPE_CHECKING:
    from collections.abc import Callable

    from rationai.mlkit.lightning.loggers import MLFlowLogger

_STEPS: dict[str, Callable[[DictConfig, MLFlowLogger], None]] = {
    "create_user": create_user.run,
    "register_dataset": register_dataset.run,
    "split_dataset": split_dataset.run,
    "embed_dataset": embed_dataset.run,
    "learn": learn.run,
}


@hydra.main(config_path="../configs", config_name="pipeline", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    mlflow.set_tag("step", str(config.pipeline.step))
    _STEPS[str(config.pipeline.step)](config, logger)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
