"""Write a demo dataset in the layout the pipeline expects.

Each file is one sample (a single row of feature values); each class lives in
its own subfolder, and basenames are unique across the whole dataset (the
MLflow dataset registry keys files by name)::

    <out>/class_0/class_0_0000.csv
    <out>/class_1/class_1_0001.csv
    ...

Try the full pipeline without real data::

    uv run python scripts/make_demo_data.py --out data/demo
    uv run python -m pipeline pipeline=create +experiment=provenance_pipeline
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.datasets import make_blobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/demo", help="output folder")
    parser.add_argument("--classes", type=int, default=3)
    parser.add_argument("--samples-per-class", type=int, default=100)
    parser.add_argument("--features", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    features, labels = make_blobs(
        n_samples=args.classes * args.samples_per_class,
        n_features=args.features,
        centers=args.classes,
        random_state=args.seed,
    )
    out = Path(args.out)
    for i, (vector, label) in enumerate(zip(features, labels, strict=True)):
        # unique global basenames: the registry keys files by name
        name = f"class_{label}/class_{label}_{i:04d}.csv"
        destination = out / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(destination, vector.reshape(1, -1), delimiter=",")
    print(f"[demo data] {len(features)} samples in {args.classes} classes → {out}/")


if __name__ == "__main__":
    main()
