"""Write a demo dataset in the layout the pipeline expects.

Feature files (one row of feature values per file) grouped in class
subfolders, plus ``samples.csv`` — a source table with paths and labels, the
shape real data usually comes in (e.g. a slide folder + a cohort table)::

    <out>/class_0/class_0_0000.csv
    <out>/class_1/class_1_0001.csv
    ...
    <out>/samples.csv

Try the full pipeline without real data::

    uv run python scripts/make_demo_data.py --out data/demo
    uv run python -m pipeline pipeline=create +experiment=provenance_pipeline \
        pipeline.source=data/demo/samples.csv pipeline.path_column=file \
        pipeline.label_column=category
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
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
    records = []
    for i, (vector, label) in enumerate(zip(features, labels, strict=True)):
        # unique global basenames: the registry keys files by name
        rel = f"class_{label}/class_{label}_{i:04d}.csv"
        destination = out / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(destination, vector.reshape(1, -1), delimiter=",")
        records.append({"file": rel, "category": int(label)})
    # source table on purpose uses arbitrary column names — the create step
    # maps them via pipeline.path_column / pipeline.label_column
    pd.DataFrame(records).to_csv(out / "samples.csv", index=False)
    print(f"[demo data] {len(records)} samples in {args.classes} classes → {out}/")


if __name__ == "__main__":
    main()
