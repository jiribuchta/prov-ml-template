# Provenance Pipeline Template

A machine learning template where **every step is provenance-tracked**: who ran
what, on which data, with which code and environment — all recorded in MLflow by
[rationai-provenance](https://github.com/RationAI/provenance) so a W3C PROV
document can be assembled from the runs later.

The demo pipeline: **create_user → register_dataset → split_dataset → embed_dataset
→ learn**. Each step lives in its own module in [`pipeline/`](pipeline) and has
its own k8s job script in [`scripts/`](scripts). The data is used **in place** —
files stay where they are on the cluster; nothing is copied or created:

| Step | Module / script | What it does | Provenance shown |
| --- | --- | --- | --- |
| `create_user` | `create_user` | registers you in the `User_Registry` (one-time) | identity params linked to all your runs |
| `register_dataset` | `register_dataset` | registers existing files (a table with paths, csv/xlsx, or the folder itself) as a dataset — in place | `register_dataset` → file sizes/mtimes in `Dataset_Registry` |
| `split_dataset` | `split_dataset` | verifies the data on disk against the registration, splits the **files** into train/test lists | `verify_dataset`, `dataset_verified` tag, `train.csv`/`test.csv` artifacts |
| `embed_dataset` | `embed_dataset` | PCA-reduces the train samples to 2-D embeddings | consumes the split run's artifact (`inputs`) |
| `learn` | `learn` | k-means + adjusted Rand index vs ground truth | consumes the embed run's artifact (`inputs`) |

Each step is one MLflow run, tagged `step=<name>`. Downstream steps find their
upstream via `dataset.mlflow_uris` in the config (pinned `runs:/...` URI), or
automatically use the latest finished run of that step.

## Data layout

Three input shapes for `register_dataset` (first match wins):

**1. A table with paths** (the usual case — e.g. a slide folder plus the
cohort csv/xlsx that usually ships with the data and describes it).
`data_path` points at the folder with the files, the table lists them:

```bash
uv run python -m pipeline pipeline=register_dataset +experiment=provenance_pipeline \
    data_path=/mnt/data/.../mammaprint \
    pipeline.source=/mnt/data/.../mammaprint/cohort.xlsx \
    pipeline.path_column=slide pipeline.label_column=label
```

Paths may be absolute or relative to `data_path` (missing files fail the
step); labels must be integers (`label_column: null` → all `0`, and split
won't stratify).

**2. A regex** (`pipeline.pattern`) — give the folder and what to find:

```bash
uv run python -m pipeline pipeline=register_dataset +experiment=provenance_pipeline \
    data_path=/mnt/data/.../slides \
    'pipeline.pattern=.*\.svs$'
```

The regex matches paths relative to `data_path` (recursively; `manifest.csv`
and hidden files are skipped). Classes come from a named `label` group when
present, otherwise from the parent folder (files directly in `data_path` get
label `0`). Regexes with Hydra special characters (`<`, `>`, `{`…) need an
inner layer of quotes:

```bash
"pipeline.pattern='^(?P<label>class_\d+)/.*\.csv$'"
```

**3. The folder itself** (`source`/`pattern`: null) — one subfolder per
class, any file type:

```
<data_path>/
├── class_0/
│   └── sample.csv
└── class_1/
    └── sample.csv
```

Filenames must be unique across the whole folder (the MLflow dataset
registry keys files by name).

`embed_dataset`/`learn` operate on per-sample feature files (one row of feature
values per file) — for image/WSI data, insert your own featurization step
there. No data at hand? Generate the demo layout (shape 1) and run the
whole pipeline:

```bash
uv run python scripts/make_demo_data.py --out data/demo
```

## Setup

```bash
uv sync                      # create the environment
export MLFLOW_TRACKING_URI=...   # your MLflow server
export MLFLOW_USER=...           # your username
```

Register yourself once (one `User_Registry` run, links all your runs to you):

```bash
uv run python -m pipeline pipeline=create_user +experiment=provenance_pipeline
```

The identity fields (real name, email, organization, lead) live in
[`configs/pipeline/create_user.yaml`](configs/pipeline/create_user.yaml).
The username defaults to your kube-config username (`JOB_USER` in-cluster) —
the exact name provenance looks runs up by, so cluster runs link to you
automatically (set it only if you have no kube config).

## Run the pipeline

Each command is one step; run them in order:

```bash
uv run python -m pipeline pipeline=register_dataset +experiment=provenance_pipeline data_path=/path/to/your/data
uv run python -m pipeline pipeline=split_dataset  +experiment=provenance_pipeline
uv run python -m pipeline pipeline=embed_dataset  +experiment=provenance_pipeline
uv run python -m pipeline pipeline=learn          +experiment=provenance_pipeline
```

Step parameters live in [`configs/pipeline/`](configs/pipeline) (samples, test
size, PCA dims, cluster count) and can be overridden on the command line:

```bash
uv run python -m pipeline pipeline=learn +experiment=provenance_pipeline pipeline.n_clusters=5
```

### Pinning upstream runs

By default, `embed_dataset`/`learn` consume the **latest** finished run of their
upstream step. To freeze the chain to specific runs, pin them in
[`configs/experiment/`](configs/experiment/)
— every step prints the exact line to paste:

```
[hint] to pin this run for downstream steps, set in the experiment config:
         dataset.mlflow_uris.split_dataset=runs:/<run_id>/train.csv
```

The dataset itself is defined by its `Dataset_Registry` run: `dataset.run_id`
pins a specific registration (printed by `register_dataset`); by default,
`split_dataset` verifies against the latest registration of
`dataset.name` at `dataset.version`.

## What to look for in the MLflow UI

- **Dataset registration** (`Dataset_Registry` experiment): a run per dataset
  version with `file_sizes` / `file_mtimes` tags — the fingerprint that
  `verify_dataset` checks your disk against.
- **Pipeline runs** (the experiment named in the config): one run per step.
  - *Tags*: `step`, `dataset_verified`, `dataset_run_id`, `user_run_id`, git
    commit, docker image, hardware, seeds — and `job_*` (cluster, CPU/memory
    limits, image) when running as a kube-job.
  - *Params*: the whole Hydra config (`cfg_*`, interpolations resolved) plus
    per-split sample counts.
  - *Artifacts*: `train.csv`/`test.csv` (the file lists), `embeddings/`,
    `cluster/` (learn assignments), the frozen requirements, the Hydra configs, and
    `console.log` (stdout capture).
  - `embed_dataset`/`learn` runs list their upstream under the `input_uris` tag.

## Running on the cluster

```bash
uv sync --extra job    # kube-jobs
# edit the TODOs in scripts/_submit.py (repo URL, image)
```

The cluster username is taken from your kube config — no need to configure it
(it becomes `JOB_USER` inside the job, so provenance links runs to you
automatically). One job script per step (`scripts/create_user.py`, `scripts/register_dataset.py`,
`scripts/split_dataset.py`, `scripts/embed_dataset.py`, `scripts/learn.py`), or
the whole chain in one job:

```bash
uv run python scripts/submit_pipeline.py
```

In-cluster runs get `MLFLOW_TRACKING_URI` from the cluster default and `job_*`
tags from the kube-jobs-injected `JOB_*` env vars.

## Development

```bash
uvx ruff check
uv run mypy .
```
