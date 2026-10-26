# Provenance Pipeline Template

A machine learning template where **every step is provenance-tracked**: who ran
what, on which data, with which code and environment — all recorded in MLflow by
[rationai-provenance](https://github.com/RationAI/provenance) so a W3C PROV
document can be assembled from the runs later.

The demo pipeline: **create → split → embed → cluster**

| Step | What it does | Provenance shown |
| --- | --- | --- |
| `create` | generates a toy dataset (gaussian blobs) + `manifest.csv` | `register_dataset` → file sizes/mtimes in `Dataset_Registry` |
| `split` | verifies the data on disk against the registration, splits train/test | `verify_dataset`, `dataset_verified` tag, split CSVs as artifacts |
| `embed` | PCA-reduces the train split to 2-D embeddings | consumes the split run's artifact (`inputs`) |
| `cluster` | k-means + adjusted Rand index vs ground truth | consumes the embed run's artifact (`inputs`) |

Each step is one MLflow run, tagged `step=<name>`. Downstream steps find their
upstream via `dataset.mlflow_uris` in the config (pinned `runs:/...` URI), or
automatically use the latest finished run of that step.

## Setup

```bash
uv sync                      # create the environment
export MLFLOW_TRACKING_URI=...   # your MLflow server
export MLFLOW_USER=...           # your username
```

Register yourself once (one `User_Registry` run, links all your runs to you):

```bash
uv run python -c "from rationai.provenance import register_new_user; \
register_new_user(username='jdoe', real_name='Jane Doe', email='jdoe@example.org', \
organization='RationAI', lead_name='Prof. Lead', lead_email='lead@example.org')"
```

## Run the pipeline

Each command is one step; run them in order:

```bash
uv run python -m pipeline pipeline=create  +experiment=provenance_pipeline
uv run python -m pipeline pipeline=split   +experiment=provenance_pipeline
uv run python -m pipeline pipeline=embed   +experiment=provenance_pipeline
uv run python -m pipeline pipeline=cluster +experiment=provenance_pipeline
```

Step parameters live in [`configs/pipeline/`](configs/pipeline) (samples, test
size, PCA dims, cluster count) and can be overridden on the command line:

```bash
uv run python -m pipeline pipeline=cluster +experiment=provenance_pipeline pipeline.n_clusters=5
```

### Pinning upstream runs

By default, `embed`/`cluster` consume the **latest** finished run of their
upstream step. To freeze the chain to specific runs, pin them in
[`configs/experiment/provenance_pipeline.yaml`](configs/experiment/provenance_pipeline.yaml)
— every step prints the exact line to paste:

```
[hint] to pin this run for downstream steps, set in the experiment config:
         dataset.mlflow_uris.split=runs:/<run_id>/train.csv
```

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
  - *Artifacts*: `train.csv`/`test.csv`, `embeddings/`, `cluster/`, the frozen
    requirements, the Hydra configs, and `console.log` (stdout capture).
  - `embed`/`cluster` runs list their upstream under the `input_uris` tag.

## Running on the cluster

```bash
uv sync --extra job    # kube-jobs (feature/job-env-metadata branch)
# edit the TODOs in scripts/submit_pipeline.py (repo URL, username, image)
uv run python scripts/submit_pipeline.py
```

One job runs all four steps; in-cluster runs get `MLFLOW_TRACKING_URI` from the
cluster default and `job_*` tags from the kube-jobs-injected `JOB_*` env vars.

## Development

```bash
uvx ruff check
uv run mypy .
```
