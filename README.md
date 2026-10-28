# Provenance Pipeline Template

A machine learning template where **every step is provenance-tracked**: who ran
what, on which data, with which code and environment — all recorded in MLflow by
[rationai-provenance](https://github.com/RationAI/provenance) so a W3C PROV
document can be assembled from the runs later.

The demo pipeline: **create_user → register_dataset → split_dataset → embed_dataset
→ learn**. Each step lives in its own module in [`pipeline/`](pipeline) and has
its own k8s job script in [`scripts/`](scripts). The data is used **in place** —
files stay where they are on the cluster; nothing is copied or created:

| Prerequisite · who | Module / script | What it does | Provenance shown |
| --- | --- | --- | --- |
| `create_user` · you, once | `create_user` | registers you in the `User_Registry` (one-time) | identity params linked to all your runs |
| `register_dataset` · data manager, once per dataset | `register_dataset` | registers the files listed in one csv with the slide paths as a dataset — in place | `register_dataset` → file sizes/mtimes in the project's experiment |
| **Pipeline step** | | | |
| `split_dataset` · you | `split_dataset` | verifies the data on disk against the registration, splits the **files** into train/test lists | `verify_dataset`, `dataset_verified` tag, `train.csv`/`test.csv` artifacts |
| `embed_dataset` · you | `embed_dataset` | PCA-reduces the train samples to 2-D embeddings | consumes the split run's artifact (`inputs`) |
| `learn` · you | `learn` | k-means + adjusted Rand index vs ground truth | consumes the embed run's artifact (`inputs`) |

Each step is one MLflow run, tagged `step=<name>`. Downstream steps consume
the run pinned in `dataset.mlflow_uris` in the config — every step prints
the `[hint]` line to paste; nothing resolves implicitly.

## Data layout

`register_dataset` takes **one csv with the slide paths** — e.g. a slide
folder plus the cohort csv that usually ships with the data and describes
it. `data_path` points at the folder with the files, the csv lists them
(absolute, or relative to `data_path`):

```bash
uv run python -m pipeline pipeline=register_dataset \
    metadata.experiment_name=MyProject data_path=/mnt/data/.../mammaprint \
    pipeline.source=slides.csv dataset.name=mammaprint
```

The data itself is never copied or modified — it is registered in place,
only `manifest.csv` (the file list, paths only) is written into `data_path`.

Labels are not part of registration — they stay in the cohort csv itself,
in whatever column/format it has. `split_dataset` takes them from there
(a `cancer` / `label` column, matched by row order) for stratification.

`embed_dataset`/`learn` operate on per-sample feature files (one row of feature
values per file) — for image/WSI data, insert your own featurization step
there. No data at hand? Generate the demo layout and run the
whole pipeline:

```bash
uv run python scripts/make_demo_data.py --out data/demo
```

## Prerequisites

Two steps happen **before** any analysis, each run once by the person
responsible for it:

### 1. Register yourself — user, once ever

```bash
uv sync                      # create the environment
export MLFLOW_TRACKING_URI=...   # your MLflow server
export MLFLOW_USER=...           # your username

uv run python -m pipeline pipeline=create_user +experiment=provenance_pipeline
```

**Needs:** `uv` + the MLflow server address. **Produces:** one `User_Registry`
run — every run you ever start is linked to you.

The identity fields (real name, email, organization, lead) live in
[`configs/pipeline/create_user.yaml`](configs/pipeline/create_user.yaml).
The username defaults to your kube-config username (`JOB_USER` in-cluster) —
the exact name provenance looks runs up by, so cluster runs link to you
automatically (set it only if you have no kube config).

### 2. Register the dataset — data manager, once per dataset version

Expected of whoever owns the data (for us, Adam Kukučka) — no analysis can
run before it. No experiment YAML is needed — the values go on the command
line:

**Needs:** the data on the cluster + three values (`data_path`,
`pipeline.source` — the csv with the slide paths — and `dataset.name`).
**Produces:** `manifest.csv` in `data_path`, a registration run in the
project's experiment (`metadata.experiment_name`) with the dataset
fingerprint (file sizes/mtimes), and the `run_id` to hand to the users:

```bash
uv run python -m pipeline pipeline=register_dataset \
    metadata.experiment_name=<project> data_path=/path/to/your/data \
    pipeline.source=slides.csv dataset.name=<name>
# -> paste the printed dataset.run_id=... into the config
```

## Run the pipeline

Prerequisites done, the pipeline is three steps. Run them in order, pasting
each step's printed `[hint]` line into the experiment config before the next
command:

```bash
uv run python -m pipeline pipeline=split_dataset  +experiment=provenance_pipeline
# -> paste the printed dataset.mlflow_uris.split_dataset=... into the config, then:
uv run python -m pipeline pipeline=embed_dataset  +experiment=provenance_pipeline
# -> paste the printed dataset.mlflow_uris.embed_dataset=... into the config, then:
uv run python -m pipeline pipeline=learn          +experiment=provenance_pipeline
```

A step whose pin is missing fails with the same hint — nothing resolves
implicitly, so a config always names the exact runs it consumed. (The
whole-chain job in [`scripts/submit_pipeline.py`](scripts/submit_pipeline.py)
automatically forwards each step's printed pins to the next step.)

Step parameters live in [`configs/pipeline/`](configs/pipeline) (samples, test
size, PCA dims, cluster count) and can be overridden on the command line:

```bash
uv run python -m pipeline pipeline=learn +experiment=provenance_pipeline pipeline.n_clusters=5
```

### Pinning upstream runs

`embed_dataset`/`learn` consume **exactly** the run pinned in
`dataset.mlflow_uris` — every step prints the line to paste, and a missing
pin fails with the same hint:

```
[hint] to pin this run for downstream steps, set in the experiment config:
         dataset.mlflow_uris.split_dataset=mlflow-artifacts:/<exp_id>/<run_id>/artifacts/train.csv
```

The dataset itself is defined by its registration run: `dataset.run_id`
(required) is the `[hint]` line printed by `register_dataset` — a config
always names the exact dataset it consumed.

## Training on the splits with mlkit

> The split steps above are the template's demo producer — your own splitter
> works too: any mlkit preprocessing app (`@hydra.main` + `@autolog`) that
> logs a `train.csv` (wsi_path column, the file list) is a valid split.
> Training verifies it against the `dataset.run_id` you already pin.

To train on this pipeline's output with the standard RationAI stack
(rationai-mlkit), you write **no provenance code at all** — the pipeline
steps above already call `log_provenance` themselves, and the training run
gets its provenance from the mlkit `@autolog` decorator (already on
`main`) plus one callback in the trainer config:

```yaml
trainer:
  callbacks:
    - _target_: rationai.mlkit.lightning.callbacks.ProvenanceCallback
      fail_fast: true
      dataset_run_id: ${dataset.run_id}    # ← the registration, from this config
      split_uris:                       # ← the bridge to this pipeline
        train: mlflow-artifacts:/<exp_id>/<run_id>/artifacts/train.csv
        test:  mlflow-artifacts:/<exp_id>/<run_id>/artifacts/test.csv
```

With `split_uris`, training **references your split run's artifacts**
instead of recomputing a split — the two runs chain in MLflow.

One form to know: `mlflow-artifacts:/…` — the run's `artifact_uri`, shown
in the MLflow UI and printed by every `[hint]` line. The callback downloads
these directly; anything else is treated as a local path.

## What to look for in the MLflow UI

- **Dataset registration** (in your project's experiment, run name
  `Dataset_<name>_<version>`): a run per dataset
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

### Adding provenance to your own step

`log_provenance(config=config, …)` logs into the **active MLflow run** — it
fails fast (`RuntimeError: No active MLflow run`) if there is none. The
`@autolog` decorator on `main` (already in both this and the mlkit template)
creates that run and injects the logger, so inside a step the function call
alone is enough; `logger` is optional and falls back to the active run.

```bash
uvx ruff check
uv run mypy .
```
