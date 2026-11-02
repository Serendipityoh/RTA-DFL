
[English](README.md)

Research code for a robust and privacy-aware decentralized federated learning
(DFL) method. The repository contains the proposed training pipeline,
personalized model boundary, trust and adaptive percentile weighting,
committee-based protected aggregation, Byzantine attack settings, dataset
preprocessing, and protocol tests.

> This is a research prototype accompanying an anonymous submission. It is not
> a production security library, and the included tests are not a formal
> security proof or a third-party cryptographic audit.

## Highlights

- Decentralized training over a dynamic directed communication graph.
- Personalized learning with a shared feature extractor and a private
  prediction head.
- Multi-view update scoring, continuous historical trust, adaptive percentile
  thresholding (APT), clipping, and neighbor-weight caps.
- Committee-based protected aggregation with authenticated messages, Shamir
  recovery, commitment checks, and fail-closed handling.
- Clean, sign-flip, ALIE, and slow-drift experiment configurations.
- Reproducible JSON configuration, command-line overrides, per-round metrics,
  and a dataset-free smoke test.

## Method-to-Code Map

| Component | Implementation |
|---|---|
| Experiment entry point and result export | [`main.py`](main.py) |
| Central configuration defaults | [`config.py`](config.py) |
| Local training and private prediction head | [`node.py`](node.py) |
| Dynamic directed communication graph | [`graph_utils.py`](graph_utils.py) |
| Trust, multi-view APT, clipping, and receiver update | [`trainer.py`](trainer.py) |
| Committee assignment and protected aggregation | [`secure_protocol.py`](secure_protocol.py) |
| Byzantine attacks | [`attacks.py`](attacks.py) |
| Image and sensor models | [`model.py`](model.py), [`sensor_models.py`](sensor_models.py) |
| Dataset loading and non-IID partitioning | [`data.py`](data.py), [`sensor_data.py`](sensor_data.py) |
| Named experiment settings | [`configs/`](configs) |
| Protocol and end-to-end smoke tests | [`tests/`](tests) |

The proposed configuration enables personalization and the committee privacy
module. High-dimensional shared updates are additively shared among committee
members. The committee commitment is fixed before the projection seed is
derived, sender summaries are checked against committee-derived summaries, and
the final weighted update is recovered only after a signed weight request is
accepted.

## Repository Scope

This repository contains only the proposed method. Implementations of the
comparison baselines are not redistributed. Consequently, the included
configurations reproduce proposed-method and ablation runs only; reproducing
baseline rows requires the corresponding baseline repositories and the
settings described in the paper.

Dataset files and trained checkpoints are not included. The
[`reported_results/`](reported_results) directory is reserved for per-seed raw
metrics associated with reported tables or figures.

## Requirements

- Python 3.12 recommended
- CUDA-capable PyTorch environment for the full default experiments
  (CPU fallback is supported)
- Enough memory and storage for the selected dataset and number of nodes

The exact validated package versions are pinned in
[`requirements.txt`](requirements.txt).

## Installation

```bash
git clone <YOUR_REPOSITORY_URL>
cd AAAI27_Code

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Weights & Biases is optional. Install it only when experiment tracking is
needed:

```bash
python -m pip install -r requirements-wandb.txt
```

All provided JSON configurations disable WandB and write results locally.

## Dataset Preparation

The following datasets are supported:

| Dataset | Model | Preparation |
|---|---|---|
| CIFAR-10 | ResNet-20 | Downloaded automatically by `torchvision` |
| MNIST | ResNet-20 | Downloaded automatically by `torchvision` |
| FEMNIST | ResNet-20 | Place LEAF-style JSON files under `data/femnist/` |
| UCI HAR | SensorMLP | Place the official feature files under `data/UCI HAR Dataset/` |
| PAMAP2 | SensorMLP | Place raw protocol files or pre-split files under `data/PAMAP2_Dataset/` |

Expected layouts for manually prepared datasets:

```text
data/
├── femnist/
│   ├── train/*.json
│   └── test/*.json
├── UCI HAR Dataset/
│   ├── train/X_train.txt
│   ├── train/y_train.txt
│   ├── test/X_test.txt
│   └── test/y_test.txt
└── PAMAP2_Dataset/
    └── Protocol/*.dat
```

PAMAP2 also accepts `X_train.txt`, `y_train.txt`, `X_test.txt`, and
`y_test.txt` directly inside `data/PAMAP2_Dataset/`. Dataset files must be
obtained separately and used according to their original licenses and citation
requirements. Set a different root with `dataset_root` when needed.

## Quick Validation

Run the protocol unit tests first:

```bash
python -m unittest -v tests.test_secure_protocol
```

Then run one complete protected training round on synthetic data:

```bash
python -m tests.smoke_train
```

The smoke test needs no dataset download. It exercises local training, graph
construction, protected committee aggregation, receiver updates, evaluation,
and local metric export on CPU.

## Running Experiments

### Clean proposed-method run

The default command loads `configs/proposed_clean.json`:

```bash
python main.py
```

Equivalent explicit commands are:

```bash
python main.py --config configs/proposed_clean.json
sh scripts/run_proposed.sh
```

### Byzantine attack runs

Run all three included attack configurations sequentially:

```bash
sh scripts/run_attacks.sh
```

Or run one setting directly:

```bash
python main.py --config configs/proposed_sign_flip.json
python main.py --config configs/proposed_alie.json
python main.py --config configs/proposed_slow_drift.json
```

### Private-head ablation

```bash
python main.py --config configs/ablation_no_private_head.json
```

### Override configuration values

Use repeated `--set KEY=VALUE` arguments without editing a named
configuration:

```bash
python main.py \
  --config configs/proposed_clean.json \
  --set seed=43 \
  --set rounds=1 \
  --set num_nodes=3 \
  --set min_out_degree=2 \
  --set max_out_degree=2 \
  --set device=cpu \
  --run-name smoke_seed43
```

Override values use JSON syntax; unquoted values that are not valid JSON are
treated as strings. All behavior-changing options and their defaults are
defined in [`config.py`](config.py).

## Reproducing Multiple Seeds

Use a unique output name for each seed:

```bash
for SEED in 42 43 44
do
  python main.py \
    --config configs/proposed_clean.json \
    --set seed="$SEED" \
    --run-name "proposed_clean_seed${SEED}"
done
```

## Outputs

Each run writes to `outputs/<run-name>/`:

```text
outputs/<run-name>/
├── resolved_config.json
├── history.json
├── history.csv
└── node_metrics_final.csv
```

- `resolved_config.json`: fully resolved configuration after CLI overrides.
- `history.json`: raw round-level records, including nested values.
- `history.csv`: tabular round-level accuracy, robustness, communication,
  timing, and secure-protocol metrics.
- `node_metrics_final.csv`: final per-node loss and accuracy.



## License

Released under the [MIT License](LICENSE).

## Anonymity

The repository is prepared for anonymous review and intentionally omits author
names, affiliations, email addresses, and identifying repository links.
