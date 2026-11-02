import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from data import (
    get_dataset,
    iid_partition,
    noniid_dirichlet_partition_pair,
    build_node_dataloaders,
    build_subset_loaders,
    build_test_loader
)
from config import Config
from sensor_models import build_model_for_dataset
from node import Node
from trainer import DecentralizedTrainer
from utils import get_device, set_seed


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PACKAGE_ROOT / "configs" / "proposed_clean.json"


def _apply_config_values(cfg, values):
    unknown = sorted(set(values) - set(asdict(cfg)))
    if unknown:
        raise ValueError(f"Unknown configuration keys: {unknown}")
    for key, value in values.items():
        setattr(cfg, key, value)
    return cfg


def load_config(config_path):
    """Load a JSON experiment configuration over the dataclass defaults."""

    cfg = Config()
    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        values = json.load(handle)
    if not isinstance(values, dict):
        raise ValueError("The experiment configuration must be a JSON object.")
    return _apply_config_values(cfg, values), path


def _parse_override(item):
    if "=" not in item:
        raise ValueError(f"Expected KEY=VALUE override, got: {item}")
    key, raw_value = item.split("=", 1)
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError:
        value = raw_value
    return key, value


def _prepare_output_directory(cfg, config_path, requested_run_name=None):
    output_root = Path(cfg.output_root)
    if not output_root.is_absolute():
        output_root = PACKAGE_ROOT / output_root
    run_name = (
        requested_run_name
        or cfg.run_name
        or f"{config_path.stem}_seed{cfg.seed}"
    )
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_root = str(output_root)
    cfg.run_name = run_name
    return output_dir


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, set):
        return sorted(value)
    return str(value)


def save_history(history, output_dir):
    """Save raw round records in JSON and tabular CSV formats."""

    json_path = output_dir / "history.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            history,
            handle,
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        )

    csv_path = output_dir / "history.csv"
    fieldnames = sorted({
        key
        for record in history
        for key in record.keys()
    })
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in history:
            row = {}
            for key in fieldnames:
                value = record.get(key, "")
                if isinstance(value, (dict, list, tuple, set)):
                    value = json.dumps(
                        value,
                        sort_keys=True,
                        default=_json_default,
                    )
                row[key] = value
            writer.writerow(row)


def run_experiment(cfg, config_path, requested_run_name=None):
    output_dir = _prepare_output_directory(
        cfg,
        config_path=config_path,
        requested_run_name=requested_run_name,
    )
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    print(f"Using device: {device}")
    print(f"Output directory: {output_dir}")

    wandb_module = None
    run = None
    if cfg.use_wandb and cfg.wandb_mode != "disabled":
        try:
            import wandb as wandb_module
        except ImportError as exc:
            raise RuntimeError(
                "Weights & Biases logging is enabled, but wandb is not "
                "installed. Install requirements-wandb.txt or disable it."
            ) from exc
        run_name = cfg.wandb_run_name if cfg.wandb_run_name else None
        run = wandb_module.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity if cfg.wandb_entity else None,
            name=run_name,
            mode=cfg.wandb_mode,
            config=asdict(cfg)
        )

    # Load the selected dataset.
    train_set, test_set = get_dataset(
        cfg.dataset_name,
        cfg.dataset_root,
        image_size=cfg.image_size,
        val_ratio=cfg.val_ratio,
        seed=cfg.seed,
    )
    if hasattr(train_set, "classes"):
        cfg.num_classes = len(train_set.classes)
        print(f"Dataset {cfg.dataset_name}: {cfg.num_classes} classes -> {train_set.classes}")
    with (output_dir / "resolved_config.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(asdict(cfg), handle, indent=2, ensure_ascii=False)

    # Partition train and test samples across nodes.
    if cfg.partition_mode.lower() == "iid":
        train_node_indices = iid_partition(
            dataset=train_set,
            num_nodes=cfg.num_nodes,
            seed=cfg.seed
        )
        test_node_indices = iid_partition(
            dataset=test_set,
            num_nodes=cfg.num_nodes,
            seed=cfg.seed
        )

    elif cfg.partition_mode.lower() == "noniid":
        train_node_indices, test_node_indices = noniid_dirichlet_partition_pair(
            train_set=train_set,
            test_set=test_set,
            num_nodes=cfg.num_nodes,
            alpha=cfg.dirichlet_alpha,
            seed=cfg.seed
        )
    else:
        raise ValueError("partition_mode must be 'iid' or 'noniid'")

    # Build node-specific and shared evaluation loaders.
    node_loaders = build_node_dataloaders(
        train_set=train_set,
        node_indices=train_node_indices,
        batch_size=cfg.batch_size
    )

    node_test_loaders = build_subset_loaders(
        dataset=test_set,
        node_indices=test_node_indices,
        batch_size=256,
        shuffle=False
    )

    test_loader = build_test_loader(test_set, batch_size=256)

    # Initialize one independently optimized model per node.
    nodes = []
    for node_id in range(cfg.num_nodes):
        node = Node(
            node_id=node_id,
            model_fn=lambda: build_model_for_dataset(
                cfg.dataset_name,
                cfg.num_classes,
                getattr(train_set, "input_dim", None),
            ),
            train_loader=node_loaders[node_id],
            device=device,
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            use_scheduler=cfg.use_scheduler,
            rounds=cfg.rounds,
            min_lr=cfg.min_lr,
            init_seed=cfg.seed,
            use_personalization=cfg.use_personalization,
        )
        nodes.append(node)

    # Execute decentralized training and persist round-level metrics.
    trainer = DecentralizedTrainer(
        nodes=nodes,
        test_loader=test_loader,
        node_test_loaders=node_test_loaders,
        config=cfg
    )
    history = trainer.train()
    save_history(history, output_dir)

    print("Training finished.")
    print("Last record:", history[-1] if len(history) > 0 else None)

    if run is not None and wandb_module is not None:
        wandb_module.finish()
    return history


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the proposed decentralized federated learning method."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to a JSON experiment configuration.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override one configuration value; may be repeated.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Override the output directory name.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg, config_path = load_config(args.config)
    overrides = dict(_parse_override(item) for item in args.set)
    _apply_config_values(cfg, overrides)
    run_experiment(
        cfg,
        config_path=config_path,
        requested_run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
