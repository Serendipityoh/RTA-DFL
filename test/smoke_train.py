"""Run one complete protected training round on synthetic image data."""

import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from config import Config
from node import Node
from sensor_models import build_model_for_dataset
from trainer import DecentralizedTrainer
from utils import set_seed


def main():
    cfg = Config()
    cfg.seed = 7
    cfg.device = "cpu"
    cfg.dataset_name = "cifar10"
    cfg.num_classes = 10
    cfg.num_nodes = 3
    cfg.rounds = 1
    cfg.local_epochs = 1
    cfg.batch_size = 4
    cfg.min_out_degree = 2
    cfg.max_out_degree = 2
    cfg.eval_every = 1
    cfg.use_scheduler = False
    cfg.use_malicious = False
    cfg.use_personalization = True
    cfg.use_secure_privacy_module = True
    cfg.secure_num_share_holders = 3
    cfg.secure_shamir_threshold = 2
    cfg.secure_min_valid_contributors = 3
    cfg.use_wandb = False
    cfg.wandb_mode = "disabled"

    set_seed(cfg.seed)
    generator = torch.Generator().manual_seed(cfg.seed)
    train_images = torch.randn(12, 3, 32, 32, generator=generator)
    train_labels = torch.arange(12) % cfg.num_classes
    test_images = torch.randn(6, 3, 32, 32, generator=generator)
    test_labels = torch.arange(6) % cfg.num_classes

    node_loaders = {
        node_id: DataLoader(
            TensorDataset(
                train_images[node_id * 4:(node_id + 1) * 4],
                train_labels[node_id * 4:(node_id + 1) * 4],
            ),
            batch_size=cfg.batch_size,
            shuffle=False,
        )
        for node_id in range(cfg.num_nodes)
    }
    test_loader = DataLoader(
        TensorDataset(test_images, test_labels),
        batch_size=3,
        shuffle=False,
    )
    node_test_loaders = {
        node_id: test_loader
        for node_id in range(cfg.num_nodes)
    }

    nodes = [
        Node(
            node_id=node_id,
            model_fn=lambda: build_model_for_dataset(
                cfg.dataset_name,
                cfg.num_classes,
            ),
            train_loader=node_loaders[node_id],
            device=torch.device("cpu"),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            use_scheduler=cfg.use_scheduler,
            rounds=cfg.rounds,
            min_lr=cfg.min_lr,
            init_seed=cfg.seed,
            use_personalization=cfg.use_personalization,
        )
        for node_id in range(cfg.num_nodes)
    ]

    with tempfile.TemporaryDirectory(prefix="aaai27-smoke-") as tmp_dir:
        cfg.output_root = str(Path(tmp_dir))
        cfg.run_name = "smoke"
        trainer = DecentralizedTrainer(
            nodes=nodes,
            test_loader=test_loader,
            node_test_loaders=node_test_loaders,
            config=cfg,
        )
        history = trainer.train()

    if len(history) != 1:
        raise RuntimeError("The smoke test did not complete exactly one round.")
    record = history[0]
    if record["secure_committee_successes"] != cfg.num_nodes:
        raise RuntimeError(
            "Not every receiver completed committee aggregation."
        )
    print(
        "Smoke test passed:",
        {
            "rounds": len(history),
            "committee_successes": record["secure_committee_successes"],
            "committee_aborts": record["secure_committee_aborts"],
        },
    )


if __name__ == "__main__":
    main()
