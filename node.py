import torch
import torch.nn as nn
import torch.optim as optim
from utils import set_seed, clone_state_dict, sub_state_dict


class Node:
    def __init__(self, node_id, model_fn, train_loader, device,
                 lr=0.01, momentum=0.9, weight_decay=5e-4,
                 use_scheduler=True, rounds=200, min_lr=1e-4, init_seed=None,
                 use_personalization=True):
        self.node_id = node_id
        self.device = device
        self.train_loader = train_loader
        self.use_personalization = use_personalization

        if init_seed is not None:
            set_seed(init_seed)

        self.model = model_fn().to(device)
        self.bn_affine_keys = self._collect_bn_affine_keys()
        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )

        self.use_scheduler = use_scheduler
        if self.use_scheduler:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=rounds,
                eta_min=min_lr
            )

        initial_shared = self.get_state_dict()
        self.shared_u = clone_state_dict(initial_shared)
        self.pushsum_mu = 1.0
        self.shared_z = clone_state_dict(initial_shared)

        # Keep the model backbone synchronized with the communicated state.
        self._load_shared_to_model(self.shared_z)

    def _is_shared_key(self, key: str) -> bool:
        """Return whether a model-state entry participates in communication.

        Personalized runs share trainable ``shared_features`` entries while
        keeping the classifier and BatchNorm running statistics local.
        Non-personalized runs communicate the complete model state.
        """
        if not self.use_personalization:
            return True

        if not key.startswith("shared_features."):
            return False

        if (
            "running_mean" in key
            or "running_var" in key
            or "num_batches_tracked" in key
        ):
            return False

        return True

    def _collect_bn_affine_keys(self):
        bn_keys = set()
        for module_name, module in self.model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                if module.affine:
                    bn_keys.add(f"{module_name}.weight")
                    bn_keys.add(f"{module_name}.bias")
        return bn_keys

    def _load_shared_to_model(self, shared_state):
        current_state = self.model.state_dict()
        for k, v in shared_state.items():
            if self._is_shared_key(k):
                current_state[k] = v.clone()
        self.model.load_state_dict(current_state, strict=True)

    def _refresh_z_from_u(self):
        self.shared_z = clone_state_dict(self.shared_u)
        return self.shared_z

    def get_state_dict(self):
        """Return a detached copy of the communicated model state."""

        shared_state = {}
        for k, v in self.model.state_dict().items():
            if self._is_shared_key(k):
                shared_state[k] = v.detach().clone()
        return shared_state

    def load_state_dict(self, state_dict):
        """Load an aggregated communicated state into the local model."""

        self._load_shared_to_model(state_dict)
        current_shared = self.get_state_dict()
        self.shared_z = clone_state_dict(current_shared)
        self.shared_u = clone_state_dict(current_shared)
        self.pushsum_mu = 1.0

    def set_pushsum_state(self, new_u, new_mu=None):
        """Set the communicated state through the push-sum interface."""

        self.shared_u = clone_state_dict(new_u)
        self.shared_z = clone_state_dict(new_u)
        self.pushsum_mu = 1.0
        self._load_shared_to_model(self.shared_z)

    def get_shared_update(self, prev_u_state):
        """Return the communicated update relative to ``prev_u_state``."""

        return sub_state_dict(self.shared_u, prev_u_state)

    def local_train(self, epochs=1):
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for _ in range(epochs):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(x)
                loss = self.criterion(logits, y)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * x.size(0)
                total_samples += x.size(0)

        avg_loss = total_loss / max(total_samples, 1)

        if self.use_scheduler:
            self.scheduler.step()

        # Synchronize the communicated state after local optimization.
        current_shared = self.get_state_dict()
        self.shared_z = clone_state_dict(current_shared)
        self.shared_u = clone_state_dict(current_shared)
        self.pushsum_mu = 1.0

        return avg_loss

    @torch.no_grad()
    def evaluate(self, test_loader):
        self.model.eval()
        total_correct = 0
        total_samples = 0
        total_loss = 0.0

        for x, y in test_loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            loss = self.criterion(logits, y)

            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += y.size(0)
            total_loss += loss.item() * x.size(0)

        acc = total_correct / total_samples
        avg_loss = total_loss / total_samples
        return avg_loss, acc
