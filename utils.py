import random
import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_str: str):
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def clone_state_dict(state_dict):
    return {k: v.detach().clone() for k, v in state_dict.items()}


def zeros_like_state_dict(state_dict):
    return {k: torch.zeros_like(v) for k, v in state_dict.items()}


def scale_state_dict(state_dict, scalar: float):
    return {k: v * scalar for k, v in state_dict.items()}


def sub_state_dict(a, b):
    out = {}
    for k in a.keys():
        out[k] = a[k] - b[k]
    return out


def add_scaled_state_dict_(target, src, scalar: float):
    for k in target.keys():
        target[k].add_(src[k] * scalar)
    return target


def average_state_dicts(state_dicts, weights):
    """Compute a normalized weighted average of model-state dictionaries."""

    assert len(state_dicts) == len(weights)
    total = sum(weights)
    norm_weights = [w / total for w in weights]

    avg_state = {}
    for k in state_dicts[0].keys():
        if "running_mean" in k or "running_var" in k or "num_batches_tracked" in k:
            avg_state[k] = state_dicts[0][k].clone()
        else:
            avg_state[k] = sum(
                norm_weights[i] * state_dicts[i][k]
                for i in range(len(state_dicts))
            )
    return avg_state


def weighted_trimmed_mean_state_dicts(state_dicts, weights, trim_ratio: float = 0.1):
    """Compute a coordinate-wise weighted trimmed mean.

    For each parameter coordinate, the function sorts contributor values,
    removes ``trim_ratio`` weight mass from both tails, and averages the
    remaining mass. Non-floating entries and BatchNorm running statistics are
    copied from the first contributor.
    """
    assert len(state_dicts) == len(weights)
    if len(state_dicts) == 0:
        raise ValueError("weighted_trimmed_mean_state_dicts needs at least one state_dict.")

    total = float(sum(weights))
    if total <= 0.0:
        raise ValueError("Aggregation weights must sum to a positive value.")

    trim_ratio = max(0.0, min(float(trim_ratio), 0.49))
    norm_weights = [float(w) / total for w in weights]

    if len(state_dicts) <= 2 or trim_ratio <= 0.0:
        return average_state_dicts(state_dicts, norm_weights)

    out = {}
    for k in state_dicts[0].keys():
        first_value = state_dicts[0][k]
        if (
            not first_value.dtype.is_floating_point
            or "running_mean" in k
            or "running_var" in k
            or "num_batches_tracked" in k
        ):
            out[k] = first_value.clone()
            continue

        stacked = torch.stack([state[k] for state in state_dicts], dim=0)
        sorted_values, sorted_indices = torch.sort(stacked, dim=0)
        weight_shape = [len(norm_weights)] + [1] * first_value.dim()
        weight_tensor = torch.tensor(
            norm_weights,
            dtype=first_value.dtype,
            device=first_value.device,
        ).view(weight_shape).expand_as(stacked)
        sorted_weights = torch.gather(weight_tensor, 0, sorted_indices)

        cumulative_weights = torch.cumsum(sorted_weights, dim=0)
        previous_cumulative = cumulative_weights - sorted_weights
        keep = (cumulative_weights > trim_ratio) & (previous_cumulative < 1.0 - trim_ratio)
        trimmed_weights = torch.where(keep, sorted_weights, torch.zeros_like(sorted_weights))
        denom = trimmed_weights.sum(dim=0)

        plain_weight_shape = [len(norm_weights)] + [1] * first_value.dim()
        plain_weights = torch.tensor(
            norm_weights,
            dtype=first_value.dtype,
            device=first_value.device,
        ).view(plain_weight_shape)
        fallback = torch.sum(stacked * plain_weights, dim=0)
        trimmed = torch.sum(sorted_values * trimmed_weights, dim=0) / denom.clamp_min(1e-12)
        out[k] = torch.where(denom > 0.0, trimmed, fallback)

    return out


def accuracy_from_logits(logits, targets):
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)
