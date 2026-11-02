import math
from typing import Dict, Iterable, List, Mapping, Sequence

import torch


StateDict = Dict[str, torch.Tensor]


def _clone_state_dict(state_dict: Mapping[str, torch.Tensor]) -> StateDict:
    return {k: v.detach().clone() for k, v in state_dict.items()}


def _is_float_tensor(tensor: torch.Tensor) -> bool:
    return tensor.dtype.is_floating_point


def _update_delta(state: Mapping[str, torch.Tensor], ref_state: Mapping[str, torch.Tensor]) -> StateDict:
    delta = {}
    for k, v in state.items():
        if _is_float_tensor(v):
            delta[k] = v - ref_state[k]
        else:
            delta[k] = v.clone()
    return delta


def _state_from_delta(ref_state: Mapping[str, torch.Tensor], delta: Mapping[str, torch.Tensor]) -> StateDict:
    out = {}
    for k, v in ref_state.items():
        if _is_float_tensor(v):
            out[k] = v + delta[k]
        else:
            out[k] = v.clone()
    return out


def _parse_attack_names(config) -> List[str]:
    flag_to_name = [
        ("use_scaling_attack", "scaling"),
        ("use_random_update_attack", "random_update"),
        ("use_alie_attack", "alie"),
        ("use_slow_drift_attack", "slow_drift"),
        ("use_layerwise_sign_flip_attack", "layerwise_sign_flip"),
    ]
    enabled = [name for flag, name in flag_to_name if getattr(config, flag, False)]
    if enabled:
        return enabled

    attack_type = str(getattr(config, "attack_type", "none")).lower().strip()
    aliases = {
        "scale": "scaling",
        "scaling_attack": "scaling",
        "random": "random_update",
        "random_update_attack": "random_update",
        "alie_attack": "alie",
        "slow": "slow_drift",
        "slow_drift_attack": "slow_drift",
        "layer_sign_flip": "layerwise_sign_flip",
        "layerwise_sign_flip_attack": "layerwise_sign_flip",
    }
    attack_type = aliases.get(attack_type, attack_type)
    if attack_type in {
        "sign_flip",
        "noise",
        "scaling",
        "random_update",
        "alie",
        "slow_drift",
        "layerwise_sign_flip",
    }:
        return [attack_type]
    return []


def get_enabled_attack_names(config) -> List[str]:
    return _parse_attack_names(config)


def sign_flip_attack(
    honest_state: Mapping[str, torch.Tensor],
    ref_state: Mapping[str, torch.Tensor],
    scale: float,
) -> StateDict:
    attacked = {}
    for k, v in honest_state.items():
        if _is_float_tensor(v):
            delta = v - ref_state[k]
            attacked[k] = ref_state[k] - scale * delta
        else:
            attacked[k] = v.clone()
    return attacked


def noise_attack(
    honest_state: Mapping[str, torch.Tensor],
    ref_state: Mapping[str, torch.Tensor],
    noise_std: float,
) -> StateDict:
    attacked = {}
    for k, v in honest_state.items():
        if _is_float_tensor(v):
            delta = v - ref_state[k]
            attacked[k] = ref_state[k] + delta + noise_std * torch.randn_like(delta)
        else:
            attacked[k] = v.clone()
    return attacked


def scaling_attack(
    honest_state: Mapping[str, torch.Tensor],
    ref_state: Mapping[str, torch.Tensor],
    scale: float,
) -> StateDict:
    attacked = {}
    for k, v in honest_state.items():
        if _is_float_tensor(v):
            delta = v - ref_state[k]
            attacked[k] = ref_state[k] + scale * delta
        else:
            attacked[k] = v.clone()
    return attacked


def random_update_attack(
    honest_state: Mapping[str, torch.Tensor],
    ref_state: Mapping[str, torch.Tensor],
    scale: float,
    normalize_to_honest_update: bool,
) -> StateDict:
    attacked = {}
    for k, v in honest_state.items():
        if not _is_float_tensor(v):
            attacked[k] = v.clone()
            continue

        delta = v - ref_state[k]
        random_delta = torch.randn_like(delta)
        if normalize_to_honest_update:
            random_norm = torch.norm(random_delta.float())
            delta_norm = torch.norm(delta.float())
            if random_norm.item() > 0.0:
                random_delta = random_delta * (delta_norm / random_norm).to(random_delta.dtype)
        attacked[k] = ref_state[k] + scale * random_delta
    return attacked


def _compute_alie_delta(
    uploaded_states: Mapping[int, Mapping[str, torch.Tensor]],
    ref_states: Mapping[int, Mapping[str, torch.Tensor]],
    reference_nodes: Sequence[int],
    z: float,
) -> StateDict:
    if not reference_nodes:
        raise ValueError("ALIE attack needs at least one reference node.")

    first_node = reference_nodes[0]
    alie_delta = {}
    for k, v in uploaded_states[first_node].items():
        if not _is_float_tensor(v):
            alie_delta[k] = v.clone()
            continue

        deltas = [
            (uploaded_states[node_id][k] - ref_states[node_id][k]).float()
            for node_id in reference_nodes
        ]
        stacked = torch.stack(deltas, dim=0)
        mean = stacked.mean(dim=0).to(v.dtype)
        std = stacked.std(dim=0, unbiased=False).to(v.dtype)
        alie_delta[k] = mean + z * std
    return alie_delta


def alie_attack(
    ref_state: Mapping[str, torch.Tensor],
    alie_delta: Mapping[str, torch.Tensor],
) -> StateDict:
    return _state_from_delta(ref_state, alie_delta)


def slow_drift_attack(
    honest_state: Mapping[str, torch.Tensor],
    ref_state: Mapping[str, torch.Tensor],
    drift_scale: float,
    max_drift_scale: float,
    active_rounds: int,
) -> StateDict:
    drift = min(max_drift_scale, drift_scale * max(active_rounds, 1))
    attacked = {}
    for k, v in honest_state.items():
        if not _is_float_tensor(v):
            attacked[k] = v.clone()
            continue

        delta = v - ref_state[k]
        mean_abs = delta.float().abs().mean()
        if not math.isfinite(mean_abs.item()) or mean_abs.item() == 0.0:
            attacked[k] = v.clone()
            continue

        direction = torch.sign(delta)
        direction = torch.where(direction == 0, torch.ones_like(direction), direction)
        attacked[k] = v - drift * mean_abs.to(v.dtype) * direction
    return attacked


def _matches_layer(key: str, layer_patterns: Iterable[str]) -> bool:
    patterns = [p.strip() for p in layer_patterns if p.strip()]
    if not patterns:
        return True
    return any(pattern in key for pattern in patterns)


def layerwise_sign_flip_attack(
    honest_state: Mapping[str, torch.Tensor],
    ref_state: Mapping[str, torch.Tensor],
    scale: float,
    layer_patterns: Iterable[str],
) -> StateDict:
    attacked = {}
    for k, v in honest_state.items():
        if _is_float_tensor(v) and _matches_layer(k, layer_patterns):
            delta = v - ref_state[k]
            attacked[k] = ref_state[k] - scale * delta
        else:
            attacked[k] = v.clone()
    return attacked


def apply_attacks_to_uploaded_states(
    uploaded_states: Mapping[int, Mapping[str, torch.Tensor]],
    pre_train_states: Mapping[int, Mapping[str, torch.Tensor]],
    malicious_nodes: Iterable[int],
    config,
    round_idx: int,
) -> Dict[int, StateDict]:
    attack_names = get_enabled_attack_names(config)
    if not attack_names:
        return {node_id: _clone_state_dict(state) for node_id, state in uploaded_states.items()}

    target_nodes = sorted(set(malicious_nodes))
    attacked_states = {
        node_id: _clone_state_dict(state)
        for node_id, state in uploaded_states.items()
    }
    if not target_nodes:
        return attacked_states

    for attack_name in attack_names:
        if attack_name == "alie":
            reference_nodes = [
                node_id for node_id in attacked_states.keys()
                if node_id not in target_nodes
            ]
            if getattr(config, "alie_use_all_updates", False):
                reference_nodes = list(attacked_states.keys())
            if not reference_nodes:
                reference_nodes = list(attacked_states.keys())
            alie_delta = _compute_alie_delta(
                uploaded_states=attacked_states,
                ref_states=pre_train_states,
                reference_nodes=reference_nodes,
                z=getattr(config, "alie_z", 1.0),
            )
            for node_id in target_nodes:
                attacked_states[node_id] = alie_attack(
                    ref_state=pre_train_states[node_id],
                    alie_delta=alie_delta,
                )
            continue

        for node_id in target_nodes:
            honest_state = attacked_states[node_id]
            ref_state = pre_train_states[node_id]

            if attack_name == "sign_flip":
                attacked_states[node_id] = sign_flip_attack(
                    honest_state=honest_state,
                    ref_state=ref_state,
                    scale=getattr(config, "attack_scale", 2.0),
                )
            elif attack_name == "noise":
                attacked_states[node_id] = noise_attack(
                    honest_state=honest_state,
                    ref_state=ref_state,
                    noise_std=getattr(config, "noise_std", 0.5),
                )
            elif attack_name == "scaling":
                attacked_states[node_id] = scaling_attack(
                    honest_state=honest_state,
                    ref_state=ref_state,
                    scale=getattr(config, "scaling_factor", 10.0),
                )
            elif attack_name == "random_update":
                attacked_states[node_id] = random_update_attack(
                    honest_state=honest_state,
                    ref_state=ref_state,
                    scale=getattr(config, "random_update_scale", 1.0),
                    normalize_to_honest_update=getattr(
                        config,
                        "random_update_normalize_to_honest_update",
                        True,
                    ),
                )
            elif attack_name == "slow_drift":
                active_rounds = (round_idx + 1) - getattr(config, "attack_start_round", 1) + 1
                attacked_states[node_id] = slow_drift_attack(
                    honest_state=honest_state,
                    ref_state=ref_state,
                    drift_scale=getattr(config, "slow_drift_scale", 0.05),
                    max_drift_scale=getattr(config, "slow_drift_max_scale", 1.0),
                    active_rounds=active_rounds,
                )
            elif attack_name == "layerwise_sign_flip":
                layer_patterns = str(
                    getattr(config, "layerwise_sign_flip_layers", "")
                ).split(",")
                attacked_states[node_id] = layerwise_sign_flip_attack(
                    honest_state=honest_state,
                    ref_state=ref_state,
                    scale=getattr(config, "layerwise_sign_flip_scale", 2.0),
                    layer_patterns=layer_patterns,
                )

    return attacked_states
