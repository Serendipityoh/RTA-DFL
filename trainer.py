import csv
import math
import os
import random
import time

import torch
import torch.nn.functional as F

try:
    import wandb
except ImportError:
    wandb = None

from attacks import apply_attacks_to_uploaded_states, get_enabled_attack_names
from graph_utils import generate_dynamic_directed_graph, build_aggregation_weights
from utils import average_state_dicts, clone_state_dict, weighted_trimmed_mean_state_dicts


class DecentralizedTrainer:
    def __init__(self, nodes, test_loader, node_test_loaders, config):
        self.nodes = nodes
        self.test_loader = test_loader
        self.node_test_loaders = node_test_loaders
        self.config = config
        self.num_nodes = len(nodes)
        self.rounds_to_target = None
        self.total_time_to_target = None
        self.total_comm_mb_to_target = None
        self.elapsed_time = 0.0
        self.cumulative_comm_mb = 0.0

        m = int(self.num_nodes * self.config.malicious_ratio) if self.config.use_malicious else 0
        rng = random.Random(self.config.seed + 999)
        self.malicious_nodes = set(rng.sample(range(self.num_nodes), m)) if m > 0 else set()

        # Receiver-specific trust assigned to each sender.
        self.trust_scores = {
            i: {j: self.config.trust_init for j in range(self.num_nodes) if j != i}
            for i in range(self.num_nodes)
        }

        # Per-node update EMA used by the momentum-distance view.
        self.momentum_ema = {i: None for i in range(self.num_nodes)}

        # Historical composite distances for receiver-sender MAD scoring.
        self.distance_history = {
            i: {j: [] for j in range(self.num_nodes) if j != i}
            for i in range(self.num_nodes)
        }

        # Initialize the privacy module only when secure aggregation is enabled.
        self.secure_protocol = None
        if getattr(self.config, "use_secure_privacy_module", False):
            from secure_protocol import SecureProtocolContext

            self.secure_protocol = SecureProtocolContext(
                num_nodes=self.num_nodes,
                config=self.config
            )

    def _model_state_num_bytes(self):
        total_bytes = 0
        for tensor in self.nodes[0].get_state_dict().values():
            total_bytes += tensor.numel() * tensor.element_size()
        return total_bytes

    def estimate_comm_mb_per_round(self, in_neighbors):
        # Count one full model-state transmission per directed edge.
        directed_edges = sum(len(in_neighbors[i]) for i in range(self.num_nodes))
        return directed_edges * self._model_state_num_bytes() / (1024.0 ** 2)

    def estimate_secure_comm_mb_per_round(self, in_neighbors):
        """Estimate per-round communication with committee aggregation enabled.

        The estimate counts serialized tensor payloads and protocol metadata:
        each update is sent as H additive shares; summary validation transmits
        field elements and commitments; binding messages add hashes, norms,
        an AEAD tag, a nonce, and an Ed25519 signature.
        """
        model_bytes = self._model_state_num_bytes()
        directed_edges = sum(len(in_neighbors[i]) for i in range(self.num_nodes))
        baseline_model_bytes = directed_edges * model_bytes

        if self.secure_protocol is None:
            baseline_model_mb = baseline_model_bytes / (1024.0 ** 2)
            return {
                "model_comm_mb_per_round": baseline_model_mb,
                "secure_share_model_mb_per_round": 0.0,
                "secure_metadata_mb_per_round": 0.0,
                "secure_total_comm_mb_per_round": baseline_model_mb,
                "secure_comm_multiplier": 1.0,
            }

        num_holders = int(getattr(self.config, "secure_num_share_holders", 3))
        threshold = int(getattr(self.config, "secure_shamir_threshold", 2))
        summary_dim = int(getattr(self.config, "secure_summary_dim", 128))

        secure_model_bytes = baseline_model_bytes * max(num_holders, 1)
        field_element_bytes = 4  # FIELD_PRIME < 2^31 fits in four bytes.
        summary_share_bytes = self.num_nodes * num_holders * summary_dim * field_element_bytes
        vss_commitment_bytes = self.num_nodes * summary_dim * threshold * field_element_bytes
        proof_message_bytes = self.num_nodes * (64 + 64 + 16 + 12 + 16 + 128)
        metadata_bytes = summary_share_bytes + vss_commitment_bytes + proof_message_bytes
        secure_total_bytes = secure_model_bytes + metadata_bytes

        return {
            "model_comm_mb_per_round": baseline_model_bytes / (1024.0 ** 2),
            "secure_share_model_mb_per_round": secure_model_bytes / (1024.0 ** 2),
            "secure_metadata_mb_per_round": metadata_bytes / (1024.0 ** 2),
            "secure_total_comm_mb_per_round": secure_total_bytes / (1024.0 ** 2),
            "secure_comm_multiplier": secure_total_bytes / max(baseline_model_bytes, 1),
        }

    def _safe_mean(self, values, default=0.0):
        return sum(values) / len(values) if values else default

    def sample_participating_nodes(self, round_idx):
        m = max(1, int(self.num_nodes * self.config.participation_rate))
        rng = random.Random(self.config.seed + round_idx)
        return sorted(rng.sample(range(self.num_nodes), m))

    def diff_state_dict(self, new_state, ref_state):
        diff = {}
        for k in new_state.keys():
            if new_state[k].dtype.is_floating_point:
                diff[k] = new_state[k] - ref_state[k]
            else:
                diff[k] = new_state[k].clone()
        return diff

    def apply_aggregated_update(self, base_state, aggregated_update, step_size=1.0):
        """Apply an aggregated shared update to the receiver's round-start state."""

        new_state = {}
        for key, base_value in base_state.items():
            if base_value.dtype.is_floating_point:
                new_state[key] = (
                    base_value
                    + float(step_size) * aggregated_update[key]
                )
            else:
                new_state[key] = base_value.clone()
        return new_state

    def state_dict_to_vector(self, state_dict):
        vecs = []
        for k, v in state_dict.items():
            if not v.dtype.is_floating_point:
                continue
            if "running_mean" in k or "running_var" in k or "num_batches_tracked" in k:
                continue
            vecs.append(v.float().reshape(-1))
        if len(vecs) == 0:
            return torch.zeros(1, dtype=torch.float32)
        return torch.cat(vecs)

    def state_dict_l2_norm(self, state_dict):
        vec = self.state_dict_to_vector(state_dict)
        return torch.norm(vec, p=2).item()

    def _median_mad(self, values):
        if len(values) == 0:
            return 0.0, 0.0
        tensor = torch.tensor(values, dtype=torch.float32)
        median = torch.median(tensor).item()
        abs_dev = torch.abs(tensor - median)
        mad = torch.median(abs_dev).item()
        return median, mad

    def _history_mad_score(self, node_id, src_node, raw_distance):
        """Score the current distance against a rolling median-MAD baseline.

        Persistent deviations receive lower scores even when they remain close
        to the cross-node distribution in a single round.
        """
        history = self.distance_history[node_id][src_node]
        if len(history) < max(3, int(getattr(self.config, "historical_mad_window", 10)) // 2):
            return 1.0, 0.0, 0.0

        median, mad = self._median_mad(history)
        threshold = median + float(getattr(self.config, "historical_mad_k", 2.5)) * (mad + 1e-12)
        if raw_distance <= threshold:
            return 1.0, median, mad

        excess = raw_distance - threshold
        score = math.exp(-excess / (mad + 1e-12))
        return max(0.0, min(1.0, float(score))), median, mad

    def _record_distance_history(self, node_id, src_node, raw_distance):
        history = self.distance_history[node_id][src_node]
        history.append(float(raw_distance))
        max_window = int(getattr(self.config, "historical_mad_window", 10))
        if len(history) > max_window:
            del history[:-max_window]

    def _ema_update_state_dict(self, old_state, new_state, beta):
        if old_state is None:
            return clone_state_dict(new_state)

        out = {}
        for k in new_state.keys():
            if new_state[k].dtype.is_floating_point:
                out[k] = beta * old_state[k] + (1.0 - beta) * new_state[k]
            else:
                out[k] = new_state[k].clone()
        return out

    def update_momentum_buffers(self, current_updates):
        beta = self.config.ema_momentum_beta
        for node_id in range(self.num_nodes):
            self.momentum_ema[node_id] = self._ema_update_state_dict(
                self.momentum_ema[node_id],
                current_updates[node_id],
                beta
            )

    def _layer_group_name(self, key: str):
        parts = key.split(".")
        if len(parts) >= 3 and parts[1].isdigit() and parts[2].isdigit():
            return ".".join(parts[:3])  # Example: shared_features.3.0
        if len(parts) >= 2:
            return ".".join(parts[:2])  # Example: shared_features.1
        return key

    def _state_dict_to_layer_vectors(self, state_dict):
        groups = {}
        for k, v in state_dict.items():
            if not v.dtype.is_floating_point:
                continue
            if "running_mean" in k or "running_var" in k or "num_batches_tracked" in k:
                continue
            g = self._layer_group_name(k)
            groups.setdefault(g, []).append(v.float().reshape(-1))

        layer_vecs = {}
        for g, vec_list in groups.items():
            layer_vecs[g] = torch.cat(vec_list) if len(vec_list) > 0 else torch.zeros(1, dtype=torch.float32)
        return layer_vecs

    def _cosine_distance(self, vec_a, vec_b):
        if vec_a.numel() == 0 or vec_b.numel() == 0:
            return 0.0

        norm_a = torch.norm(vec_a, p=2).item()
        norm_b = torch.norm(vec_b, p=2).item()

        if norm_a < 1e-12 and norm_b < 1e-12:
            return 0.0

        sim = F.cosine_similarity(vec_a.unsqueeze(0), vec_b.unsqueeze(0), dim=1).item()
        sim = max(-1.0, min(1.0, sim))
        return 0.5 * (1.0 - sim)  # Scale cosine distance to [0, 1].

    def _norm_ratio_distance(self, state_a, state_b):
        norm_a = self.state_dict_l2_norm(state_a)
        norm_b = self.state_dict_l2_norm(state_b)

        ratio = max(norm_a, norm_b) / (min(norm_a, norm_b) + 1e-12)
        dist = abs(norm_b - norm_a) / (norm_a + norm_b + 1e-12)  # Bounded in [0, 1].
        return dist, ratio

    def _layerwise_distance(self, state_a, state_b):
        layers_a = self._state_dict_to_layer_vectors(state_a)
        layers_b = self._state_dict_to_layer_vectors(state_b)

        common_keys = sorted(set(layers_a.keys()) & set(layers_b.keys()))
        if len(common_keys) == 0:
            return 0.0

        dists = []
        for key in common_keys:
            dists.append(self._cosine_distance(layers_a[key], layers_b[key]))

        return sum(dists) / len(dists)

    def _compute_pair_metrics(self, node_id, src_node, uploaded_states, pre_train_states):
        self_update = self.diff_state_dict(
            uploaded_states[node_id],
            pre_train_states[node_id]
        )
        nbr_update = self.diff_state_dict(
            uploaded_states[src_node],
            pre_train_states[src_node]
        )

        self_vec = self.state_dict_to_vector(self_update)
        nbr_vec = self.state_dict_to_vector(nbr_update)

        cosine_dist = 0.0
        if self.config.use_cosine_distance:
            cosine_dist = self._cosine_distance(self_vec, nbr_vec)

        norm_ratio_dist = 0.0
        norm_ratio = 1.0
        if self.config.use_norm_ratio_distance:
            norm_ratio_dist, norm_ratio = self._norm_ratio_distance(self_update, nbr_update)

        momentum_dist = 0.0
        if self.config.use_ema_momentum_distance:
            ema_self = self.momentum_ema[node_id] if self.momentum_ema[node_id] is not None else self_update
            ema_nbr = self.momentum_ema[src_node] if self.momentum_ema[src_node] is not None else nbr_update
            ema_self_vec = self.state_dict_to_vector(ema_self)
            ema_nbr_vec = self.state_dict_to_vector(ema_nbr)
            momentum_dist = self._cosine_distance(ema_self_vec, ema_nbr_vec)

        layerwise_dist = 0.0
        if self.config.use_layerwise_distance:
            layerwise_dist = self._layerwise_distance(self_update, nbr_update)

        return {
            "cosine_dist": cosine_dist,
            "norm_ratio_dist": norm_ratio_dist,
            "norm_ratio": norm_ratio,
            "momentum_dist": momentum_dist,
            "layerwise_dist": layerwise_dist,
        }

    def _combine_distance(self, metrics):
        parts = []
        weights = []

        if self.config.use_cosine_distance:
            parts.append(metrics["cosine_dist"])
            weights.append(self.config.dist_weight_cosine)

        if self.config.use_norm_ratio_distance:
            parts.append(metrics["norm_ratio_dist"])
            weights.append(self.config.dist_weight_norm_ratio)

        if self.config.use_ema_momentum_distance:
            parts.append(metrics["momentum_dist"])
            weights.append(self.config.dist_weight_momentum)

        if self.config.use_layerwise_distance:
            parts.append(metrics["layerwise_dist"])
            weights.append(self.config.dist_weight_layerwise)

        if len(parts) == 0 or sum(weights) <= 0:
            return 0.0

        total_w = sum(weights)
        return sum(d * w for d, w in zip(parts, weights)) / total_w

    def _compute_apt_scores(self, raw_distance_map):
        if len(raw_distance_map) == 0:
            return {}, 0.0, 0.0

        if len(raw_distance_map) == 1:
            only_key = next(iter(raw_distance_map.keys()))
            only_val = raw_distance_map[only_key]
            return {only_key: 1.0}, only_val, only_val

        vals = torch.tensor(list(raw_distance_map.values()), dtype=torch.float32)
        q_low = torch.quantile(vals, self.config.apt_accept_quantile).item()
        q_high = torch.quantile(vals, self.config.apt_reject_quantile).item()

        if q_high <= q_low:
            q_high = q_low + 1e-6

        scores = {}
        for src_node, d in raw_distance_map.items():
            if d <= q_low:
                s = 1.0
            elif d >= q_high:
                s = 0.0
            else:
                s = 1.0 - (d - q_low) / (q_high - q_low)

            scores[src_node] = max(0.0, min(1.0, float(s)))

        return scores, q_low, q_high

    def update_trust_score(self, node_id, src_node, instant_score):
        # Asymmetric smoothing penalizes trust decreases faster than it rewards recovery.
        old_trust = self.trust_scores[node_id][src_node]
        if instant_score < old_trust:
            beta = getattr(self.config, "trust_punish_beta", self.config.trust_beta)
        else:
            beta = getattr(self.config, "trust_reward_beta", self.config.trust_beta)

        new_trust = beta * old_trust + (1.0 - beta) * instant_score
        new_trust = max(self.config.trust_min, min(self.config.trust_max, new_trust))

        self.trust_scores[node_id][src_node] = new_trust
        return new_trust

    def clip_update_state_dict(self, uploaded_state, ref_state):
        # Bound the L2 norm of the uploaded update without changing its direction.
        max_norm = float(getattr(self.config, "max_update_norm", 0.0))
        update_state = self.diff_state_dict(uploaded_state, ref_state)
        update_norm = self.state_dict_l2_norm(update_state)

        if max_norm <= 0.0 or update_norm <= max_norm:
            return {
                k: v.detach().clone() if v.dtype.is_floating_point else v.clone()
                for k, v in uploaded_state.items()
            }, update_norm, False

        scale = max_norm / (update_norm + 1e-12)
        clipped_state = {}
        for k, v in uploaded_state.items():
            if v.dtype.is_floating_point:
                clipped_state[k] = ref_state[k] + (v - ref_state[k]) * scale
            else:
                clipped_state[k] = v.clone()
        return clipped_state, update_norm, True

    def clip_updates_by_adaptive_mad(self, uploaded_states, pre_train_states):
        """Clip update norms at the round median plus a MAD-scaled margin.

        The data-dependent bound adapts to training scale and remains capped by
        ``max_update_norm`` when a positive fixed bound is configured.
        """
        update_norms = []
        for node_id in range(self.num_nodes):
            update = self.diff_state_dict(uploaded_states[node_id], pre_train_states[node_id])
            update_norms.append(self.state_dict_l2_norm(update))

        median, mad = self._median_mad(update_norms)
        adaptive_bound = median + float(getattr(self.config, "mad_clip_k", 2.5)) * (mad + 1e-12)
        fixed_bound = float(getattr(self.config, "max_update_norm", 0.0))
        if fixed_bound > 0.0:
            adaptive_bound = min(adaptive_bound, fixed_bound)
        adaptive_bound = max(adaptive_bound, 1e-12)

        clipped_states = {}
        clipped_count = 0
        for node_id in range(self.num_nodes):
            update_norm = update_norms[node_id]
            if update_norm <= adaptive_bound:
                clipped_states[node_id] = {
                    k: v.detach().clone() if v.dtype.is_floating_point else v.clone()
                    for k, v in uploaded_states[node_id].items()
                }
                continue

            clipped_count += 1
            scale = adaptive_bound / (update_norm + 1e-12)
            clipped_state = {}
            for k, v in uploaded_states[node_id].items():
                if v.dtype.is_floating_point:
                    clipped_state[k] = pre_train_states[node_id][k] + (v - pre_train_states[node_id][k]) * scale
                else:
                    clipped_state[k] = v.clone()
            clipped_states[node_id] = clipped_state

        return clipped_states, update_norms, clipped_count, adaptive_bound, median, mad

    def _cap_neighbor_weights(self, node_id, self_weight, neighbor_weights):
        # Cap each neighbor and return unused aggregation mass to the receiver.
        max_neighbor_weight = float(getattr(self.config, "max_neighbor_weight", 1.0))
        if max_neighbor_weight <= 0.0:
            total = self_weight + sum(neighbor_weights.values())
            if total <= 0.0:
                return {node_id: 1.0}, 0
            normalized = {node_id: self_weight / total}
            for key, value in neighbor_weights.items():
                normalized[key] = value / total
            return normalized, 0

        neighbor_items = {
            key: max(float(value), 0.0)
            for key, value in neighbor_weights.items()
            if key != node_id and float(value) > 0.0
        }
        if not neighbor_items:
            return {node_id: 1.0}, 0

        capped_count = 0
        remaining_mass = max(1.0 - max(float(self_weight), 0.0), 0.0)
        active = dict(neighbor_items)
        capped = {}

        while active and remaining_mass > 0.0:
            active_total = sum(active.values())
            if active_total <= 0.0:
                break

            scale = remaining_mass / active_total
            newly_capped = []
            for key, raw_value in active.items():
                proposed = raw_value * scale
                if proposed > max_neighbor_weight:
                    capped[key] = max_neighbor_weight
                    remaining_mass -= max_neighbor_weight
                    newly_capped.append(key)

            if not newly_capped:
                for key, raw_value in active.items():
                    capped[key] = raw_value * scale
                remaining_mass = 0.0
                break

            capped_count += len(newly_capped)
            for key in newly_capped:
                active.pop(key, None)

        final_self_weight = max(float(self_weight), 0.0) + max(remaining_mass, 0.0)
        final_weights = {node_id: final_self_weight}
        final_weights.update(capped)

        total = sum(final_weights.values())
        if total <= 0.0:
            return {node_id: 1.0}, capped_count

        return {key: value / total for key, value in final_weights.items()}, capped_count

    def _collect_malicious_weight_stats(self, node_id, filtered_weights):
        # Separate neighbor weight assigned to configured malicious and honest nodes.
        malicious_nodes = getattr(self, "malicious_nodes", set())
        if not malicious_nodes:
            return {
                "malicious_weight_sum": 0.0,
                "honest_weight_sum": 0.0,
                "malicious_neighbor_count": 0,
                "honest_neighbor_count": 0,
            }

        malicious_weight_sum = 0.0
        honest_weight_sum = 0.0
        malicious_neighbor_count = 0
        honest_neighbor_count = 0

        for src_node, weight in filtered_weights.items():
            if src_node == node_id:
                continue
            if src_node in malicious_nodes:
                malicious_weight_sum += float(weight)
                malicious_neighbor_count += 1
            else:
                honest_weight_sum += float(weight)
                honest_neighbor_count += 1

        total_neighbor_weight = malicious_weight_sum + honest_weight_sum
        if total_neighbor_weight <= 0.0:
            return {
                "malicious_weight_sum": 0.0,
                "honest_weight_sum": 0.0,
                "malicious_neighbor_count": malicious_neighbor_count,
                "honest_neighbor_count": honest_neighbor_count,
            }

        return {
            "malicious_weight_sum": malicious_weight_sum,
            "honest_weight_sum": honest_weight_sum,
            "malicious_neighbor_count": malicious_neighbor_count,
            "honest_neighbor_count": honest_neighbor_count,
        }

    def _make_empty_robust_accumulator(self):
        return {
            "total_neighbors": 0,
            "rejected_neighbors": 0,
            "suspicious_neighbors": 0,
            "accepted_neighbors": 0,
            "capped_neighbors": 0,
            "honest_total_neighbors": 0,
            "byzantine_total_neighbors": 0,
            "honest_rejected_neighbors": 0,
            "byzantine_accepted_neighbors": 0,
            "malicious_weight_sum": 0.0,
            "honest_weight_sum": 0.0,
            "malicious_neighbor_count": 0,
            "honest_neighbor_count": 0,
            "avg_raw_distance": [],
            "avg_apt_score": [],
            "avg_apt_score_honest": [],
            "avg_apt_score_byzantine": [],
            "avg_cos_distance_honest": [],
            "avg_cos_distance_byzantine": [],
            "avg_norm_distance_honest": [],
            "avg_norm_distance_byzantine": [],
            "avg_historical_mad_score": [],
            "avg_instant_score": [],
        }

    def _add_robust_stats(self, accumulator, robust_stats, weight_stats):
        accumulator["total_neighbors"] += robust_stats["total_neighbors"]
        accumulator["rejected_neighbors"] += robust_stats["rejected_neighbors"]
        accumulator["suspicious_neighbors"] += robust_stats["suspicious_neighbors"]
        accumulator["accepted_neighbors"] += robust_stats["accepted_neighbors"]
        accumulator["capped_neighbors"] += robust_stats.get("capped_neighbors", 0)
        accumulator["honest_total_neighbors"] += robust_stats.get("honest_total_neighbors", 0)
        accumulator["byzantine_total_neighbors"] += robust_stats.get("byzantine_total_neighbors", 0)
        accumulator["honest_rejected_neighbors"] += robust_stats.get("honest_rejected_neighbors", 0)
        accumulator["byzantine_accepted_neighbors"] += robust_stats.get("byzantine_accepted_neighbors", 0)
        accumulator["malicious_weight_sum"] += weight_stats["malicious_weight_sum"]
        accumulator["honest_weight_sum"] += weight_stats["honest_weight_sum"]
        accumulator["malicious_neighbor_count"] += weight_stats["malicious_neighbor_count"]
        accumulator["honest_neighbor_count"] += weight_stats["honest_neighbor_count"]
        accumulator["avg_raw_distance"].append(robust_stats["avg_raw_distance"])
        accumulator["avg_apt_score"].append(robust_stats["avg_apt_score"])
        if robust_stats.get("honest_total_neighbors", 0) > 0:
            accumulator["avg_apt_score_honest"].append(robust_stats.get("avg_apt_score_honest", 0.0))
            accumulator["avg_cos_distance_honest"].append(robust_stats.get("avg_cos_distance_honest", 0.0))
            accumulator["avg_norm_distance_honest"].append(robust_stats.get("avg_norm_distance_honest", 0.0))
        if robust_stats.get("byzantine_total_neighbors", 0) > 0:
            accumulator["avg_apt_score_byzantine"].append(robust_stats.get("avg_apt_score_byzantine", 0.0))
            accumulator["avg_cos_distance_byzantine"].append(robust_stats.get("avg_cos_distance_byzantine", 0.0))
            accumulator["avg_norm_distance_byzantine"].append(robust_stats.get("avg_norm_distance_byzantine", 0.0))
        accumulator["avg_historical_mad_score"].append(robust_stats.get("avg_historical_mad_score", 1.0))
        accumulator["avg_instant_score"].append(robust_stats["avg_instant_score"])

    def _summarize_robust_accumulator(self, accumulator):
        malicious_weight_sum = accumulator["malicious_weight_sum"]
        honest_weight_sum = accumulator["honest_weight_sum"]
        malicious_neighbor_count = accumulator["malicious_neighbor_count"]
        honest_neighbor_count = accumulator["honest_neighbor_count"]
        total_neighbor_weight = malicious_weight_sum + honest_weight_sum
        avg_weight_malicious = malicious_weight_sum / max(malicious_neighbor_count, 1)
        avg_weight_honest = honest_weight_sum / max(honest_neighbor_count, 1)
        total_neighbors = accumulator["total_neighbors"]
        honest_total_neighbors = accumulator["honest_total_neighbors"]
        byzantine_total_neighbors = accumulator["byzantine_total_neighbors"]

        return {
            "total_neighbors": total_neighbors,
            "rejected_neighbors": accumulator["rejected_neighbors"],
            "suspicious_neighbors": accumulator["suspicious_neighbors"],
            "accepted_neighbors": accumulator["accepted_neighbors"],
            "capped_neighbors": accumulator["capped_neighbors"],
            "honest_total_neighbors": honest_total_neighbors,
            "byzantine_total_neighbors": byzantine_total_neighbors,
            "honest_rejected_neighbors": accumulator["honest_rejected_neighbors"],
            "byzantine_accepted_neighbors": accumulator["byzantine_accepted_neighbors"],
            "rejection_rate": accumulator["rejected_neighbors"] / max(total_neighbors, 1),
            "false_rejection_rate": accumulator["honest_rejected_neighbors"] / max(honest_total_neighbors, 1),
            "false_acceptance_rate": accumulator["byzantine_accepted_neighbors"] / max(byzantine_total_neighbors, 1),
            "avg_raw_distance": sum(accumulator["avg_raw_distance"]) / max(len(accumulator["avg_raw_distance"]), 1),
            "avg_apt_score": sum(accumulator["avg_apt_score"]) / max(len(accumulator["avg_apt_score"]), 1),
            "avg_apt_score_honest": self._safe_mean(accumulator["avg_apt_score_honest"]),
            "avg_apt_score_byzantine": self._safe_mean(accumulator["avg_apt_score_byzantine"]),
            "avg_cos_distance_honest": self._safe_mean(accumulator["avg_cos_distance_honest"]),
            "avg_cos_distance_byzantine": self._safe_mean(accumulator["avg_cos_distance_byzantine"]),
            "avg_norm_distance_honest": self._safe_mean(accumulator["avg_norm_distance_honest"]),
            "avg_norm_distance_byzantine": self._safe_mean(accumulator["avg_norm_distance_byzantine"]),
            "avg_historical_mad_score": sum(accumulator["avg_historical_mad_score"]) / max(len(accumulator["avg_historical_mad_score"]), 1),
            "avg_instant_score": sum(accumulator["avg_instant_score"]) / max(len(accumulator["avg_instant_score"]), 1),
            "malicious_weight_share": (
                malicious_weight_sum / total_neighbor_weight
                if total_neighbor_weight > 0.0 else 0.0
            ),
            "avg_weight_malicious": avg_weight_malicious,
            "avg_weight_honest": avg_weight_honest,
            "malicious_to_honest_weight_ratio": (
                avg_weight_malicious / max(avg_weight_honest, 1e-12)
                if honest_neighbor_count > 0 else 0.0
            ),
        }

    def aggregate_state_dicts(self, state_dicts, weights):
        aggregation_rule = str(getattr(self.config, "aggregation_rule", "weighted_average")).lower()
        if aggregation_rule == "weighted_trimmed_mean":
            return weighted_trimmed_mean_state_dicts(
                state_dicts=state_dicts,
                weights=weights,
                trim_ratio=getattr(self.config, "trim_ratio", 0.1),
            )
        return average_state_dicts(state_dicts, weights)

    def filter_neighbors_by_trust_apt(self, round_idx, node_id, agg_weights, uploaded_states, pre_train_states):
        filtered = {node_id: agg_weights[node_id]}

        raw_distance_map = {}
        pair_metrics = {}

        for src_node, _ in agg_weights.items():
            if src_node == node_id:
                continue
            metrics = self._compute_pair_metrics(
                node_id=node_id,
                src_node=src_node,
                uploaded_states=uploaded_states,
                pre_train_states=pre_train_states
            )
            pair_metrics[src_node] = metrics
            raw_distance_map[src_node] = self._combine_distance(metrics)

        apt_enabled = (
            self.config.use_apt
            and (round_idx + 1) >= self.config.apt_start_round
            and len(raw_distance_map) > 0
        )

        if apt_enabled:
            apt_scores, q_low, q_high = self._compute_apt_scores(raw_distance_map)
        else:
            apt_scores = {src_node: 1.0 for src_node in raw_distance_map.keys()}
            q_low, q_high = 0.0, 0.0

        total_neighbors = len(raw_distance_map)
        rejected_neighbors = 0
        suspicious_neighbors = 0
        accepted_neighbors = 0

        raw_d_list = []
        apt_score_list = []
        apt_score_honest_list = []
        apt_score_byzantine_list = []
        cos_distance_honest_list = []
        cos_distance_byzantine_list = []
        norm_distance_honest_list = []
        norm_distance_byzantine_list = []
        historical_mad_score_list = []
        instant_score_list = []
        honest_total_neighbors = 0
        byzantine_total_neighbors = 0
        honest_rejected_neighbors = 0
        byzantine_accepted_neighbors = 0

        for src_node, base_w in agg_weights.items():
            if src_node == node_id:
                continue

            raw_d = raw_distance_map[src_node]
            apt_score = apt_scores[src_node]
            historical_mad_score = 1.0

            historical_mad_enabled = (
                getattr(self.config, "use_historical_mad_apt", False)
                and (round_idx + 1) >= getattr(self.config, "historical_mad_apt_start_round", 10)
            )
            if historical_mad_enabled:
                historical_mad_score, _, _ = self._history_mad_score(
                    node_id=node_id,
                    src_node=src_node,
                    raw_distance=raw_d,
                )
                apt_score *= max(historical_mad_score, 1e-6) ** getattr(
                    self.config,
                    "historical_mad_weight_power",
                    2.0,
                )

            # Convert the composite distance to a continuous similarity score.
            metric_score = math.exp(
                - raw_d / max(self.config.trust_distance_temperature, 1e-12)
            )

            # Blend distribution-based APT and pairwise metric evidence.
            instant_score = (
                self.config.trust_instant_mix * apt_score
                + (1.0 - self.config.trust_instant_mix) * metric_score
            )

            if self.config.use_trust_score:
                trust = self.update_trust_score(
                    node_id=node_id,
                    src_node=src_node,
                    instant_score=instant_score
                )
            else:
                trust = 1.0

            raw_d_list.append(raw_d)
            apt_score_list.append(apt_score)
            historical_mad_score_list.append(historical_mad_score)
            instant_score_list.append(instant_score)

            is_byzantine = src_node in self.malicious_nodes
            if is_byzantine:
                byzantine_total_neighbors += 1
                apt_score_byzantine_list.append(apt_score)
                cos_distance_byzantine_list.append(pair_metrics[src_node]["cosine_dist"])
                norm_distance_byzantine_list.append(pair_metrics[src_node]["norm_ratio_dist"])
            else:
                honest_total_neighbors += 1
                apt_score_honest_list.append(apt_score)
                cos_distance_honest_list.append(pair_metrics[src_node]["cosine_dist"])
                norm_distance_honest_list.append(pair_metrics[src_node]["norm_ratio_dist"])

            # Append only after scoring so the current value is not its own baseline.
            self._record_distance_history(node_id, src_node, raw_d)

            # Optionally reject neighbors below the configured APT threshold.
            if apt_enabled and apt_score <= self.config.apt_hard_reject_score:
                rejected_neighbors += 1
                if is_byzantine:
                    pass
                else:
                    honest_rejected_neighbors += 1
                continue

            if apt_score >= (2.0 / 3.0):
                accepted_neighbors += 1
                if is_byzantine:
                    byzantine_accepted_neighbors += 1
            elif apt_score <= (1.0 / 3.0):
                suspicious_neighbors += 1
            else:
                suspicious_neighbors += 1

            eff_w = base_w

            if apt_enabled:
                eff_w *= max(apt_score, 1e-6) ** self.config.apt_weight_power

            if self.config.use_trust_score:
                eff_w *= max(trust, 1e-6) ** self.config.trust_weight_power

            if eff_w > 0.0:
                filtered[src_node] = eff_w

        total_w = sum(filtered.values())
        if total_w <= 0.0:
            filtered = {node_id: 1.0}
        else:
            filtered = {k: v / total_w for k, v in filtered.items()}

        self_weight = filtered.get(node_id, 0.0)
        neighbor_weights = {k: v for k, v in filtered.items() if k != node_id}
        filtered, capped_neighbors = self._cap_neighbor_weights(
            node_id=node_id,
            self_weight=self_weight,
            neighbor_weights=neighbor_weights,
        )

        stats = {
            "total_neighbors": total_neighbors,
            "rejected_neighbors": rejected_neighbors,
            "suspicious_neighbors": suspicious_neighbors,
            "accepted_neighbors": accepted_neighbors,
            "capped_neighbors": capped_neighbors,
            "honest_total_neighbors": honest_total_neighbors,
            "byzantine_total_neighbors": byzantine_total_neighbors,
            "honest_rejected_neighbors": honest_rejected_neighbors,
            "byzantine_accepted_neighbors": byzantine_accepted_neighbors,
            "avg_raw_distance": sum(raw_d_list) / max(len(raw_d_list), 1),
            "avg_apt_score": sum(apt_score_list) / max(len(apt_score_list), 1),
            "avg_apt_score_honest": self._safe_mean(apt_score_honest_list),
            "avg_apt_score_byzantine": self._safe_mean(apt_score_byzantine_list),
            "avg_cos_distance_honest": self._safe_mean(cos_distance_honest_list),
            "avg_cos_distance_byzantine": self._safe_mean(cos_distance_byzantine_list),
            "avg_norm_distance_honest": self._safe_mean(norm_distance_honest_list),
            "avg_norm_distance_byzantine": self._safe_mean(norm_distance_byzantine_list),
            "avg_historical_mad_score": sum(historical_mad_score_list) / max(len(historical_mad_score_list), 1),
            "avg_instant_score": sum(instant_score_list) / max(len(instant_score_list), 1),
            "apt_q_low": q_low,
            "apt_q_high": q_high,
        }
        return filtered, stats

    def filter_neighbors_by_secure_mpc_apt(self, round_idx, node_id, agg_weights, secure_round):
        """Compute aggregation weights from four-view, committee-validated APT scores."""

        filtered = {node_id: agg_weights[node_id]}
        scores, score_stats = self.secure_protocol.compute_mpc_friendly_apt_scores(
            receiver_node=node_id,
            candidate_nodes=agg_weights.keys(),
            report=secure_round
        )
        receiver_material = secure_round.receiver_materials.get(node_id)
        valid_nodes = (
            receiver_material.valid_nodes
            if receiver_material is not None
            else set()
        )
        apt_enabled = bool(score_stats.get("apt_enabled", False))
        metrics_by_node = score_stats.get("metrics_by_node", {})

        total_neighbors = max(len(agg_weights) - 1, 0)
        rejected_neighbors = 0
        suspicious_neighbors = 0
        accepted_neighbors = 0
        instant_score_list = []
        apt_score_honest_list = []
        apt_score_byzantine_list = []
        honest_total_neighbors = 0
        byzantine_total_neighbors = 0
        honest_rejected_neighbors = 0
        byzantine_accepted_neighbors = 0
        raw_distance_list = []
        cos_distance_honest_list = []
        cos_distance_byzantine_list = []
        norm_distance_honest_list = []
        norm_distance_byzantine_list = []
        historical_mad_score_list = []

        for src_node, base_w in agg_weights.items():
            if src_node == node_id:
                continue

            is_byzantine = src_node in self.malicious_nodes
            if is_byzantine:
                byzantine_total_neighbors += 1
            else:
                honest_total_neighbors += 1

            if (
                src_node not in valid_nodes
                and getattr(self.config, "secure_strict_validation", True)
            ):
                rejected_neighbors += 1
                if not is_byzantine:
                    honest_rejected_neighbors += 1
                continue

            secure_score = scores.get(src_node, 0.0)
            metrics = metrics_by_node.get(src_node, {})
            raw_distance = float(metrics.get("raw_distance", 0.0))
            cosine_distance = float(metrics.get("cosine_dist", 0.0))
            norm_distance = float(metrics.get("norm_ratio_dist", 0.0))
            momentum_distance = float(metrics.get("momentum_dist", 0.0))
            historical_score = math.exp(-momentum_distance)

            raw_distance_list.append(raw_distance)
            historical_mad_score_list.append(historical_score)
            instant_score_list.append(secure_score)
            if is_byzantine:
                apt_score_byzantine_list.append(secure_score)
                cos_distance_byzantine_list.append(cosine_distance)
                norm_distance_byzantine_list.append(norm_distance)
            else:
                apt_score_honest_list.append(secure_score)
                cos_distance_honest_list.append(cosine_distance)
                norm_distance_honest_list.append(norm_distance)

            if self.config.use_trust_score:
                trust = self.update_trust_score(
                    node_id=node_id,
                    src_node=src_node,
                    instant_score=secure_score
                )
            else:
                trust = 1.0

            if (
                apt_enabled
                and secure_score <= self.config.apt_hard_reject_score
            ):
                rejected_neighbors += 1
                if not is_byzantine:
                    honest_rejected_neighbors += 1
                continue

            if secure_score >= (2.0 / 3.0):
                accepted_neighbors += 1
                if is_byzantine:
                    byzantine_accepted_neighbors += 1
            else:
                suspicious_neighbors += 1

            eff_w = base_w
            if apt_enabled:
                eff_w *= max(secure_score, 1e-6) ** self.config.apt_weight_power

            if self.config.use_trust_score:
                eff_w *= max(trust, 1e-6) ** self.config.trust_weight_power

            if eff_w > 0.0:
                filtered[src_node] = eff_w

        total_w = sum(filtered.values())
        if total_w <= 0.0:
            filtered = {node_id: 1.0}
        else:
            filtered = {k: v / total_w for k, v in filtered.items()}

        self_weight = filtered.get(node_id, 0.0)
        neighbor_weights = {
            key: value
            for key, value in filtered.items()
            if key != node_id
        }
        filtered, capped_neighbors = self._cap_neighbor_weights(
            node_id=node_id,
            self_weight=self_weight,
            neighbor_weights=neighbor_weights,
        )

        stats = {
            "total_neighbors": total_neighbors,
            "rejected_neighbors": rejected_neighbors,
            "suspicious_neighbors": suspicious_neighbors,
            "accepted_neighbors": accepted_neighbors,
            "capped_neighbors": capped_neighbors,
            "honest_total_neighbors": honest_total_neighbors,
            "byzantine_total_neighbors": byzantine_total_neighbors,
            "honest_rejected_neighbors": honest_rejected_neighbors,
            "byzantine_accepted_neighbors": byzantine_accepted_neighbors,
            "avg_raw_distance": (
                sum(raw_distance_list) / max(len(raw_distance_list), 1)
            ),
            "avg_apt_score": score_stats["avg_apt_score"],
            "avg_apt_score_honest": self._safe_mean(apt_score_honest_list),
            "avg_apt_score_byzantine": self._safe_mean(apt_score_byzantine_list),
            "avg_cos_distance_honest": self._safe_mean(
                cos_distance_honest_list
            ),
            "avg_cos_distance_byzantine": self._safe_mean(
                cos_distance_byzantine_list
            ),
            "avg_norm_distance_honest": self._safe_mean(
                norm_distance_honest_list
            ),
            "avg_norm_distance_byzantine": self._safe_mean(
                norm_distance_byzantine_list
            ),
            "avg_historical_mad_score": (
                sum(historical_mad_score_list)
                / max(len(historical_mad_score_list), 1)
            ),
            "avg_instant_score": sum(instant_score_list) / max(len(instant_score_list), 1),
            "apt_q_low": score_stats.get("apt_q_low", 0.0),
            "apt_q_high": score_stats.get("apt_q_high", 0.0),
        }
        return filtered, stats

    def attack_state_dict(self, honest_state, ref_state):
        attacked = apply_attacks_to_uploaded_states(
            uploaded_states={0: honest_state},
            pre_train_states={0: ref_state},
            malicious_nodes={0},
            config=self.config,
            round_idx=0,
        )
        return attacked[0]

    def export_node_metrics_to_csv(self, round_idx, node_metrics, csv_path="node_metrics_final.csv"):
        fieldnames = ["round", "node_id", "num_test_samples", "local_test_loss", "local_test_acc"]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in node_metrics:
                writer.writerow({
                    "round": round_idx,
                    "node_id": row["node_id"],
                    "num_test_samples": row["num_test_samples"],
                    "local_test_loss": row["local_test_loss"],
                    "local_test_acc": row["local_test_acc"],
                })

    def train(self):
        history = []

        out_neighbors, in_neighbors = generate_dynamic_directed_graph(
            num_nodes=self.num_nodes,
            min_out_degree=self.config.min_out_degree,
            max_out_degree=self.config.max_out_degree,
            seed=self.config.seed
        )

        for r in range(self.config.rounds):
            round_start_time = time.perf_counter()
            # 1. Refresh the directed communication graph.
            if r % 5 == 0:
                out_neighbors, in_neighbors = generate_dynamic_directed_graph(
                    num_nodes=self.num_nodes,
                    min_out_degree=self.config.min_out_degree,
                    max_out_degree=self.config.max_out_degree,
                    seed=self.config.seed + r
                )
            comm_mb_per_round = self.estimate_comm_mb_per_round(in_neighbors)
            secure_comm_stats = self.estimate_secure_comm_mb_per_round(in_neighbors)
            effective_comm_mb_per_round = (
                secure_comm_stats["secure_total_comm_mb_per_round"]
                if self.secure_protocol is not None else comm_mb_per_round
            )

            # 2. Select nodes participating in local training.
            active_nodes = self.sample_participating_nodes(r)

            # 3. Capture round-start states for update construction.
            pre_train_states = {
                node_id: self.nodes[node_id].get_state_dict()
                for node_id in range(self.num_nodes)
            }

            # 4. Run local training.
            local_losses = {}
            for node_id in active_nodes:
                loss = self.nodes[node_id].local_train(epochs=3)
                local_losses[node_id] = loss

            # 5. Collect uploaded states; inactive nodes retain their prior state.
            uploaded_states = {
                node_id: self.nodes[node_id].get_state_dict()
                for node_id in range(self.num_nodes)
            }

            # 6. Apply configured attacks to selected uploads.
            active_attack_names = []
            if self.config.use_malicious and (r + 1) >= self.config.attack_start_round:
                target_nodes = self.malicious_nodes
                if self.config.malicious_only_when_active:
                    target_nodes = target_nodes.intersection(set(active_nodes))

                active_attack_names = get_enabled_attack_names(self.config)
                uploaded_states = apply_attacks_to_uploaded_states(
                    uploaded_states=uploaded_states,
                    pre_train_states=pre_train_states,
                    malicious_nodes=target_nodes,
                    config=self.config,
                    round_idx=r,
                )

            clipped_update_count = 0
            update_norms_before_clip = []
            mad_clip_bound = 0.0
            mad_clip_median = 0.0
            mad_clip_mad = 0.0
            if getattr(self.config, "use_mad_clipping", False):
                # Derive the update-norm bound from the round median and MAD.
                (
                    uploaded_states,
                    update_norms_before_clip,
                    clipped_update_count,
                    mad_clip_bound,
                    mad_clip_median,
                    mad_clip_mad,
                ) = self.clip_updates_by_adaptive_mad(
                    uploaded_states=uploaded_states,
                    pre_train_states=pre_train_states,
                )
            elif float(getattr(self.config, "max_update_norm", 0.0)) > 0.0:
                # Clip transmitted updates before distance scoring and aggregation.
                clipped_uploaded_states = {}
                for node_id in range(self.num_nodes):
                    clipped_state, update_norm, was_clipped = self.clip_update_state_dict(
                        uploaded_state=uploaded_states[node_id],
                        ref_state=pre_train_states[node_id],
                    )
                    clipped_uploaded_states[node_id] = clipped_state
                    update_norms_before_clip.append(update_norm)
                    if was_clipped:
                        clipped_update_count += 1
                uploaded_states = clipped_uploaded_states

            # 7. Build round updates and refresh momentum buffers.
            current_updates = {
                node_id: self.diff_state_dict(uploaded_states[node_id], pre_train_states[node_id])
                for node_id in range(self.num_nodes)
            }
            self.update_momentum_buffers(current_updates)

            # Validate the post-attack, post-clipping updates supplied to aggregation.
            secure_round = None
            round_secure_stats = {
                "secure_valid_nodes": 0,
                "secure_invalid_nodes": 0,
                "secure_signature_failures": 0,
                "secure_vss_failures": 0,
                "secure_zkp_failures": 0,
                "secure_binding_failures": 0,
                "secure_committee_successes": 0,
                "secure_committee_aborts": 0,
            }
            if self.secure_protocol is not None:
                secure_round = self.secure_protocol.prepare_round(
                    round_idx=r,
                    uploaded_states=uploaded_states,
                    pre_train_states=pre_train_states,
                    in_neighbors=in_neighbors
                )
                round_secure_stats.update(secure_round.stats)

            # 8. Compute receiver-specific robust weights and aggregate updates.
            new_states = {}

            round_total_neighbors = 0
            round_rejected = 0
            round_suspicious = 0
            round_accepted = 0
            # Number of neighbors whose normalized weight reached the configured cap.
            round_capped_neighbors = 0
            # Accumulators for malicious-versus-honest neighbor weight statistics.
            round_malicious_weight_sum = 0.0
            round_honest_weight_sum = 0.0
            round_malicious_neighbor_count = 0
            round_honest_neighbor_count = 0
            round_avg_raw_distance = []
            round_avg_apt_score = []
            round_avg_historical_mad_score = []
            round_avg_instant_score = []
            honest_robust_accumulator = self._make_empty_robust_accumulator()

            for node_id in range(self.num_nodes):
                agg_weights = build_aggregation_weights(
                    node_id=node_id,
                    in_neighbors=in_neighbors,
                    self_weight=self.config.self_weight
                )

                if (
                    self.secure_protocol is not None
                    and getattr(self.config, "secure_use_mpc_friendly_apt", True)
                ):
                    # Score neighbors from committee-validated projected summaries.
                    filtered_weights, robust_stats = self.filter_neighbors_by_secure_mpc_apt(
                        round_idx=r,
                        node_id=node_id,
                        agg_weights=agg_weights,
                        secure_round=secure_round
                    )
                else:
                    # Use the full-update Trust + APT path when secure scoring is disabled.
                    filtered_weights, robust_stats = self.filter_neighbors_by_trust_apt(
                        round_idx=r,
                        node_id=node_id,
                        agg_weights=agg_weights,
                        uploaded_states=uploaded_states,
                        pre_train_states=pre_train_states
                    )

                round_total_neighbors += robust_stats["total_neighbors"]
                round_rejected += robust_stats["rejected_neighbors"]
                round_suspicious += robust_stats["suspicious_neighbors"]
                round_accepted += robust_stats["accepted_neighbors"]
                round_capped_neighbors += robust_stats.get("capped_neighbors", 0)
                round_avg_raw_distance.append(robust_stats["avg_raw_distance"])
                round_avg_apt_score.append(robust_stats["avg_apt_score"])
                round_avg_historical_mad_score.append(robust_stats.get("avg_historical_mad_score", 1.0))
                round_avg_instant_score.append(robust_stats["avg_instant_score"])

                # Measure final weight assigned to malicious and honest neighbors.
                weight_stats = self._collect_malicious_weight_stats(node_id, filtered_weights)
                round_malicious_weight_sum += weight_stats["malicious_weight_sum"]
                round_honest_weight_sum += weight_stats["honest_weight_sum"]
                round_malicious_neighbor_count += weight_stats["malicious_neighbor_count"]
                round_honest_neighbor_count += weight_stats["honest_neighbor_count"]

                if node_id not in self.malicious_nodes:
                    self._add_robust_stats(
                        accumulator=honest_robust_accumulator,
                        robust_stats=robust_stats,
                        weight_stats=weight_stats,
                    )

                update_dicts = []
                sender_nodes = []
                weights = []
                for src_node, w in filtered_weights.items():
                    update_dicts.append(current_updates[src_node])
                    sender_nodes.append(src_node)
                    weights.append(w)

                if self.secure_protocol is not None:
                    # Reconstruct the committee aggregate, then apply it to the round-start state.
                    aggregated_update, committee_status = (
                        self.secure_protocol.try_secure_aggregate_update_dicts(
                            update_dicts=update_dicts,
                            sender_nodes=sender_nodes,
                            weights=weights,
                            round_idx=r,
                            receiver_node=node_id,
                            valid_nodes=(
                                secure_round
                                .receiver_materials[node_id]
                                .valid_nodes
                            ),
                            round_report=secure_round,
                        )
                    )
                    if committee_status["success"]:
                        new_states[node_id] = self.apply_aggregated_update(
                            base_state=pre_train_states[node_id],
                            aggregated_update=aggregated_update,
                            step_size=getattr(
                                self.config,
                                "secure_global_update_lr",
                                1.0,
                            ),
                        )
                        round_secure_stats["secure_committee_successes"] += 1
                    else:
                        # Keep the receiver's uploaded state when committee validation aborts.
                        new_states[node_id] = clone_state_dict(
                            uploaded_states[node_id]
                        )
                        round_secure_stats["secure_committee_aborts"] += 1
                else:
                    state_dicts = [
                        uploaded_states[src_node]
                        for src_node in sender_nodes
                    ]
                    new_states[node_id] = self.aggregate_state_dicts(
                        state_dicts,
                        weights,
                    )

            # 9. Install the aggregated states.
            for node_id in range(self.num_nodes):
                self.nodes[node_id].load_state_dict(new_states[node_id])

            time_per_round = time.perf_counter() - round_start_time
            self.elapsed_time += time_per_round
            self.cumulative_comm_mb += effective_comm_mb_per_round

            # 10. Evaluate and record round metrics.
            if (r + 1) % self.config.eval_every == 0:
                honest_nodes = [
                    node_id for node_id in range(self.num_nodes)
                    if node_id not in self.malicious_nodes
                ]
                honest_node_set = set(honest_nodes)
                honest_active_nodes = [node_id for node_id in active_nodes if node_id in honest_node_set]

                global_losses, global_accs = [], []
                for node_id in range(self.num_nodes):
                    loss, acc = self.nodes[node_id].evaluate(self.test_loader)
                    global_losses.append(loss)
                    global_accs.append(acc)

                node_metrics = []
                node_local_test_losses, node_local_test_accs = [], []

                for node_id in range(self.num_nodes):
                    local_test_loader = self.node_test_loaders[node_id]
                    local_loss, local_acc = self.nodes[node_id].evaluate(local_test_loader)

                    num_test_samples = len(local_test_loader.dataset)

                    node_metrics.append({
                        "node_id": node_id,
                        "num_test_samples": num_test_samples,
                        "local_test_loss": local_loss,
                        "local_test_acc": local_acc,
                    })

                    node_local_test_losses.append(local_loss)
                    node_local_test_accs.append(local_acc)

                trust_values = []
                for i in range(self.num_nodes):
                    for j in self.trust_scores[i]:
                        trust_values.append(self.trust_scores[i][j])

                avg_trust = sum(trust_values) / max(len(trust_values), 1)
                min_trust = min(trust_values) if trust_values else 0.0
                max_trust = max(trust_values) if trust_values else 0.0

                honest_receiver_trust_values = []
                honest_to_honest_trust_values = []
                honest_to_malicious_trust_values = []
                for i in honest_nodes:
                    for j, trust_value in self.trust_scores[i].items():
                        honest_receiver_trust_values.append(trust_value)
                        if j in self.malicious_nodes:
                            honest_to_malicious_trust_values.append(trust_value)
                        else:
                            honest_to_honest_trust_values.append(trust_value)

                honest_receiver_avg_trust = (
                    sum(honest_receiver_trust_values) / max(len(honest_receiver_trust_values), 1)
                )
                honest_to_honest_avg_trust = (
                    sum(honest_to_honest_trust_values) / max(len(honest_to_honest_trust_values), 1)
                )
                honest_to_malicious_avg_trust = (
                    sum(honest_to_malicious_trust_values) / max(len(honest_to_malicious_trust_values), 1)
                )

                avg_train_loss = sum(local_losses.values()) / max(len(local_losses), 1)
                honest_train_losses = [
                    local_losses[node_id]
                    for node_id in honest_active_nodes
                    if node_id in local_losses
                ]
                honest_avg_train_loss = sum(honest_train_losses) / max(len(honest_train_losses), 1)

                avg_test_loss = sum(global_losses) / len(global_losses)
                avg_test_acc = sum(global_accs) / len(global_accs)
                max_test_acc = max(global_accs)
                min_test_acc = min(global_accs)
                std_test_acc = torch.tensor(global_accs, dtype=torch.float32).std(unbiased=False).item()

                honest_global_losses = [global_losses[node_id] for node_id in honest_nodes]
                honest_global_accs = [global_accs[node_id] for node_id in honest_nodes]
                honest_avg_test_loss = sum(honest_global_losses) / max(len(honest_global_losses), 1)
                honest_avg_test_acc = sum(honest_global_accs) / max(len(honest_global_accs), 1)
                honest_max_test_acc = max(honest_global_accs) if honest_global_accs else 0.0
                honest_min_test_acc = min(honest_global_accs) if honest_global_accs else 0.0
                honest_std_test_acc = (
                    torch.tensor(honest_global_accs, dtype=torch.float32).std(unbiased=False).item()
                    if honest_global_accs else 0.0
                )

                avg_local_test_loss = sum(node_local_test_losses) / len(node_local_test_losses)
                avg_local_test_acc = sum(node_local_test_accs) / len(node_local_test_accs)
                honest_local_test_losses = [node_local_test_losses[node_id] for node_id in honest_nodes]
                honest_local_test_accs = [node_local_test_accs[node_id] for node_id in honest_nodes]
                honest_avg_local_test_loss = (
                    sum(honest_local_test_losses) / max(len(honest_local_test_losses), 1)
                )
                honest_avg_local_test_acc = (
                    sum(honest_local_test_accs) / max(len(honest_local_test_accs), 1)
                )
                honest_worst_local_acc = min(honest_local_test_accs) if honest_local_test_accs else 0.0
                honest_acc_std = (
                    torch.tensor(honest_local_test_accs, dtype=torch.float32).std(unbiased=False).item()
                    if honest_local_test_accs else 0.0
                )
                per_node_local_acc = {
                    node_id: node_local_test_accs[node_id]
                    for node_id in range(self.num_nodes)
                }

                avg_raw_distance = sum(round_avg_raw_distance) / max(len(round_avg_raw_distance), 1)
                avg_apt_score = sum(round_avg_apt_score) / max(len(round_avg_apt_score), 1)
                avg_historical_mad_score = (
                    sum(round_avg_historical_mad_score) / max(len(round_avg_historical_mad_score), 1)
                )
                avg_instant_score = sum(round_avg_instant_score) / max(len(round_avg_instant_score), 1)
                avg_update_norm_before_clip = (
                    sum(update_norms_before_clip) / max(len(update_norms_before_clip), 1)
                )
                max_update_norm_before_clip = (
                    max(update_norms_before_clip) if update_norms_before_clip else 0.0
                )
                total_neighbor_weight = round_malicious_weight_sum + round_honest_weight_sum
                # Fraction of neighbor aggregation mass assigned to malicious nodes.
                malicious_weight_share = (
                    round_malicious_weight_sum / total_neighbor_weight
                    if total_neighbor_weight > 0.0 else 0.0
                )
                avg_weight_malicious = (
                    round_malicious_weight_sum / max(round_malicious_neighbor_count, 1)
                )
                avg_weight_honest = (
                    round_honest_weight_sum / max(round_honest_neighbor_count, 1)
                )
                malicious_to_honest_weight_ratio = (
                    avg_weight_malicious / max(avg_weight_honest, 1e-12)
                    if round_honest_neighbor_count > 0 else 0.0
                )
                honest_robust_summary = self._summarize_robust_accumulator(honest_robust_accumulator)
                target_metric = getattr(self.config, "target_metric", "honest_mean_local_acc")
                if target_metric == "honest_worst_local_acc":
                    target_value = honest_worst_local_acc
                else:
                    target_value = honest_avg_local_test_acc
                if self.rounds_to_target is None and target_value >= getattr(self.config, "target_acc", 0.0):
                    self.rounds_to_target = r + 1
                    self.total_time_to_target = self.elapsed_time
                    self.total_comm_mb_to_target = self.cumulative_comm_mb

                record = {
                    "round": r + 1,
                    "per_node_local_acc": per_node_local_acc,
                    "train_loss": avg_train_loss,
                    "test_loss": avg_test_loss,
                    "test_acc": avg_test_acc,
                    "max_test_acc": max_test_acc,
                    "min_test_acc": min_test_acc,
                    "std_test_acc": std_test_acc,
                    "honest_train_loss": honest_avg_train_loss,
                    "honest_test_loss": honest_avg_test_loss,
                    "honest_test_acc": honest_avg_test_acc,
                    "honest_max_test_acc": honest_max_test_acc,
                    "honest_min_test_acc": honest_min_test_acc,
                    "honest_std_test_acc": honest_std_test_acc,
                    "avg_local_test_loss": avg_local_test_loss,
                    "avg_local_test_acc": avg_local_test_acc,
                    "honest_avg_local_test_loss": honest_avg_local_test_loss,
                    "honest_avg_local_test_acc": honest_avg_local_test_acc,
                    "honest_mean_local_acc": honest_avg_local_test_acc,
                    "honest_worst_local_acc": honest_worst_local_acc,
                    "honest_acc_std": honest_acc_std,
                    "avg_in_degree": sum(len(in_neighbors[i]) for i in range(self.num_nodes)) / self.num_nodes,
                    "avg_out_degree": sum(len(out_neighbors[i]) for i in range(self.num_nodes)) / self.num_nodes,
                    "filtered_avg_in_degree": (
                        honest_robust_summary["accepted_neighbors"]
                        + honest_robust_summary["suspicious_neighbors"]
                    ) / max(len(honest_nodes), 1),
                    "avg_trust": avg_trust,
                    "min_trust": min_trust,
                    "max_trust": max_trust,
                    "honest_receiver_avg_trust": honest_receiver_avg_trust,
                    "honest_to_honest_avg_trust": honest_to_honest_avg_trust,
                    "honest_to_malicious_avg_trust": honest_to_malicious_avg_trust,
                    "honest_avg_trust": honest_to_honest_avg_trust,
                    "byzantine_avg_trust": honest_to_malicious_avg_trust,
                    "avg_raw_distance": avg_raw_distance,
                    "avg_apt_score": avg_apt_score,
                    "avg_apt_score_honest": honest_robust_summary["avg_apt_score_honest"],
                    "avg_apt_score_byzantine": honest_robust_summary["avg_apt_score_byzantine"],
                    "avg_cos_distance_honest": honest_robust_summary["avg_cos_distance_honest"],
                    "avg_cos_distance_byzantine": honest_robust_summary["avg_cos_distance_byzantine"],
                    "avg_norm_distance_honest": honest_robust_summary["avg_norm_distance_honest"],
                    "avg_norm_distance_byzantine": honest_robust_summary["avg_norm_distance_byzantine"],
                    "avg_historical_mad_score": avg_historical_mad_score,
                    "avg_instant_score": avg_instant_score,
                    "clipped_update_count": clipped_update_count,
                    "avg_update_norm_before_clip": avg_update_norm_before_clip,
                    "max_update_norm_before_clip": max_update_norm_before_clip,
                    "mad_clip_bound": mad_clip_bound,
                    "mad_clip_median": mad_clip_median,
                    "mad_clip_mad": mad_clip_mad,
                    "capped_neighbors": round_capped_neighbors,
                    "malicious_weight_share": malicious_weight_share,
                    "malicious_effective_weight": honest_robust_summary["malicious_weight_share"],
                    "rejection_rate": honest_robust_summary["rejection_rate"],
                    "false_rejection_rate": honest_robust_summary["false_rejection_rate"],
                    "false_acceptance_rate": honest_robust_summary["false_acceptance_rate"],
                    "avg_weight_malicious": avg_weight_malicious,
                    "avg_weight_honest": avg_weight_honest,
                    "malicious_to_honest_weight_ratio": malicious_to_honest_weight_ratio,
                    "honest_malicious_weight_share": honest_robust_summary["malicious_weight_share"],
                    "honest_avg_weight_malicious": honest_robust_summary["avg_weight_malicious"],
                    "honest_avg_weight_honest": honest_robust_summary["avg_weight_honest"],
                    "honest_malicious_to_honest_weight_ratio": honest_robust_summary["malicious_to_honest_weight_ratio"],
                    "secure_valid_nodes": round_secure_stats["secure_valid_nodes"],
                    "secure_invalid_nodes": round_secure_stats["secure_invalid_nodes"],
                    "secure_signature_failures": round_secure_stats["secure_signature_failures"],
                    "secure_vss_failures": round_secure_stats["secure_vss_failures"],
                    "secure_zkp_failures": round_secure_stats["secure_zkp_failures"],
                    "secure_binding_failures": round_secure_stats["secure_binding_failures"],
                    "secure_committee_successes": round_secure_stats["secure_committee_successes"],
                    "secure_committee_aborts": round_secure_stats["secure_committee_aborts"],
                    "rounds_to_target": self.rounds_to_target if self.rounds_to_target is not None else -1,
                    "time_per_round": time_per_round,
                    "total_time_to_target": self.total_time_to_target if self.total_time_to_target is not None else -1,
                    "comm_mb_per_round": comm_mb_per_round,
                    "effective_comm_mb_per_round": effective_comm_mb_per_round,
                    "secure_total_comm_mb_per_round": secure_comm_stats["secure_total_comm_mb_per_round"],
                    "secure_share_model_mb_per_round": secure_comm_stats["secure_share_model_mb_per_round"],
                    "secure_metadata_mb_per_round": secure_comm_stats["secure_metadata_mb_per_round"],
                    "secure_comm_multiplier": secure_comm_stats["secure_comm_multiplier"],
                    "total_comm_mb_to_target": self.total_comm_mb_to_target if self.total_comm_mb_to_target is not None else -1,
                }
                history.append(record)

                if self.config.use_wandb and self.config.wandb_mode != "disabled":
                    if wandb is None:
                        raise RuntimeError(
                            "Weights & Biases logging is enabled, but wandb "
                            "is not installed."
                        )
                    log_dict = {
                        "round": r + 1,
                        "train/loss": avg_train_loss,
                        "test/loss": avg_test_loss,
                        "test/acc_mean": avg_test_acc,
                        "test/acc_max": max_test_acc,
                        "test/acc_min": min_test_acc,
                        "test/acc_std": std_test_acc,
                        "test/local_loss_mean": avg_local_test_loss,
                        "test/local_acc_mean": avg_local_test_acc,
                        "honest/train_loss": honest_avg_train_loss,
                        "honest/test_loss": honest_avg_test_loss,
                        "honest/test_acc_mean": honest_avg_test_acc,
                        "honest/test_acc_max": honest_max_test_acc,
                        "honest/test_acc_min": honest_min_test_acc,
                        "honest/test_acc_std": honest_std_test_acc,
                        "honest/local_test_loss_mean": honest_avg_local_test_loss,
                        "honest/local_test_acc_mean": honest_avg_local_test_acc,
                        "honest_mean_local_acc": honest_avg_local_test_acc,
                        "honest_worst_local_acc": honest_worst_local_acc,
                        "honest_acc_std": honest_acc_std,
                        "honest/num_nodes": len(honest_nodes),
                        "honest/num_active_nodes": len(honest_active_nodes),
                        "graph/avg_in_degree": record["avg_in_degree"],
                        "graph/avg_out_degree": record["avg_out_degree"],
                        "filtered_avg_in_degree": record["filtered_avg_in_degree"],
                        "system/num_active_nodes": len(active_nodes),
                        "rounds_to_target": record["rounds_to_target"],
                        "time_per_round": time_per_round,
                        "total_time_to_target": record["total_time_to_target"],
                        "comm_mb_per_round": comm_mb_per_round,
                        "effective_comm_mb_per_round": effective_comm_mb_per_round,
                        "secure/total_comm_mb_per_round": secure_comm_stats["secure_total_comm_mb_per_round"],
                        "secure/share_model_mb_per_round": secure_comm_stats["secure_share_model_mb_per_round"],
                        "secure/metadata_mb_per_round": secure_comm_stats["secure_metadata_mb_per_round"],
                        "secure/comm_multiplier": secure_comm_stats["secure_comm_multiplier"],
                        "total_comm_mb_to_target": record["total_comm_mb_to_target"],
                        "trust/avg": avg_trust,
                        "trust/min": min_trust,
                        "trust/max": max_trust,
                        "honest_avg_trust": honest_to_honest_avg_trust,
                        "byzantine_avg_trust": honest_to_malicious_avg_trust,
                        "honest_trust/receiver_avg": honest_receiver_avg_trust,
                        "honest_trust/to_honest_avg": honest_to_honest_avg_trust,
                        "honest_trust/to_malicious_avg": honest_to_malicious_avg_trust,
                        "robust/avg_raw_distance": avg_raw_distance,
                        "robust/avg_apt_score": avg_apt_score,
                        "avg_apt_score_honest": record["avg_apt_score_honest"],
                        "avg_apt_score_byzantine": record["avg_apt_score_byzantine"],
                        "avg_cos_distance_honest": record["avg_cos_distance_honest"],
                        "avg_cos_distance_byzantine": record["avg_cos_distance_byzantine"],
                        "avg_norm_distance_honest": record["avg_norm_distance_honest"],
                        "avg_norm_distance_byzantine": record["avg_norm_distance_byzantine"],
                        "robust/avg_historical_mad_score": avg_historical_mad_score,
                        "robust/avg_instant_score": avg_instant_score,
                        "robust/total_neighbors": round_total_neighbors,
                        "robust/rejected_neighbors": round_rejected,
                        "robust/suspicious_neighbors": round_suspicious,
                        "robust/accepted_neighbors": round_accepted,
                        "rejection_rate": record["rejection_rate"],
                        "false_rejection_rate": record["false_rejection_rate"],
                        "false_acceptance_rate": record["false_acceptance_rate"],
                        "robust/capped_neighbors": round_capped_neighbors,
                        "robust/clipped_update_count": clipped_update_count,
                        "robust/avg_update_norm_before_clip": avg_update_norm_before_clip,
                        "robust/max_update_norm_before_clip": max_update_norm_before_clip,
                        "robust/mad_clip_bound": mad_clip_bound,
                        "robust/mad_clip_median": mad_clip_median,
                        "robust/mad_clip_mad": mad_clip_mad,
                        "robust/aggregation_rule_weighted_trimmed_mean": int(
                            str(getattr(self.config, "aggregation_rule", "weighted_average")).lower()
                            == "weighted_trimmed_mean"
                        ),
                        "honest_robust/avg_raw_distance": honest_robust_summary["avg_raw_distance"],
                        "honest_robust/avg_apt_score": honest_robust_summary["avg_apt_score"],
                        "honest_robust/avg_historical_mad_score": honest_robust_summary["avg_historical_mad_score"],
                        "honest_robust/avg_instant_score": honest_robust_summary["avg_instant_score"],
                        "honest_robust/total_neighbors": honest_robust_summary["total_neighbors"],
                        "honest_robust/rejected_neighbors": honest_robust_summary["rejected_neighbors"],
                        "honest_robust/suspicious_neighbors": honest_robust_summary["suspicious_neighbors"],
                        "honest_robust/accepted_neighbors": honest_robust_summary["accepted_neighbors"],
                        "honest_robust/capped_neighbors": honest_robust_summary["capped_neighbors"],
                        "attack/malicious_weight_share": malicious_weight_share,
                        "malicious_effective_weight": record["malicious_effective_weight"],
                        "attack/avg_weight_malicious": avg_weight_malicious,
                        "attack/avg_weight_honest": avg_weight_honest,
                        "attack/malicious_to_honest_weight_ratio": malicious_to_honest_weight_ratio,
                        "honest_attack/malicious_weight_share": honest_robust_summary["malicious_weight_share"],
                        "honest_attack/avg_weight_malicious": honest_robust_summary["avg_weight_malicious"],
                        "honest_attack/avg_weight_honest": honest_robust_summary["avg_weight_honest"],
                        "honest_attack/malicious_to_honest_weight_ratio": honest_robust_summary["malicious_to_honest_weight_ratio"],
                    }

                    for node_id, local_acc in per_node_local_acc.items():
                        log_dict[f"per_node_local_acc/node_{node_id}"] = local_acc

                    if self.secure_protocol is not None:
                        log_dict.update({
                            "secure/enabled": 1,
                            "secure/valid_nodes": round_secure_stats["secure_valid_nodes"],
                            "secure/invalid_nodes": round_secure_stats["secure_invalid_nodes"],
                            "secure/signature_failures": round_secure_stats["secure_signature_failures"],
                            "secure/vss_failures": round_secure_stats["secure_vss_failures"],
                            "secure/zkp_failures": round_secure_stats["secure_zkp_failures"],
                            "secure/binding_failures": round_secure_stats["secure_binding_failures"],
                            "secure/committee_successes": round_secure_stats["secure_committee_successes"],
                            "secure/committee_aborts": round_secure_stats["secure_committee_aborts"],
                        })
                    else:
                        log_dict.update({"secure/enabled": 0})

                    if self.config.use_malicious:
                        log_dict.update({
                            "attack/enabled": 1,
                            "attack/types": ",".join(active_attack_names),
                            "attack/malicious_ratio": self.config.malicious_ratio,
                            "attack/malicious_count": len(self.malicious_nodes),
                        })
                    else:
                        log_dict.update({
                            "attack/enabled": 0,
                            "attack/malicious_ratio": 0.0,
                            "attack/malicious_count": 0,
                        })

                    wandb.log(log_dict, step=r + 1)

                msg = (
                    f"[Round {r + 1:03d}] "
                    f"train_loss={avg_train_loss:.4f} "
                    f"test_loss={avg_test_loss:.4f} "
                    f"test_acc={avg_test_acc:.4f} "
                    f"max_acc={max_test_acc:.4f} "
                    f"min_acc={min_test_acc:.4f} "
                    f"std_acc={std_test_acc:.4f} "
                    f"avg_in={record['avg_in_degree']:.2f} "
                    f"avg_out={record['avg_out_degree']:.2f} "
                    f"avg_local_test_loss={avg_local_test_loss:.4f} "
                    f"avg_local_test_acc={avg_local_test_acc:.4f}"
                )

                if self.config.use_trust_score and self.config.log_trust:
                    msg += (
                        f" | avg_trust={avg_trust:.4f} "
                        f"min_trust={min_trust:.4f} "
                        f"max_trust={max_trust:.4f}"
                    )

                if self.config.use_apt and self.config.log_apt:
                    msg += (
                        f" | avg_raw_d={avg_raw_distance:.4f} "
                        f"avg_apt={avg_apt_score:.4f} "
                        f"avg_inst={avg_instant_score:.4f} "
                        f"rej={round_rejected} "
                        f"susp={round_suspicious} "
                        f"acc={round_accepted}"
                    )

                if self.secure_protocol is not None and self.config.log_secure_protocol:
                    msg += (
                        f" | secure_valid={round_secure_stats['secure_valid_nodes']} "
                        f"secure_invalid={round_secure_stats['secure_invalid_nodes']} "
                        f"sig_fail={round_secure_stats['secure_signature_failures']} "
                        f"vss_fail={round_secure_stats['secure_vss_failures']} "
                        f"zkp_fail={round_secure_stats['secure_zkp_failures']} "
                        f"binding_fail={round_secure_stats['secure_binding_failures']} "
                        f"committee_ok={round_secure_stats['secure_committee_successes']} "
                        f"committee_abort={round_secure_stats['secure_committee_aborts']}"
                    )

                print(msg)

                honest_msg = (
                    f"[Round {r + 1:03d} Honest] "
                    f"honest_nodes={len(honest_nodes)} "
                    f"honest_active={len(honest_active_nodes)} "
                    f"train_loss={honest_avg_train_loss:.4f} "
                    f"test_loss={honest_avg_test_loss:.4f} "
                    f"test_acc={honest_avg_test_acc:.4f} "
                    f"max_acc={honest_max_test_acc:.4f} "
                    f"min_acc={honest_min_test_acc:.4f} "
                    f"std_acc={honest_std_test_acc:.4f} "
                    f"local_test_loss={honest_avg_local_test_loss:.4f} "
                    f"local_test_acc={honest_avg_local_test_acc:.4f} "
                    f"mal_w_share={honest_robust_summary['malicious_weight_share']:.4f}"
                )

                if self.config.use_trust_score and self.config.log_trust:
                    honest_msg += (
                        f" | honest_recv_trust={honest_receiver_avg_trust:.4f} "
                        f"h2h_trust={honest_to_honest_avg_trust:.4f} "
                        f"h2m_trust={honest_to_malicious_avg_trust:.4f}"
                    )

                if self.config.use_apt and self.config.log_apt:
                    honest_msg += (
                        f" | avg_raw_d={honest_robust_summary['avg_raw_distance']:.4f} "
                        f"avg_apt={honest_robust_summary['avg_apt_score']:.4f} "
                        f"avg_inst={honest_robust_summary['avg_instant_score']:.4f} "
                        f"rej={honest_robust_summary['rejected_neighbors']} "
                        f"susp={honest_robust_summary['suspicious_neighbors']} "
                        f"acc={honest_robust_summary['accepted_neighbors']}"
                    )

                print(honest_msg)

                if (r + 1) == self.config.rounds:
                    output_root = getattr(
                        self.config,
                        "output_root",
                        ".",
                    )
                    run_name = getattr(self.config, "run_name", "")
                    output_dir = os.path.join(output_root, run_name)
                    os.makedirs(output_dir, exist_ok=True)
                    self.export_node_metrics_to_csv(
                        round_idx=r + 1,
                        node_metrics=node_metrics,
                        csv_path=os.path.join(
                            output_dir,
                            "node_metrics_final.csv",
                        )
                    )

        return history
