import hashlib
import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

StateDict = Dict[str, torch.Tensor]
# Summary sharing operates in a prime field for modular interpolation.
FIELD_PRIME = 2_147_483_647
# Fixed-point encoding maps floating-point summaries into the field.
SUMMARY_SCALE = 1_000_000.0


class CommitteeAggregationError(ValueError):
    """Raised when a committee cannot accept or complete an aggregation request."""


@dataclass
class SecureMessage:
    """Protocol message protected by AEAD and an Ed25519 sender signature."""

    meta: Dict
    ciphertext: bytes
    signature: bytes


@dataclass(frozen=True)
class CommitteeAssignment:
    """Committee assigned to one receiver in one training round."""

    committee_id: str
    receiver_node: int
    round_idx: int
    member_ids: Tuple[int, ...]


@dataclass(frozen=True)
class WeightRequest:
    """Signed aggregation-weight request submitted by a receiver."""

    request_id: str
    committee_id: str
    receiver_node: int
    round_idx: int
    request_counter: int
    contributor_nodes: Tuple[int, ...]
    weights: Tuple[Tuple[int, float], ...]
    commitment_root: str
    signature: bytes


@dataclass
class CommitteeAggregationRecord:
    """Stored request and result for idempotent committee aggregation."""

    request: WeightRequest
    aggregated_update: StateDict


@dataclass
class SummaryShare:
    """One Shamir summary share assigned to a committee member."""

    holder_id: int
    x: int
    values: List[int]


@dataclass
class VSSCommitment:
    """Polynomial coefficients used to validate summary shares."""

    coeffs_by_dim: List[List[int]]


@dataclass
class BindingProof:
    """Hashes, norms, and seed binding a projected summary to an update."""

    update_hash: str
    summary_hash: str
    update_norm: float
    summary_norm: float
    projection_seed: int


@dataclass
class NodeSecurityMaterial:
    """Security material generated for one sender-receiver pair in a round."""

    summary: torch.Tensor
    committee_summary: torch.Tensor
    group_summaries: Dict[str, torch.Tensor]
    committee_group_summaries: Dict[str, torch.Tensor]
    historical_summary: torch.Tensor
    trusted_reference_summary: torch.Tensor
    layer_proportions: Dict[str, float]
    summary_shares: List[SummaryShare]
    vss_commitment: VSSCommitment
    binding_proof: BindingProof
    message: SecureMessage
    valid: bool
    failures: List[str] = field(default_factory=list)


@dataclass
class ReceiverCommitteeMaterial:
    """Committee inputs and validation results for one receiver."""

    committee: CommitteeAssignment
    candidate_nodes: Tuple[int, ...]
    commitment_root: str
    projection_seed: int
    share_commitments: Dict[int, Tuple[str, ...]]
    materials: Dict[int, NodeSecurityMaterial]
    valid_nodes: set


@dataclass
class RoundSecurityReport:
    """Validated protocol state consumed by the aggregation stage."""

    round_idx: int
    graph_hash: str
    materials: Dict[int, NodeSecurityMaterial]
    valid_nodes: set
    receiver_materials: Dict[int, ReceiverCommitteeMaterial]
    stats: Dict[str, float]


def _canonical_json(data) -> bytes:
    # Canonical encoding keeps signatures, associated data, and graph hashes stable.
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _is_shared_float_tensor(key: str, tensor: torch.Tensor) -> bool:
    # Aggregate floating-point parameters while keeping batch-normalization buffers local.
    if not tensor.dtype.is_floating_point:
        return False
    if "running_mean" in key or "running_var" in key or "num_batches_tracked" in key:
        return False
    return True


def _state_dict_update(uploaded_state: Mapping[str, torch.Tensor], pre_train_state: Mapping[str, torch.Tensor]) -> StateDict:
    # The communicated update is Δ = uploaded state - pre-training state.
    out = {}
    for key, value in uploaded_state.items():
        if value.dtype.is_floating_point:
            out[key] = value - pre_train_state[key]
        else:
            out[key] = value.clone()
    return out


def _hash_tensor(tensor: torch.Tensor) -> bytes:
    cpu_tensor = tensor.detach().cpu().contiguous()
    header = f"{cpu_tensor.dtype}:{tuple(cpu_tensor.shape)}:".encode("utf-8")
    return header + cpu_tensor.numpy().tobytes()


def hash_state_dict(state_dict: Mapping[str, torch.Tensor]) -> str:
    # Include tensor names and metadata so structurally different states hash differently.
    digest = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_hash_tensor(state_dict[key]))
        digest.update(b"\0")
    return digest.hexdigest()


def hash_summary(summary: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(_hash_tensor(summary.detach().float()))
    return digest.hexdigest()


def graph_hash_from_neighbors(in_neighbors: Mapping[int, Sequence[int]]) -> str:
    # Bind messages to the round topology to reject stale-graph replay.
    canonical = {int(k): [int(v) for v in vals] for k, vals in sorted(in_neighbors.items())}
    return hashlib.sha256(_canonical_json(canonical)).hexdigest()


def _mod(value: int) -> int:
    return int(value) % FIELD_PRIME


def _eval_poly(coeffs: Sequence[int], x: int) -> int:
    # Evaluate the Shamir polynomial f(x) in the prime field.
    acc = 0
    power = 1
    for coeff in coeffs:
        acc = (acc + coeff * power) % FIELD_PRIME
        power = (power * x) % FIELD_PRIME
    return acc


def split_shamir_secret(secret: int, threshold: int, num_shares: int, seed: int) -> Tuple[List[Tuple[int, int]], List[int]]:
    """Split a field element into shares recoverable from any threshold subset."""

    if threshold < 2:
        raise ValueError("Shamir threshold must be at least 2.")
    if num_shares < threshold:
        raise ValueError("Number of shares must be >= threshold.")

    rng = random.Random(seed)
    coeffs = [_mod(secret)]
    coeffs.extend(rng.randrange(0, FIELD_PRIME) for _ in range(threshold - 1))
    shares = [(x, _eval_poly(coeffs, x)) for x in range(1, num_shares + 1)]
    return shares, coeffs


def recover_shamir_secret(shares: Sequence[Tuple[int, int]], threshold: int) -> int:
    """Recover f(0) from a threshold subset using Lagrange interpolation."""

    if len(shares) < threshold:
        raise ValueError("Not enough Shamir shares to recover the secret.")

    selected = list(shares[:threshold])
    secret = 0
    for i, (x_i, y_i) in enumerate(selected):
        numerator = 1
        denominator = 1
        for j, (x_j, _) in enumerate(selected):
            if i == j:
                continue
            numerator = (numerator * (-x_j)) % FIELD_PRIME
            denominator = (denominator * (x_i - x_j)) % FIELD_PRIME
        inv_denominator = pow(denominator % FIELD_PRIME, FIELD_PRIME - 2, FIELD_PRIME)
        secret = (secret + y_i * numerator * inv_denominator) % FIELD_PRIME
    return secret


def verify_vss_share(share: SummaryShare, commitment: VSSCommitment) -> bool:
    # Re-evaluate f(x) from the commitment data to validate the share.
    if len(share.values) != len(commitment.coeffs_by_dim):
        return False
    for value, coeffs in zip(share.values, commitment.coeffs_by_dim):
        if _eval_poly(coeffs, share.x) != value % FIELD_PRIME:
            return False
    return True


def quantize_summary(summary: torch.Tensor) -> List[int]:
    # Encode the floating-point summary as field elements before Shamir sharing.
    quantized = torch.round(summary.detach().cpu().float() * SUMMARY_SCALE).to(torch.int64)
    return [_mod(int(v.item())) for v in quantized.reshape(-1)]


class SecureProtocolContext:
    """Coordinate protected messages, summary validation, and committee aggregation."""

    def __init__(self, num_nodes: int, config):
        self.num_nodes = num_nodes
        self.config = config
        self.num_share_holders = int(getattr(config, "secure_num_share_holders", 3))
        self.shamir_threshold = int(getattr(config, "secure_shamir_threshold", 2))
        if self.num_share_holders < self.shamir_threshold:
            raise ValueError("secure_num_share_holders must be >= secure_shamir_threshold.")
        if self.shamir_threshold < 2:
            raise ValueError("secure_shamir_threshold must be at least 2.")
        self.min_valid_contributors = int(
            getattr(config, "secure_min_valid_contributors", 3)
        )
        if self.min_valid_contributors < 1:
            raise ValueError("secure_min_valid_contributors must be at least 1.")

        # Each training node signs protocol messages with its own key pair.
        self.private_keys = {
            node_id: Ed25519PrivateKey.generate()
            for node_id in range(num_nodes)
        }
        self.public_keys = {
            node_id: self.private_keys[node_id].public_key()
            for node_id in range(num_nodes)
        }
        self.committee_records: Dict[Tuple[int, int], CommitteeAggregationRecord] = {}
        self.historical_updates: Dict[Tuple[int, int], StateDict] = {}
        self.trusted_reference_updates: Dict[int, StateDict] = {}

    def _session_key(self, sender: int, receiver: int) -> bytes:
        # Derive a fixed-length message-protection key for each sender-receiver pair.
        seed = getattr(self.config, "seed", 0)
        material = f"{seed}:{sender}:{receiver}:secure-session".encode("utf-8")
        return hashlib.sha256(material).digest()

    def get_committee(self, receiver_node: int, round_idx: int) -> CommitteeAssignment:
        """Return the committee assignment and identifier for a receiver round."""

        member_ids = tuple(range(self.num_share_holders))
        committee_payload = {
            "receiver": int(receiver_node),
            "round": int(round_idx),
            "members": member_ids,
        }
        committee_id = hashlib.sha256(_canonical_json(committee_payload)).hexdigest()
        return CommitteeAssignment(
            committee_id=committee_id,
            receiver_node=int(receiver_node),
            round_idx=int(round_idx),
            member_ids=member_ids,
        )

    def protect_message(
        self,
        sender: int,
        receiver: int,
        msg_type: str,
        round_idx: int,
        graph_hash: str,
        payload,
    ) -> SecureMessage:
        """Encrypt a payload with AES-GCM and sign it with the sender's Ed25519 key."""

        nonce = os.urandom(12)
        meta = {
            "round": int(round_idx),
            "sender": int(sender),
            "receiver": int(receiver),
            "type": str(msg_type),
            "nonce": nonce.hex(),
            "graph_hash": graph_hash,
        }
        payload_bytes = _canonical_json(payload)
        # Authenticate routing and round metadata as AEAD associated data.
        ciphertext = AESGCM(self._session_key(sender, receiver)).encrypt(
            nonce,
            payload_bytes,
            _canonical_json(meta),
        )
        # Sign the metadata-ciphertext digest before transmission.
        signed = hashlib.sha256(_canonical_json(meta) + ciphertext).digest()
        signature = self.private_keys[sender].sign(signed)
        return SecureMessage(meta=meta, ciphertext=ciphertext, signature=signature)

    def open_message(
        self,
        message: SecureMessage,
        expected_receiver: int,
        expected_round: int,
        expected_graph_hash: str,
    ):
        """Validate message context, signature, and AEAD tag, then decode the payload."""

        meta = message.meta
        sender = int(meta["sender"])
        if int(meta["receiver"]) != int(expected_receiver):
            raise ValueError("Message receiver mismatch.")
        if int(meta["round"]) != int(expected_round):
            raise ValueError("Message round mismatch.")
        if meta["graph_hash"] != expected_graph_hash:
            raise ValueError("Message graph hash mismatch.")

        signed = hashlib.sha256(_canonical_json(meta) + message.ciphertext).digest()
        try:
            self.public_keys[sender].verify(message.signature, signed)
        except InvalidSignature as exc:
            raise ValueError("Message signature verification failed.") from exc

        nonce = bytes.fromhex(meta["nonce"])
        payload_bytes = AESGCM(self._session_key(sender, expected_receiver)).decrypt(
            nonce,
            message.ciphertext,
            _canonical_json(meta),
        )
        return json.loads(payload_bytes.decode("utf-8"))

    def _layer_group_name(self, key: str) -> str:
        parts = key.split(".")
        if len(parts) >= 3 and parts[1].isdigit() and parts[2].isdigit():
            return ".".join(parts[:3])
        if len(parts) >= 2:
            return ".".join(parts[:2])
        return key

    def _group_shared_state(
        self,
        update_state: Mapping[str, torch.Tensor],
    ) -> Dict[str, StateDict]:
        groups: Dict[str, StateDict] = {}
        for key, value in update_state.items():
            if not _is_shared_float_tensor(key, value):
                continue
            group_name = self._layer_group_name(key)
            groups.setdefault(group_name, {})[key] = value
        return groups

    def _project_group(
        self,
        group_state: Mapping[str, torch.Tensor],
        output_dim: int,
        projection_seed: int,
    ) -> torch.Tensor:
        """Project one shared parameter group into a fixed number of dimensions."""

        summary = torch.zeros(output_dim, dtype=torch.float64)
        offset = 0
        for key in sorted(group_state.keys()):
            flat = group_state[key].detach().float().cpu().reshape(-1)
            if flat.numel() == 0:
                continue
            idx = torch.arange(offset, offset + flat.numel(), dtype=torch.int64)
            buckets = ((idx * 1_103_515_245 + projection_seed) % output_dim).long()
            signs = torch.where(
                ((idx * 2_654_435_761 + projection_seed) % 2) == 0,
                torch.ones_like(flat, dtype=torch.float64),
                -torch.ones_like(flat, dtype=torch.float64),
            )
            summary.scatter_add_(0, buckets, flat.to(torch.float64) * signs)
            offset += flat.numel()
        return (summary / math.sqrt(max(offset, 1))).to(torch.float32)

    def state_dict_to_group_summaries(
        self,
        update_state: Mapping[str, torch.Tensor],
        projection_seed: int | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Build group-wise linear projections with a fixed total dimension."""

        if projection_seed is None:
            projection_seed = int(
                getattr(self.config, "secure_projection_seed", 2026)
            )
        summary_dim = int(getattr(self.config, "secure_summary_dim", 128))
        groups = self._group_shared_state(update_state)
        group_names = sorted(groups.keys())
        if not group_names:
            return {}

        base_dim = summary_dim // len(group_names)
        remainder = summary_dim % len(group_names)
        group_summaries = {}
        for group_idx, group_name in enumerate(group_names):
            output_dim = base_dim + (1 if group_idx < remainder else 0)
            output_dim = max(output_dim, 1)
            seed_payload = (
                f"{int(projection_seed)}:{group_idx}:{group_name}:group-projection"
            ).encode("utf-8")
            group_seed = int.from_bytes(
                hashlib.sha256(seed_payload).digest()[:8],
                "big",
            ) % FIELD_PRIME
            group_summaries[group_name] = self._project_group(
                group_state=groups[group_name],
                output_dim=output_dim,
                projection_seed=group_seed,
            )
        return group_summaries

    def _combine_group_summaries(
        self,
        group_summaries: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        summary_dim = int(getattr(self.config, "secure_summary_dim", 128))
        if not group_summaries:
            return torch.zeros(summary_dim, dtype=torch.float32)
        combined = torch.cat(
            [group_summaries[name].float() for name in sorted(group_summaries.keys())]
        )
        if combined.numel() < summary_dim:
            combined = torch.cat([
                combined,
                torch.zeros(summary_dim - combined.numel(), dtype=torch.float32),
            ])
        return combined[:summary_dim]

    def state_dict_to_summary(
        self,
        update_state: Mapping[str, torch.Tensor],
        projection_seed: int | None = None,
    ) -> torch.Tensor:
        """Compress a high-dimensional shared update into the summary Phi(Δ)."""

        group_summaries = self.state_dict_to_group_summaries(
            update_state=update_state,
            projection_seed=projection_seed,
        )
        return self._combine_group_summaries(group_summaries)

    def _layer_proportions(
        self,
        group_summaries: Mapping[str, torch.Tensor],
    ) -> Dict[str, float]:
        norms = {
            name: float(torch.norm(summary.float(), p=2).item())
            for name, summary in group_summaries.items()
        }
        total = sum(norms.values()) + 1e-12
        return {
            name: value / total
            for name, value in norms.items()
        }

    def state_dict_l2_norm(self, state_dict: Mapping[str, torch.Tensor]) -> float:
        """Compute the L2 norm of shared floating-point update tensors."""

        total = 0.0
        for key, value in state_dict.items():
            if _is_shared_float_tensor(key, value):
                total += float(torch.sum(value.detach().float().cpu() ** 2).item())
        return math.sqrt(total)

    def create_summary_shares(
        self,
        summary: torch.Tensor,
        node_id: int,
        round_idx: int,
        receiver_node: int | None = None,
    ) -> Tuple[List[SummaryShare], VSSCommitment]:
        """Create Shamir shares and validation data for every summary dimension."""

        quantized = quantize_summary(summary)
        holder_values = [[] for _ in range(self.num_share_holders)]
        coeffs_by_dim = []

        for dim_idx, secret in enumerate(quantized):
            seed_material = (
                f"{getattr(self.config, 'seed', 0)}:{round_idx}:"
                f"{receiver_node}:{node_id}:{dim_idx}:summary-share"
            ).encode("utf-8")
            seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
            shares, coeffs = split_shamir_secret(
                secret=secret,
                threshold=self.shamir_threshold,
                num_shares=self.num_share_holders,
                seed=seed,
            )
            coeffs_by_dim.append(coeffs)
            for holder_idx, (_, value) in enumerate(shares):
                holder_values[holder_idx].append(value)

        shares = [
            SummaryShare(holder_id=i + 1, x=i + 1, values=holder_values[i])
            for i in range(self.num_share_holders)
        ]
        return shares, VSSCommitment(coeffs_by_dim=coeffs_by_dim)

    def create_binding_proof(
        self,
        update_state: Mapping[str, torch.Tensor],
        summary: torch.Tensor,
        projection_seed: int | None = None,
    ) -> BindingProof:
        """Build the hashes and range metadata that bind a summary to an update."""

        if projection_seed is None:
            projection_seed = int(
                getattr(self.config, "secure_projection_seed", 2026)
            )
        return BindingProof(
            update_hash=hash_state_dict(update_state),
            summary_hash=hash_summary(summary),
            update_norm=self.state_dict_l2_norm(update_state),
            summary_norm=float(torch.norm(summary.float(), p=2).item()),
            projection_seed=int(projection_seed),
        )

    def verify_binding_proof(
        self,
        update_state: Mapping[str, torch.Tensor],
        summary: torch.Tensor,
        proof: BindingProof,
    ) -> bool:
        """Verify update-summary consistency and configured norm bounds."""

        if hash_state_dict(update_state) != proof.update_hash:
            return False
        if hash_summary(summary) != proof.summary_hash:
            return False
        expected_summary = self.state_dict_to_summary(
            update_state,
            projection_seed=proof.projection_seed,
        )
        if not torch.allclose(expected_summary, summary, rtol=1e-5, atol=1e-6):
            return False
        current_update_norm = self.state_dict_l2_norm(update_state)
        current_summary_norm = float(torch.norm(summary.float(), p=2).item())
        if abs(current_update_norm - proof.update_norm) > 1e-5 * max(current_update_norm, proof.update_norm, 1.0):
            return False
        if abs(current_summary_norm - proof.summary_norm) > 1e-5 * max(current_summary_norm, proof.summary_norm, 1.0):
            return False
        update_bound = float(getattr(self.config, "secure_update_norm_bound", 1e9))
        summary_bound = float(getattr(self.config, "secure_summary_norm_bound", 1e9))
        if proof.update_norm > update_bound:
            return False
        if proof.summary_norm > summary_bound:
            return False
        return True

    def _derive_projection_seed(
        self,
        round_idx: int,
        receiver_node: int,
        graph_hash: str,
        commitment_root: str,
    ) -> int:
        """Derive the receiver-specific projection seed from committed round context."""

        payload = {
            "seed": int(getattr(self.config, "seed", 0)),
            "round": int(round_idx),
            "receiver": int(receiver_node),
            "graph_hash": graph_hash,
            "commitment_root": commitment_root,
        }
        return int.from_bytes(
            hashlib.sha256(_canonical_json(payload)).digest()[:8],
            "big",
        ) % FIELD_PRIME

    def _ema_state_dict(
        self,
        old_state: Mapping[str, torch.Tensor] | None,
        new_state: Mapping[str, torch.Tensor],
    ) -> StateDict:
        """Update the exponential moving average used by protocol-side scoring."""

        if old_state is None:
            return self._clone_state_dict(new_state)
        beta = float(getattr(self.config, "ema_momentum_beta", 0.90))
        out = {}
        for key, value in new_state.items():
            if value.dtype.is_floating_point:
                out[key] = (
                    beta * old_state[key].to(value.device)
                    + (1.0 - beta) * value
                ).detach().clone()
            else:
                out[key] = value.detach().clone()
        return out

    def _sum_group_summaries(
        self,
        summaries_by_holder: Sequence[Mapping[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Combine group summaries computed from committee-held additive shares."""

        if not summaries_by_holder:
            return {}
        names = sorted({
            name
            for holder_summaries in summaries_by_holder
            for name in holder_summaries.keys()
        })
        combined = {}
        for name in names:
            values = [
                holder_summaries[name]
                for holder_summaries in summaries_by_holder
                if name in holder_summaries
            ]
            combined[name] = sum(values[1:], values[0].clone())
        return combined

    def _group_summaries_match(
        self,
        sender_summaries: Mapping[str, torch.Tensor],
        committee_summaries: Mapping[str, torch.Tensor],
    ) -> bool:
        """Check the sender summary against the summary reconstructed from shares."""

        if set(sender_summaries.keys()) != set(committee_summaries.keys()):
            return False
        tolerance = float(
            getattr(self.config, "secure_binding_tolerance", 1e-5)
        )
        for name in sender_summaries.keys():
            if not torch.allclose(
                sender_summaries[name],
                committee_summaries[name],
                rtol=tolerance,
                atol=tolerance,
            ):
                return False
        return True

    def prepare_round(
        self,
        round_idx: int,
        uploaded_states: Mapping[int, Mapping[str, torch.Tensor]],
        pre_train_states: Mapping[int, Mapping[str, torch.Tensor]],
        in_neighbors: Mapping[int, Sequence[int]],
    ) -> RoundSecurityReport:
        """Prepare commitments, binding data, and valid contributors for each receiver."""

        graph_hash = graph_hash_from_neighbors(in_neighbors)
        updates = {
            node_id: _state_dict_update(
                uploaded_states[node_id],
                pre_train_states[node_id],
            )
            for node_id in range(self.num_nodes)
        }
        receiver_materials = {}
        stats = {
            "secure_valid_nodes": 0,
            "secure_invalid_nodes": 0,
            "secure_signature_failures": 0,
            "secure_vss_failures": 0,
            "secure_zkp_failures": 0,
            "secure_binding_failures": 0,
        }

        for receiver_node in range(self.num_nodes):
            candidates = tuple(dict.fromkeys(
                [receiver_node] + [
                    int(node_id)
                    for node_id in in_neighbors.get(receiver_node, [])
                ]
            ))
            committee = self.get_committee(
                receiver_node=receiver_node,
                round_idx=round_idx,
            )
            sender_shares = {
                sender_node: self._create_additive_update_shares(
                    update_state=updates[sender_node],
                    sender_node=sender_node,
                    receiver_node=receiver_node,
                    round_idx=round_idx,
                )
                for sender_node in candidates
            }
            share_commitments, commitment_root = (
                self._committee_share_commitments(
                    sender_shares=sender_shares,
                    committee=committee,
                )
            )
            projection_seed = self._derive_projection_seed(
                round_idx=round_idx,
                receiver_node=receiver_node,
                graph_hash=graph_hash,
                commitment_root=commitment_root,
            )
            trusted_reference_update = self.trusted_reference_updates.get(
                receiver_node,
                updates[receiver_node],
            )
            trusted_reference_summary = self.state_dict_to_summary(
                trusted_reference_update,
                projection_seed=projection_seed,
            )
            materials = {}

            for sender_node in candidates:
                update_state = updates[sender_node]
                group_summaries = self.state_dict_to_group_summaries(
                    update_state,
                    projection_seed=projection_seed,
                )
                summary = self._combine_group_summaries(group_summaries)
                committee_group_summaries = self._sum_group_summaries([
                    self.state_dict_to_group_summaries(
                        share,
                        projection_seed=projection_seed,
                    )
                    for share in sender_shares[sender_node]
                ])
                committee_summary = self._combine_group_summaries(
                    committee_group_summaries
                )
                historical_update = self.historical_updates.get(
                    (receiver_node, sender_node),
                    update_state,
                )
                historical_summary = self.state_dict_to_summary(
                    historical_update,
                    projection_seed=projection_seed,
                )
                shares, commitment = self.create_summary_shares(
                    summary,
                    node_id=sender_node,
                    round_idx=round_idx,
                    receiver_node=receiver_node,
                )
                proof = self.create_binding_proof(
                    update_state,
                    summary,
                    projection_seed=projection_seed,
                )
                payload = {
                    "node_id": sender_node,
                    "receiver_node": receiver_node,
                    "summary_hash": proof.summary_hash,
                    "update_hash": proof.update_hash,
                    "update_norm": proof.update_norm,
                    "summary_norm": proof.summary_norm,
                    "projection_seed": projection_seed,
                    "commitment_root": commitment_root,
                }
                message = self.protect_message(
                    sender=sender_node,
                    receiver=receiver_node,
                    msg_type="summary-binding-proof",
                    round_idx=round_idx,
                    graph_hash=graph_hash,
                    payload=payload,
                )

                failures = []
                try:
                    opened_payload = self.open_message(
                        message,
                        expected_receiver=receiver_node,
                        expected_round=round_idx,
                        expected_graph_hash=graph_hash,
                    )
                    if opened_payload != payload:
                        raise ValueError("Binding payload mismatch.")
                except ValueError:
                    failures.append("signature")
                    stats["secure_signature_failures"] += 1

                if not all(
                    verify_vss_share(share, commitment)
                    for share in shares
                ):
                    failures.append("vss")
                    stats["secure_vss_failures"] += 1

                if not self.verify_binding_proof(
                    update_state,
                    summary,
                    proof,
                ):
                    failures.append("zkp")
                    stats["secure_zkp_failures"] += 1

                if not self._group_summaries_match(
                    sender_summaries=group_summaries,
                    committee_summaries=committee_group_summaries,
                ):
                    failures.append("binding")
                    stats["secure_binding_failures"] += 1

                valid = len(failures) == 0
                if valid:
                    stats["secure_valid_nodes"] += 1
                else:
                    stats["secure_invalid_nodes"] += 1

                materials[sender_node] = NodeSecurityMaterial(
                    summary=summary,
                    committee_summary=committee_summary,
                    group_summaries=group_summaries,
                    committee_group_summaries=committee_group_summaries,
                    historical_summary=historical_summary,
                    trusted_reference_summary=trusted_reference_summary,
                    layer_proportions=self._layer_proportions(group_summaries),
                    summary_shares=shares,
                    vss_commitment=commitment,
                    binding_proof=proof,
                    message=message,
                    valid=valid,
                    failures=failures,
                )

            valid_nodes = {
                node_id
                for node_id, material in materials.items()
                if material.valid
            }
            receiver_materials[receiver_node] = ReceiverCommitteeMaterial(
                committee=committee,
                candidate_nodes=candidates,
                commitment_root=commitment_root,
                projection_seed=projection_seed,
                share_commitments=share_commitments,
                materials=materials,
                valid_nodes=valid_nodes,
            )
            for sender_node in valid_nodes:
                history_key = (receiver_node, sender_node)
                self.historical_updates[history_key] = self._ema_state_dict(
                    self.historical_updates.get(history_key),
                    updates[sender_node],
                )

        representative = receiver_materials.get(0)
        if representative is None:
            representative_materials = {}
            representative_valid_nodes = set()
        else:
            representative_materials = representative.materials
            representative_valid_nodes = representative.valid_nodes
        return RoundSecurityReport(
            round_idx=round_idx,
            graph_hash=graph_hash,
            materials=representative_materials,
            valid_nodes=representative_valid_nodes,
            receiver_materials=receiver_materials,
            stats=stats,
        )

    def compute_mpc_friendly_apt_scores(
        self,
        receiver_node: int,
        candidate_nodes: Iterable[int],
        report: RoundSecurityReport,
    ) -> Tuple[Dict[int, float], Dict[str, float]]:
        """Compute four-view quantile APT scores from committee-validated summaries."""

        candidates = [int(node_id) for node_id in candidate_nodes]
        receiver_material = report.receiver_materials.get(receiver_node)
        if receiver_material is None:
            return {node_id: 0.0 for node_id in candidates}, {
                "apt_enabled": False,
                "avg_raw_distance": 0.0,
                "avg_apt_score": 0.0,
                "avg_instant_score": 0.0,
                "apt_q_low": 0.0,
                "apt_q_high": 0.0,
                "metrics_by_node": {},
            }

        materials = receiver_material.materials
        valid_candidates = [
            node_id
            for node_id in candidates
            if (
                node_id in receiver_material.valid_nodes
                and node_id in materials
            )
        ]
        valid_neighbors = [
            node_id
            for node_id in valid_candidates
            if node_id != receiver_node
        ]
        if not valid_candidates:
            return {node_id: 0.0 for node_id in candidates}, {
                "apt_enabled": False,
                "avg_raw_distance": 0.0,
                "avg_apt_score": 0.0,
                "avg_instant_score": 0.0,
                "apt_q_low": 0.0,
                "apt_q_high": 0.0,
                "metrics_by_node": {},
            }

        summary_norms = {
            node_id: float(torch.norm(
                materials[node_id].summary.float(),
                p=2,
            ).item())
            for node_id in valid_candidates
        }
        norm_values = torch.tensor(
            list(summary_norms.values()),
            dtype=torch.float64,
        )
        norm_median = float(torch.median(norm_values).item())
        norm_mad = float(
            torch.median(torch.abs(norm_values - norm_median)).item()
        )

        layer_names = sorted({
            name
            for node_id in valid_candidates
            for name in materials[node_id].layer_proportions.keys()
        })
        layer_medians = {}
        for name in layer_names:
            values = torch.tensor([
                materials[node_id].layer_proportions.get(name, 0.0)
                for node_id in valid_candidates
            ], dtype=torch.float64)
            layer_medians[name] = float(torch.median(values).item())

        metric_weights = []
        if getattr(self.config, "use_cosine_distance", True):
            metric_weights.append((
                "cosine_dist",
                float(getattr(self.config, "dist_weight_cosine", 0.25)),
            ))
        if getattr(self.config, "use_norm_ratio_distance", True):
            metric_weights.append((
                "norm_ratio_dist",
                float(getattr(self.config, "dist_weight_norm_ratio", 0.25)),
            ))
        if getattr(self.config, "use_ema_momentum_distance", True):
            metric_weights.append((
                "momentum_dist",
                float(getattr(self.config, "dist_weight_momentum", 0.25)),
            ))
        if getattr(self.config, "use_layerwise_distance", True):
            metric_weights.append((
                "layerwise_dist",
                float(getattr(self.config, "dist_weight_layerwise", 0.25)),
            ))
        weight_total = sum(weight for _, weight in metric_weights)
        if weight_total <= 0.0:
            metric_weights = [("cosine_dist", 1.0)]
            weight_total = 1.0

        metrics_by_node = {}
        raw_distances = {}
        for node_id in valid_candidates:
            material = materials[node_id]
            cosine_dist = self._cosine_distance(
                material.summary,
                material.trusted_reference_summary,
            )
            norm_ratio_dist = min(
                1.0,
                abs(summary_norms[node_id] - norm_median)
                / max(abs(norm_median) + norm_mad, 1e-12),
            )
            momentum_dist = self._cosine_distance(
                material.summary,
                material.historical_summary,
            )
            layerwise_dist = min(
                1.0,
                0.5 * sum(
                    abs(
                        material.layer_proportions.get(name, 0.0)
                        - layer_medians[name]
                    )
                    for name in layer_names
                ),
            )
            metrics = {
                "cosine_dist": cosine_dist,
                "norm_ratio_dist": norm_ratio_dist,
                "momentum_dist": momentum_dist,
                "layerwise_dist": layerwise_dist,
            }
            raw_distance = sum(
                metrics[name] * weight
                for name, weight in metric_weights
            ) / weight_total
            metrics["raw_distance"] = float(raw_distance)
            metrics_by_node[node_id] = metrics
            raw_distances[node_id] = float(raw_distance)

        apt_enabled = bool(
            getattr(self.config, "use_apt", True)
            and (report.round_idx + 1)
            >= int(getattr(self.config, "apt_start_round", 1))
            and valid_neighbors
        )
        q_low = 0.0
        q_high = 0.0
        scores = {
            node_id: (1.0 if node_id == receiver_node else 0.0)
            for node_id in candidates
        }
        if not apt_enabled:
            for node_id in valid_neighbors:
                scores[node_id] = 1.0
        elif len(valid_neighbors) < 3:
            temperature = float(
                getattr(
                    self.config,
                    "secure_small_degree_temperature",
                    getattr(self.config, "trust_distance_temperature", 0.35),
                )
            )
            for node_id in valid_neighbors:
                scores[node_id] = math.exp(
                    -raw_distances[node_id] / max(temperature, 1e-12)
                )
        else:
            neighbor_distances = torch.tensor(
                [raw_distances[node_id] for node_id in valid_neighbors],
                dtype=torch.float64,
            )
            q_low = float(torch.quantile(
                neighbor_distances,
                float(getattr(self.config, "apt_accept_quantile", 0.40)),
            ).item())
            q_high = float(torch.quantile(
                neighbor_distances,
                float(getattr(self.config, "apt_reject_quantile", 0.85)),
            ).item())
            gap = max(q_high - q_low, 1e-12)
            for node_id in valid_neighbors:
                distance = raw_distances[node_id]
                scores[node_id] = max(
                    0.0,
                    min(1.0, (q_high - distance) / gap),
                )

        neighbor_scores = [scores[node_id] for node_id in valid_neighbors]
        neighbor_distances = [
            raw_distances[node_id]
            for node_id in valid_neighbors
        ]
        stats = {
            "apt_enabled": apt_enabled,
            "avg_raw_distance": (
                sum(neighbor_distances) / max(len(neighbor_distances), 1)
            ),
            "avg_apt_score": (
                sum(neighbor_scores) / max(len(neighbor_scores), 1)
            ),
            "avg_instant_score": (
                sum(neighbor_scores) / max(len(neighbor_scores), 1)
            ),
            "apt_q_low": q_low,
            "apt_q_high": q_high,
            "metrics_by_node": metrics_by_node,
        }
        return scores, stats

    def _cosine_distance(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
    ) -> float:
        """Return cosine distance scaled to the interval [0, 1]."""

        first = first.detach().float().reshape(-1)
        second = second.detach().float().reshape(-1)
        denominator = float(
            torch.norm(first, p=2).item()
            * torch.norm(second, p=2).item()
        )
        if denominator <= 1e-12:
            return 0.0 if torch.allclose(first, second) else 1.0
        cosine = float(torch.dot(first, second).item()) / denominator
        return max(0.0, min(1.0, 0.5 * (1.0 - cosine)))

    def _clone_state_dict(self, state_dict: Mapping[str, torch.Tensor]) -> StateDict:
        return {key: value.detach().clone() for key, value in state_dict.items()}

    def _create_additive_update_shares(
        self,
        update_state: Mapping[str, torch.Tensor],
        sender_node: int,
        receiver_node: int,
        round_idx: int,
    ) -> List[StateDict]:
        """Split an update into additive shares held by committee members."""

        member_shares = [dict() for _ in range(self.num_share_holders)]
        for key, value in update_state.items():
            if not _is_shared_float_tensor(key, value):
                for holder_idx in range(self.num_share_holders):
                    member_shares[holder_idx][key] = torch.zeros_like(value)
                continue

            if self.num_share_holders == 1:
                member_shares[0][key] = value.detach().clone()
                continue

            seed_material = (
                f"{getattr(self.config, 'seed', 0)}:{round_idx}:"
                f"{receiver_node}:{sender_node}:{key}:committee-share"
            ).encode("utf-8")
            share_seed = int.from_bytes(
                hashlib.sha256(seed_material).digest()[:8],
                "big",
            )
            generator = torch.Generator(device=value.device)
            generator.manual_seed(share_seed)
            magnitude = max(
                float(value.detach().float().abs().mean().item()),
                1e-3,
            )

            random_shares = []
            for holder_idx in range(self.num_share_holders - 1):
                share = torch.randn(
                    value.shape,
                    generator=generator,
                    device=value.device,
                    dtype=value.dtype,
                ) * magnitude
                member_shares[holder_idx][key] = share
                random_shares.append(share)
            member_shares[-1][key] = value - sum(random_shares)

        return member_shares

    def _committee_commitment_root(
        self,
        sender_shares: Mapping[int, Sequence[Mapping[str, torch.Tensor]]],
        committee: CommitteeAssignment,
    ) -> str:
        """Compute the commitment root for the committee's round inputs."""

        _, root = self._committee_share_commitments(
            sender_shares=sender_shares,
            committee=committee,
        )
        return root

    def _committee_share_commitments(
        self,
        sender_shares: Mapping[int, Sequence[Mapping[str, torch.Tensor]]],
        committee: CommitteeAssignment,
    ) -> Tuple[Dict[int, Tuple[str, ...]], str]:
        """Hash every sender share and compute the committee commitment root."""

        commitments = []
        share_commitments = {}
        for sender_node in sorted(sender_shares.keys()):
            shares = sender_shares[sender_node]
            sender_hashes = []
            for holder_idx, share in enumerate(shares):
                share_hash = hash_state_dict(share)
                sender_hashes.append(share_hash)
                commitments.append({
                    "sender": int(sender_node),
                    "holder": int(committee.member_ids[holder_idx]),
                    "share_hash": share_hash,
                })
            share_commitments[int(sender_node)] = tuple(sender_hashes)
        payload = {
            "committee_id": committee.committee_id,
            "commitments": commitments,
        }
        root = hashlib.sha256(_canonical_json(payload)).hexdigest()
        return share_commitments, root

    def _create_weight_request(
        self,
        committee: CommitteeAssignment,
        contributor_nodes: Sequence[int],
        weights: Sequence[float],
        commitment_root: str,
    ) -> WeightRequest:
        """Create the receiver's signed weight request for this round."""

        weight_items = tuple(
            (int(node_id), float(weight))
            for node_id, weight in zip(contributor_nodes, weights)
        )
        request_payload = {
            "committee_id": committee.committee_id,
            "receiver": committee.receiver_node,
            "round": committee.round_idx,
            "request_counter": 1,
            "contributors": [int(node_id) for node_id in contributor_nodes],
            "weights": weight_items,
            "commitment_root": commitment_root,
        }
        request_digest = hashlib.sha256(_canonical_json(request_payload)).digest()
        signature = self.private_keys[committee.receiver_node].sign(request_digest)
        return WeightRequest(
            request_id=request_digest.hex(),
            committee_id=committee.committee_id,
            receiver_node=committee.receiver_node,
            round_idx=committee.round_idx,
            request_counter=1,
            contributor_nodes=tuple(int(node_id) for node_id in contributor_nodes),
            weights=weight_items,
            commitment_root=commitment_root,
            signature=signature,
        )

    def _verify_weight_request(
        self,
        request: WeightRequest,
        committee: CommitteeAssignment,
        valid_nodes: Iterable[int],
    ) -> None:
        """Validate committee context, contributors, weights, and receiver signature."""

        if request.committee_id != committee.committee_id:
            raise CommitteeAggregationError("Committee identifier mismatch.")
        if request.receiver_node != committee.receiver_node:
            raise CommitteeAggregationError("Weight request receiver mismatch.")
        if request.round_idx != committee.round_idx:
            raise CommitteeAggregationError("Weight request round mismatch.")
        if request.request_counter != 1:
            raise CommitteeAggregationError("Weight request counter mismatch.")
        if len(request.contributor_nodes) < self.min_valid_contributors:
            raise CommitteeAggregationError("Not enough valid contributors.")
        if len(set(request.contributor_nodes)) != len(request.contributor_nodes):
            raise CommitteeAggregationError("Duplicate contributor in weight request.")

        valid_node_set = {int(node_id) for node_id in valid_nodes}
        if not set(request.contributor_nodes).issubset(valid_node_set):
            raise CommitteeAggregationError("Weight request contains an invalid contributor.")

        request_weights = dict(request.weights)
        if set(request_weights.keys()) != set(request.contributor_nodes):
            raise CommitteeAggregationError("Weight keys do not match contributors.")
        if any(not math.isfinite(weight) or weight < 0.0 for weight in request_weights.values()):
            raise CommitteeAggregationError("Aggregation weights must be finite and nonnegative.")
        if not math.isclose(sum(request_weights.values()), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise CommitteeAggregationError("Aggregation weights must sum to one.")

        request_payload = {
            "committee_id": request.committee_id,
            "receiver": request.receiver_node,
            "round": request.round_idx,
            "request_counter": request.request_counter,
            "contributors": list(request.contributor_nodes),
            "weights": request.weights,
            "commitment_root": request.commitment_root,
        }
        request_digest = hashlib.sha256(_canonical_json(request_payload)).digest()
        if request.request_id != request_digest.hex():
            raise CommitteeAggregationError("Weight request identifier mismatch.")
        try:
            self.public_keys[request.receiver_node].verify(
                request.signature,
                request_digest,
            )
        except InvalidSignature as exc:
            raise CommitteeAggregationError("Weight request signature verification failed.") from exc

    def _committee_weighted_sum(
        self,
        sender_shares: Mapping[int, Sequence[Mapping[str, torch.Tensor]]],
        contributor_nodes: Sequence[int],
        weights: Sequence[float],
    ) -> StateDict:
        """Sum weighted shares at each member and reconstruct the aggregate update."""

        reference_update = sender_shares[contributor_nodes[0]][0]
        member_aggregates = [
            {
                key: torch.zeros_like(value)
                for key, value in reference_update.items()
            }
            for _ in range(self.num_share_holders)
        ]

        for sender_node, weight in zip(contributor_nodes, weights):
            shares = sender_shares[sender_node]
            for holder_idx, share in enumerate(shares):
                for key, value in share.items():
                    if _is_shared_float_tensor(key, value):
                        member_aggregates[holder_idx][key] += value * float(weight)

        recovered = {}
        for key, value in reference_update.items():
            if _is_shared_float_tensor(key, value):
                recovered[key] = sum(
                    member_aggregates[holder_idx][key]
                    for holder_idx in range(self.num_share_holders)
                ).to(dtype=value.dtype)
            else:
                recovered[key] = torch.zeros_like(value)
        return recovered

    def secure_aggregate_update_dicts(
        self,
        update_dicts: Sequence[Mapping[str, torch.Tensor]],
        sender_nodes: Sequence[int],
        weights: Sequence[float],
        round_idx: int,
        receiver_node: int,
        valid_nodes: Iterable[int],
        round_report: RoundSecurityReport | None = None,
    ) -> StateDict:
        """Aggregate validated updates through the receiver's committee."""

        if not (len(update_dicts) == len(sender_nodes) == len(weights)):
            raise CommitteeAggregationError(
                "Updates, sender nodes, and weights must have the same length."
            )
        if len(update_dicts) == 0:
            raise CommitteeAggregationError("Committee aggregation needs contributors.")
        if len(set(int(node_id) for node_id in sender_nodes)) != len(sender_nodes):
            raise CommitteeAggregationError("Contributor node identifiers must be unique.")

        total = float(sum(weights))
        if not math.isfinite(total) or total <= 0.0:
            raise CommitteeAggregationError(
                "Aggregation weights must sum to a positive value."
            )
        normalized_weights = [float(weight) / total for weight in weights]
        contributor_nodes = [int(node_id) for node_id in sender_nodes]
        receiver_material = None
        if round_report is not None:
            if round_report.round_idx != int(round_idx):
                raise CommitteeAggregationError(
                    "Round report does not match the aggregation round."
                )
            receiver_material = round_report.receiver_materials.get(
                int(receiver_node)
            )
            if receiver_material is None:
                raise CommitteeAggregationError(
                    "Receiver committee material is unavailable."
                )
            committee = receiver_material.committee
            valid_nodes = receiver_material.valid_nodes
        else:
            committee = self.get_committee(
                receiver_node=receiver_node,
                round_idx=round_idx,
            )

        sender_shares = {}
        for sender_node, update_state in zip(contributor_nodes, update_dicts):
            sender_shares[sender_node] = self._create_additive_update_shares(
                update_state=update_state,
                sender_node=sender_node,
                receiver_node=receiver_node,
                round_idx=round_idx,
            )

        if receiver_material is not None:
            for sender_node in contributor_nodes:
                expected_hashes = receiver_material.share_commitments.get(
                    sender_node
                )
                actual_hashes = tuple(
                    hash_state_dict(share)
                    for share in sender_shares[sender_node]
                )
                if (
                    expected_hashes is None
                    or actual_hashes != expected_hashes
                ):
                    raise CommitteeAggregationError(
                        "Contributor shares do not match the committed update."
                    )
            commitment_root = receiver_material.commitment_root
        else:
            commitment_root = self._committee_commitment_root(
                sender_shares=sender_shares,
                committee=committee,
            )
        request = self._create_weight_request(
            committee=committee,
            contributor_nodes=contributor_nodes,
            weights=normalized_weights,
            commitment_root=commitment_root,
        )
        self._verify_weight_request(
            request=request,
            committee=committee,
            valid_nodes=valid_nodes,
        )

        record_key = (int(round_idx), int(receiver_node))
        existing_record = self.committee_records.get(record_key)
        if existing_record is not None:
            if existing_record.request.request_id != request.request_id:
                raise CommitteeAggregationError(
                    "A conflicting weight request already exists for this round."
                )
            return self._clone_state_dict(existing_record.aggregated_update)

        aggregated_update = self._committee_weighted_sum(
            sender_shares=sender_shares,
            contributor_nodes=contributor_nodes,
            weights=normalized_weights,
        )
        self.committee_records[record_key] = CommitteeAggregationRecord(
            request=request,
            aggregated_update=self._clone_state_dict(aggregated_update),
        )
        self.trusted_reference_updates[int(receiver_node)] = (
            self._ema_state_dict(
                self.trusted_reference_updates.get(int(receiver_node)),
                aggregated_update,
            )
        )
        return aggregated_update

    def try_secure_aggregate_update_dicts(
        self,
        update_dicts: Sequence[Mapping[str, torch.Tensor]],
        sender_nodes: Sequence[int],
        weights: Sequence[float],
        round_idx: int,
        receiver_node: int,
        valid_nodes: Iterable[int],
        round_report: RoundSecurityReport | None = None,
    ) -> Tuple[StateDict | None, Dict[str, object]]:
        """Run committee aggregation and return a result with protocol status."""

        try:
            aggregated_update = self.secure_aggregate_update_dicts(
                update_dicts=update_dicts,
                sender_nodes=sender_nodes,
                weights=weights,
                round_idx=round_idx,
                receiver_node=receiver_node,
                valid_nodes=valid_nodes,
                round_report=round_report,
            )
        except CommitteeAggregationError as exc:
            return None, {
                "success": False,
                "reason": str(exc),
            }

        record = self.committee_records[(int(round_idx), int(receiver_node))]
        return aggregated_update, {
            "success": True,
            "reason": "",
            "committee_id": record.request.committee_id,
            "request_id": record.request.request_id,
            "commitment_root": record.request.commitment_root,
            "contributors": len(record.request.contributor_nodes),
        }

    def secure_aggregate_state_dicts(
        self,
        state_dicts: Sequence[Mapping[str, torch.Tensor]],
        weights: Sequence[float],
        round_idx: int,
        receiver_node: int,
    ) -> StateDict:
        """Aggregate state dictionaries through the update-dictionary interface."""

        sender_nodes = list(range(len(state_dicts)))
        return self.secure_aggregate_update_dicts(
            update_dicts=state_dicts,
            sender_nodes=sender_nodes,
            weights=weights,
            round_idx=round_idx,
            receiver_node=receiver_node,
            valid_nodes=sender_nodes,
        )
