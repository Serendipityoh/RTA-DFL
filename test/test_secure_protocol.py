import unittest
from dataclasses import dataclass

import torch

from secure_protocol import (
    SecureMessage,
    SecureProtocolContext,
    SummaryShare,
    recover_shamir_secret,
    split_shamir_secret,
    verify_vss_share,
)
from utils import average_state_dicts


@dataclass
class DummyConfig:
    # Minimal values required by SecureProtocolContext.
    seed: int = 42
    secure_num_share_holders: int = 3
    secure_shamir_threshold: int = 2
    secure_min_valid_contributors: int = 3
    secure_summary_dim: int = 8
    secure_projection_seed: int = 2026
    secure_update_norm_bound: float = 1e9
    secure_summary_norm_bound: float = 1e9
    secure_binding_tolerance: float = 1e-5
    secure_small_degree_temperature: float = 0.35
    use_apt: bool = True
    apt_start_round: int = 1
    apt_accept_quantile: float = 0.40
    apt_reject_quantile: float = 0.85
    use_cosine_distance: bool = True
    use_norm_ratio_distance: bool = True
    use_ema_momentum_distance: bool = True
    use_layerwise_distance: bool = True
    dist_weight_cosine: float = 0.25
    dist_weight_norm_ratio: float = 0.25
    dist_weight_momentum: float = 0.25
    dist_weight_layerwise: float = 0.25
    ema_momentum_beta: float = 0.90


class SecureProtocolTests(unittest.TestCase):
    def setUp(self):
        self.cfg = DummyConfig()
        self.protocol = SecureProtocolContext(num_nodes=3, config=self.cfg)

    def test_aead_and_signature_reject_tampering(self):
        # Metadata or ciphertext changes must invalidate a protected message.
        message = self.protocol.protect_message(
            sender=0,
            receiver=1,
            msg_type="zkp-proof",
            round_idx=3,
            graph_hash="abc",
            payload={"ok": True},
        )
        payload = self.protocol.open_message(message, expected_receiver=1, expected_round=3, expected_graph_hash="abc")
        self.assertEqual(payload, {"ok": True})

        tampered_meta = SecureMessage(
            meta={**message.meta, "type": "summary-share"},
            ciphertext=message.ciphertext,
            signature=message.signature,
        )
        with self.assertRaises(ValueError):
            self.protocol.open_message(tampered_meta, expected_receiver=1, expected_round=3, expected_graph_hash="abc")

        tampered_ciphertext = SecureMessage(
            meta=message.meta,
            ciphertext=message.ciphertext[:-1] + bytes([message.ciphertext[-1] ^ 1]),
            signature=message.signature,
        )
        with self.assertRaises(ValueError):
            self.protocol.open_message(tampered_ciphertext, expected_receiver=1, expected_round=3, expected_graph_hash="abc")

    def test_shamir_recovery_and_vss_share_check(self):
        # The threshold recovers the secret and a modified share fails validation.
        secret = 123456
        shares, coeffs = split_shamir_secret(
            secret=secret,
            threshold=self.cfg.secure_shamir_threshold,
            num_shares=self.cfg.secure_num_share_holders,
            seed=7,
        )
        recovered = recover_shamir_secret(shares[:2], threshold=self.cfg.secure_shamir_threshold)
        self.assertEqual(recovered, secret)

        with self.assertRaises(ValueError):
            recover_shamir_secret(shares[:1], threshold=self.cfg.secure_shamir_threshold)

        summary = torch.tensor([0.1, -0.2, 0.3])
        summary_shares, commitment = self.protocol.create_summary_shares(summary, node_id=0, round_idx=1)
        self.assertTrue(verify_vss_share(summary_shares[0], commitment))

        tampered_values = list(summary_shares[0].values)
        tampered_values[0] += 1
        tampered_share = SummaryShare(
            holder_id=summary_shares[0].holder_id,
            x=summary_shares[0].x,
            values=tampered_values,
        )
        self.assertFalse(verify_vss_share(tampered_share, commitment))

        self.assertEqual(coeffs[0], secret)

    def test_binding_proof_detects_mismatched_summary(self):
        # A binding proof accepts only the summary derived from its update.
        update = {
            "shared_features.0.weight": torch.tensor([1.0, -2.0, 3.0]),
        }
        summary = self.protocol.state_dict_to_summary(update)
        proof = self.protocol.create_binding_proof(update, summary)
        self.assertTrue(self.protocol.verify_binding_proof(update, summary, proof))

        bad_summary = summary.clone()
        bad_summary[0] += 1.0
        self.assertFalse(self.protocol.verify_binding_proof(update, bad_summary, proof))

    def test_secure_aggregation_matches_average_state_dicts(self):
        # Share-domain aggregation must match the direct weighted result.
        states = [
            {"shared_features.0.weight": torch.tensor([1.0, 2.0]), "shared_features.0.bias": torch.tensor([0.5])},
            {"shared_features.0.weight": torch.tensor([3.0, 4.0]), "shared_features.0.bias": torch.tensor([1.5])},
            {"shared_features.0.weight": torch.tensor([5.0, 6.0]), "shared_features.0.bias": torch.tensor([2.5])},
        ]
        weights = [0.2, 0.3, 0.5]

        secure_avg = self.protocol.secure_aggregate_state_dicts(
            state_dicts=states,
            weights=weights,
            round_idx=0,
            receiver_node=0,
        )
        plain_avg = average_state_dicts(states, weights)

        for key in plain_avg:
            self.assertTrue(torch.allclose(secure_avg[key], plain_avg[key], rtol=1e-5, atol=1e-6))

    def test_committee_aggregates_updates_after_sharing(self):
        updates = [
            {"shared_features.0.weight": torch.tensor([0.5, -0.5])},
            {"shared_features.0.weight": torch.tensor([1.0, 2.0])},
            {"shared_features.0.weight": torch.tensor([-1.0, 3.0])},
        ]
        sender_nodes = [0, 1, 2]
        weights = [0.2, 0.3, 0.5]

        aggregated = self.protocol.secure_aggregate_update_dicts(
            update_dicts=updates,
            sender_nodes=sender_nodes,
            weights=weights,
            round_idx=4,
            receiver_node=1,
            valid_nodes=sender_nodes,
        )
        expected = average_state_dicts(updates, weights)

        self.assertTrue(torch.allclose(
            aggregated["shared_features.0.weight"],
            expected["shared_features.0.weight"],
            rtol=1e-5,
            atol=1e-6,
        ))
        record = self.protocol.committee_records[(4, 1)]
        self.assertEqual(len(record.request.contributor_nodes), 3)
        self.assertEqual(
            len(self.protocol.get_committee(receiver_node=1, round_idx=4).member_ids),
            3,
        )

    def test_conflicting_same_round_weight_request_is_rejected(self):
        updates = [
            {"shared_features.0.weight": torch.tensor([0.1])},
            {"shared_features.0.weight": torch.tensor([0.2])},
            {"shared_features.0.weight": torch.tensor([0.3])},
        ]
        sender_nodes = [0, 1, 2]

        first, first_status = self.protocol.try_secure_aggregate_update_dicts(
            update_dicts=updates,
            sender_nodes=sender_nodes,
            weights=[0.2, 0.3, 0.5],
            round_idx=6,
            receiver_node=0,
            valid_nodes=sender_nodes,
        )
        second, second_status = self.protocol.try_secure_aggregate_update_dicts(
            update_dicts=updates,
            sender_nodes=sender_nodes,
            weights=[0.4, 0.3, 0.3],
            round_idx=6,
            receiver_node=0,
            valid_nodes=sender_nodes,
        )

        self.assertIsNotNone(first)
        self.assertTrue(first_status["success"])
        self.assertIsNone(second)
        self.assertFalse(second_status["success"])
        self.assertIn("conflicting weight request", second_status["reason"])

    def test_exact_weight_request_retransmission_returns_cached_result(self):
        updates = [
            {"shared_features.0.weight": torch.tensor([0.1, -0.1])},
            {"shared_features.0.weight": torch.tensor([0.2, -0.2])},
            {"shared_features.0.weight": torch.tensor([0.3, -0.3])},
        ]
        kwargs = {
            "update_dicts": updates,
            "sender_nodes": [0, 1, 2],
            "weights": [0.2, 0.3, 0.5],
            "round_idx": 7,
            "receiver_node": 2,
            "valid_nodes": {0, 1, 2},
        }

        first = self.protocol.secure_aggregate_update_dicts(**kwargs)
        second = self.protocol.secure_aggregate_update_dicts(**kwargs)

        self.assertTrue(torch.equal(
            first["shared_features.0.weight"],
            second["shared_features.0.weight"],
        ))
        self.assertEqual(len(self.protocol.committee_records), 1)

    def test_committee_aborts_when_valid_contributors_are_insufficient(self):
        updates = [
            {"shared_features.0.weight": torch.tensor([0.1])},
            {"shared_features.0.weight": torch.tensor([0.2])},
        ]

        aggregated, status = self.protocol.try_secure_aggregate_update_dicts(
            update_dicts=updates,
            sender_nodes=[0, 1],
            weights=[0.5, 0.5],
            round_idx=8,
            receiver_node=0,
            valid_nodes={0, 1},
        )

        self.assertIsNone(aggregated)
        self.assertFalse(status["success"])
        self.assertIn("Not enough valid contributors", status["reason"])

    def test_mpc_friendly_apt_scores_penalize_far_neighbor(self):
        # A neighbor farther from the trusted reference receives a lower score.
        pre_states = {
            0: {"shared_features.0.weight": torch.zeros(16)},
            1: {"shared_features.0.weight": torch.zeros(16)},
            2: {"shared_features.0.weight": torch.zeros(16)},
        }
        uploaded_states = {
            0: {"shared_features.0.weight": torch.zeros(16)},
            1: {"shared_features.0.weight": torch.ones(16) * 0.01},
            2: {"shared_features.0.weight": torch.ones(16) * 10.0},
        }
        report = self.protocol.prepare_round(
            round_idx=0,
            uploaded_states=uploaded_states,
            pre_train_states=pre_states,
            in_neighbors={0: [1, 2], 1: [0, 2], 2: [0, 1]},
        )
        scores, _ = self.protocol.compute_mpc_friendly_apt_scores(
            receiver_node=0,
            candidate_nodes=[0, 1, 2],
            report=report,
        )

        self.assertGreater(scores[1], scores[2])
        self.assertGreaterEqual(scores[2], 0.0)
        self.assertLessEqual(scores[1], 1.0)

    def test_receiver_committee_binds_sender_and_share_summaries(self):
        pre_states = {
            node_id: {
                "shared_features.0.weight": torch.zeros(12),
                "shared_features.1.weight": torch.zeros(6),
            }
            for node_id in range(3)
        }
        uploaded_states = {
            node_id: {
                "shared_features.0.weight": torch.arange(
                    12,
                    dtype=torch.float32,
                ) * (node_id + 1) / 100.0,
                "shared_features.1.weight": torch.ones(6) * node_id / 10.0,
            }
            for node_id in range(3)
        }
        report = self.protocol.prepare_round(
            round_idx=2,
            uploaded_states=uploaded_states,
            pre_train_states=pre_states,
            in_neighbors={0: [1, 2], 1: [0, 2], 2: [0, 1]},
        )

        receiver_material = report.receiver_materials[1]
        self.assertEqual(receiver_material.valid_nodes, {0, 1, 2})
        self.assertEqual(len(receiver_material.share_commitments), 3)
        for material in receiver_material.materials.values():
            self.assertTrue(torch.allclose(
                material.summary,
                material.committee_summary,
                rtol=self.cfg.secure_binding_tolerance,
                atol=self.cfg.secure_binding_tolerance,
            ))

    def test_round_commitment_rejects_changed_high_dimensional_update(self):
        pre_states = {
            node_id: {
                "shared_features.0.weight": torch.zeros(4),
            }
            for node_id in range(3)
        }
        uploaded_states = {
            0: {"shared_features.0.weight": torch.tensor([0.1, 0.2, 0.3, 0.4])},
            1: {"shared_features.0.weight": torch.tensor([0.2, 0.3, 0.4, 0.5])},
            2: {"shared_features.0.weight": torch.tensor([0.3, 0.4, 0.5, 0.6])},
        }
        report = self.protocol.prepare_round(
            round_idx=3,
            uploaded_states=uploaded_states,
            pre_train_states=pre_states,
            in_neighbors={0: [1, 2], 1: [0, 2], 2: [0, 1]},
        )
        updates = [
            {
                key: uploaded_states[node_id][key] - pre_states[node_id][key]
                for key in uploaded_states[node_id]
            }
            for node_id in range(3)
        ]
        changed_updates = [
            {key: value.clone() for key, value in update.items()}
            for update in updates
        ]
        changed_updates[2]["shared_features.0.weight"][0] += 1.0

        aggregated, status = self.protocol.try_secure_aggregate_update_dicts(
            update_dicts=changed_updates,
            sender_nodes=[0, 1, 2],
            weights=[0.2, 0.3, 0.5],
            round_idx=3,
            receiver_node=0,
            valid_nodes={0, 1, 2},
            round_report=report,
        )
        self.assertIsNone(aggregated)
        self.assertFalse(status["success"])
        self.assertIn("committed update", status["reason"])

        aggregated, status = self.protocol.try_secure_aggregate_update_dicts(
            update_dicts=updates,
            sender_nodes=[0, 1, 2],
            weights=[0.2, 0.3, 0.5],
            round_idx=3,
            receiver_node=0,
            valid_nodes={0, 1, 2},
            round_report=report,
        )
        expected = average_state_dicts(updates, [0.2, 0.3, 0.5])
        self.assertTrue(status["success"])
        self.assertTrue(torch.allclose(
            aggregated["shared_features.0.weight"],
            expected["shared_features.0.weight"],
            rtol=1e-5,
            atol=1e-6,
        ))


if __name__ == "__main__":
    unittest.main()
