from dataclasses import dataclass


@dataclass
class Config:
    # Runtime
    seed: int = 42
    device: str = "cuda"  # "cuda" or "cpu"

    # Dataset
    dataset_root: str = "./data"
    dataset_name: str = "cifar10"
    num_classes: int = 10
    image_size: int = 32
    val_ratio: float = 0.2

    # Decentralized training
    num_nodes: int = 50
    rounds: int = 100
    local_epochs: int = 1
    batch_size: int = 64
    lr: float = 0.03
    momentum: float = 0.9
    weight_decay: float = 5e-4
    use_scheduler: bool = True
    min_lr: float = 1e-5
    # True shares only the backbone; False shares the complete model.
    use_personalization: bool = True

    # Data partition
    partition_mode: str = "noniid"  # "iid" or "noniid"
    dirichlet_alpha: float = 0.5

    # Dynamic directed graph
    graph_mode: str = "dynamic"
    min_out_degree: int = 6
    max_out_degree: int = 12
    self_weight: float = 0.3

    # Evaluation and output
    eval_every: int = 1
    target_acc: float = 0.80
    target_metric: str = "honest_mean_local_acc"
    output_root: str = "./outputs"
    run_name: str = ""
    participation_rate: float = 1.0

    # Optional double-threshold parameters
    use_double_threshold: bool = False
    sim_low_th: float = -0.2
    sim_high_th: float = 0.05
    norm_ratio_th: float = 3.0
    suspicious_scale: float = 0.6
    double_threshold_start_round: int = 20
    log_double_threshold: bool = True
    log_double_threshold_every: int = 1

    # Continuous historical trust
    use_trust_score: bool = True
    trust_beta: float = 0.90
    # Punishment uses a shorter memory than recovery.
    trust_punish_beta: float = 0.60
    trust_reward_beta: float = 0.95
    trust_init: float = 0.60
    trust_min: float = 0.05
    trust_max: float = 0.995
    trust_instant_mix: float = 0.50
    trust_distance_temperature: float = 0.35
    trust_weight_power: float = 1.50
    log_trust: bool = True

    # Adaptive Percentile Thresholding (APT)
    use_apt: bool = True
    apt_start_round: int = 5
    apt_accept_quantile: float = 0.40
    apt_reject_quantile: float = 0.85
    apt_weight_power: float = 1.20
    apt_hard_reject_score: float = 0.20
    log_apt: bool = True

    # Multi-view distance
    ema_momentum_beta: float = 0.90
    use_cosine_distance: bool = True
    use_norm_ratio_distance: bool = True
    use_ema_momentum_distance: bool = True
    use_layerwise_distance: bool = True

    # Enabled view weights are normalized by the trainer.
    dist_weight_cosine: float = 0.25
    dist_weight_norm_ratio: float = 0.25
    dist_weight_momentum: float = 0.25
    dist_weight_layerwise: float = 0.25

    # Update clipping and per-neighbor weight cap
    max_update_norm: float = 5.0
    max_neighbor_weight: float = 0.15

    # Median absolute deviation (MAD) controls
    use_mad_clipping: bool = True
    mad_clip_k: float = 2.5
    use_historical_mad_apt: bool = True
    historical_mad_apt_start_round: int = 10
    historical_mad_window: int = 10
    historical_mad_k: float = 2.5
    historical_mad_weight_power: float = 2.0

    # Aggregator used when committee aggregation is disabled.
    aggregation_rule: str = "weighted_trimmed_mean"
    trim_ratio: float = 0.1

    # Byzantine attack evaluation
    use_malicious: bool = False
    malicious_ratio: float = 0.2
    attack_type: str = "sign_flip"
    attack_start_round: int = 20
    attack_scale: float = 2.0
    noise_std: float = 0.5
    malicious_only_when_active: bool = True

    # Multiple enabled attacks are applied in the order defined in attacks.py.
    use_scaling_attack: bool = False
    use_random_update_attack: bool = False
    use_alie_attack: bool = False
    use_slow_drift_attack: bool = False
    use_layerwise_sign_flip_attack: bool = False

    scaling_factor: float = 10.0

    random_update_scale: float = 1.0
    random_update_normalize_to_honest_update: bool = True

    alie_z: float = 1.0
    alie_use_all_updates: bool = False

    slow_drift_scale: float = 0.05
    slow_drift_max_scale: float = 1.0

    layerwise_sign_flip_scale: float = 2.0
    layerwise_sign_flip_layers: str = ""

    # Committee privacy and protected aggregation
    use_secure_privacy_module: bool = False
    secure_num_share_holders: int = 3
    secure_shamir_threshold: int = 2
    secure_min_valid_contributors: int = 3
    secure_global_update_lr: float = 1.0
    secure_binding_tolerance: float = 1e-5
    secure_summary_dim: int = 128
    secure_projection_seed: int = 2026
    secure_update_norm_bound: float = 1e9
    secure_summary_norm_bound: float = 1e9
    secure_use_mpc_friendly_apt: bool = True
    secure_small_degree_temperature: float = 0.35
    secure_strict_validation: bool = True
    log_secure_protocol: bool = True

    # Optional experiment tracking
    use_wandb: bool = False
    wandb_project: str = "decentralized-fl"
    wandb_entity: str = ""
    wandb_run_name: str = ""
    wandb_mode: str = "online"
