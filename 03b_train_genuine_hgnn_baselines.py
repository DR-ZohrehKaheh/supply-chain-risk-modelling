# Project supervisor: Dr. Zohreh Kaheh.
# Undergraduate project collaborator: Yuyang Zhang.
# Initial student coding was developed with supervisory teaching and improvement guidance.
# -*- coding: utf-8 -*-
"""Technical implementation for the supply-chain code portfolio."""

from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import random

import numpy as np
import pandas as pd

import torch
import os
torch.set_num_threads(int(os.environ.get("PORTFOLIO_TORCH_THREADS", "1")))
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# 0. Paths, legacy utilities, and reconstructed settings
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
LEGACY_STAGE3_SCRIPT = BASE_DIR / "03_train_raht_residual_hazard.py"


def _load_legacy_stage3():
    if not LEGACY_STAGE3_SCRIPT.exists():
        raise FileNotFoundError(LEGACY_STAGE3_SCRIPT)
    spec = importlib.util.spec_from_file_location(
        "retained_stage3_relation_neighbour",
        LEGACY_STAGE3_SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_legacy = _load_legacy_stage3()

PROTOCOL_FILE = BASE_DIR / "00_raht_protocol_locked.json"
PACKAGE_DIR = BASE_DIR / "raht_training_package"
SELF_DIR = BASE_DIR / "self_hazard_baseline_results"
SELF_CHECKPOINT_DIR = SELF_DIR / "checkpoints"

OUT_DIR = BASE_DIR / "hgnn_ablation_results"
CHECKPOINT_DIR = OUT_DIR / "checkpoints"
PREDICTION_DIR = OUT_DIR / "predictions"

INPUT_DIM = 8
HAZARD_DAYS = 3
LOOKBACK_DAYS = 14

BATCH_SIZE = 256
MAX_EPOCHS = 140
MIN_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 18
GRADIENT_CLIP_NORM = 5.0
TOP_ALERT_RATE = 0.10
NUM_WORKERS = 0

LEARNING_RATE = 0.0003
WEIGHT_DECAY = 0.001
RELATION_DROPOUT = 0.20
GATE_L1_WEIGHT = 0.001
DELTA_L2_WEIGHT = 0.0001
ANCHOR_LOSS_WEIGHT = 0.001
RANKING_LOSS_WEIGHT = 0.05
INITIAL_GATE_BIAS = -2.0

BOOTSTRAP_BLOCK_LENGTH = 7
BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_RANDOM_STATE = 20260810

MODEL_NAMES = [
    "HGNN-Gated-Residual",
    "HGNN-Residual-NoGate",
    "HGNN-Gated-NoResidual",
]


# Reuse retained, checkpoint-compatible utilities.
SelfGRUHazardModel = _legacy.SelfGRUHazardModel
set_global_seed = _legacy.set_global_seed
sha256_file = _legacy.sha256_file
safe_torch_load = _legacy.safe_torch_load
safe_average_precision = _legacy.safe_average_precision
safe_roc_auc = _legacy.safe_roc_auc
topk_metrics = _legacy.topk_metrics
evaluate_risk_predictions = _legacy.evaluate_risk_predictions
hazards_to_three_day_risk_torch = _legacy.hazards_to_three_day_risk_torch
load_locked_protocol = _legacy.load_locked_protocol
load_package = _legacy.load_package
load_stage2_normalization = _legacy.load_stage2_normalization
load_raw_temporal_tensor = _legacy.load_raw_temporal_tensor
load_frozen_self_model = _legacy.load_frozen_self_model
calculate_hazard_positive_weights = _legacy.calculate_hazard_positive_weights
masked_hazard_loss = _legacy.masked_hazard_loss
pairwise_ranking_loss = _legacy.pairwise_ranking_loss


def save_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def count_trainable_parameters(model):
    return int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
    )


def transform_sample_tensor(sample_tensor, normalizer):
    """Apply the frozen Stage-2 log1p/z-score transform to [S,L,F]."""
    return _legacy.transform_with_stage2_normalizer(
        sample_tensor,
        normalizer,
    )


def transform_all_node_tensor(node_tensor, normalizer):
    """Apply the frozen Stage-2 transform to an all-node [T,N,F] tensor."""
    values = np.asarray(node_tensor, dtype=np.float32)
    if values.ndim != 3 or values.shape[2] != INPUT_DIM:
        raise ValueError(
            "node_tensor must have shape [time, node, 8]; "
            f"received {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("node_tensor contains non-finite values")
    if (values < 0).any():
        raise ValueError("negative operational values cannot be log1p transformed")

    transformed = np.log1p(values)
    mean = np.asarray(normalizer["mean"], dtype=np.float32)[None, None, :]
    std = np.asarray(normalizer["std"], dtype=np.float32)[None, None, :]
    if (std <= 0).any():
        raise ValueError("normalisation standard deviations must be positive")
    return ((transformed - mean) / std).astype(np.float32)


# ============================================================
# 1. Incidence operator and sequence construction
# ============================================================

def validate_incidence_matrix(H, relation_name="relation"):
    matrix = np.asarray(H, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"{relation_name}: incidence matrix must be two-dimensional")
    if matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{relation_name}: incidence matrix is empty")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{relation_name}: incidence matrix contains non-finite values")
    if not np.isin(matrix, [0.0, 1.0]).all():
        raise ValueError(f"{relation_name}: incidence matrix must be binary")

    node_degree = matrix.sum(axis=1)
    edge_degree = matrix.sum(axis=0)
    # Isolated nodes are valid: their relational context is the all-zero
    # vector, while their focal history remains available through Self-GRU.
    # The count is recorded in the package's structure summary and manifests.
    if (edge_degree <= 0).any():
        bad = np.where(edge_degree <= 0)[0].tolist()
        raise ValueError(f"{relation_name}: empty hyperedges at columns {bad}")
    return matrix


def standard_incidence_operator(H):
    """Return the conventional self-inclusive HGNN operator for auditing."""
    matrix = validate_incidence_matrix(H)
    node_degree = matrix.sum(axis=1)
    edge_degree = matrix.sum(axis=0)
    inverse_node_sqrt = np.zeros_like(node_degree, dtype=np.float32)
    nonzero = node_degree > 0
    inverse_node_sqrt[nonzero] = np.power(node_degree[nonzero], -0.5)
    dv_inv_sqrt = np.diag(inverse_node_sqrt)
    de_inv = np.diag(np.power(edge_degree, -1.0))
    operator = dv_inv_sqrt @ matrix @ de_inv @ matrix.T @ dv_inv_sqrt
    return operator.astype(np.float32)


def focal_excluded_incidence_operator(H):
    """Return a row-normalised leave-one-node-out incidence operator.

    For focal node ``i``, each incident non-singleton hyperedge first averages
    the *other* nodes in that hyperedge.  The focal context is then the mean of
    those hyperedge-specific averages.  Consequently the diagonal is exactly
    zero and changing a focal node's own features cannot change its relational
    context.  Nodes belonging only to singleton hyperedges receive zero
    context rather than a leaked copy of their own features.

    With edge size ``d_e`` and ``m_i`` eligible incident hyperedges,

        P_ij = 1[i != j] / m_i * sum_e H_ie H_je / (d_e - 1).
    """
    matrix = validate_incidence_matrix(H)
    edge_degree = matrix.sum(axis=0).astype(np.float32)
    eligible = edge_degree > 1.0

    operator = np.zeros(
        (matrix.shape[0], matrix.shape[0]),
        dtype=np.float32,
    )
    if not np.any(eligible):
        return operator

    eligible_incidence = matrix[:, eligible]
    inverse_other_count = 1.0 / (edge_degree[eligible] - 1.0)
    operator = (
        eligible_incidence
        * inverse_other_count[None, :]
    ) @ eligible_incidence.T
    np.fill_diagonal(operator, 0.0)

    eligible_edge_count = eligible_incidence.sum(axis=1)
    nonzero = eligible_edge_count > 0.0
    operator[nonzero] /= eligible_edge_count[nonzero, None]
    return operator.astype(np.float32)


def incidence_structure_summary(H):
    """Return JSON-serialisable coverage and hyperedge-size diagnostics."""
    matrix = validate_incidence_matrix(H)
    node_degree = matrix.sum(axis=1)
    edge_degree = matrix.sum(axis=0)
    return {
        "node_count": int(matrix.shape[0]),
        "hyperedge_count": int(matrix.shape[1]),
        "covered_node_count": int(np.sum(node_degree > 0)),
        "isolated_node_indices": np.where(node_degree == 0)[0].astype(int).tolist(),
        "singleton_hyperedge_count": int(np.sum(edge_degree == 1)),
        "minimum_hyperedge_size": float(np.min(edge_degree)),
        "median_hyperedge_size": float(np.median(edge_degree)),
        "maximum_hyperedge_size": float(np.max(edge_degree)),
    }


def focal_exclusion_invariants_hold(H):
    """Check zero diagonal, active-row sums, and isolated-row behaviour."""
    matrix = validate_incidence_matrix(H)
    operator = focal_excluded_incidence_operator(matrix)
    eligible_edges = matrix[:, matrix.sum(axis=0) > 1]
    active = eligible_edges.sum(axis=1) > 0
    return bool(
        np.allclose(np.diag(operator), 0.0)
        and np.allclose(operator[active].sum(axis=1), 1.0)
        and np.allclose(operator[~active], 0.0)
    )


def genuine_node_hyperedge_node_propagation(node_tensor, H):
    """Propagate [T,N,F] features through focal-excluded hyperedges.

    The fixed operator averages the other nodes within each incident
    hyperedge and then averages across the focal node's eligible hyperedges.
    Learning occurs in the relation-specific feature projection and GRU.
    """
    values = np.asarray(node_tensor, dtype=np.float32)
    matrix = validate_incidence_matrix(H)
    if values.ndim != 3:
        raise ValueError("node_tensor must have shape [time, node, feature]")
    if values.shape[1] != matrix.shape[0]:
        raise ValueError(
            "node count mismatch: "
            f"tensor has {values.shape[1]}, incidence has {matrix.shape[0]}"
        )

    returned = np.einsum(
        "nm,tmf->tnf",
        focal_excluded_incidence_operator(matrix),
        values,
        optimize=True,
    )
    return returned.astype(np.float32)


def build_target_relation_sequences(
    metadata,
    propagated_tensor,
    selected_indices,
    lookback_days=LOOKBACK_DAYS,
):
    metadata = metadata.reset_index(drop=True)
    propagated_tensor = np.asarray(propagated_tensor, dtype=np.float32)
    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    feature_dim = int(propagated_tensor.shape[2])

    sequences = np.zeros(
        (len(selected_indices), int(lookback_days), feature_dim),
        dtype=np.float32,
    )
    for output_position, sample_index in enumerate(selected_indices):
        row = metadata.iloc[int(sample_index)]
        node_index = int(row["node_index"])
        raw_time_index = int(row["raw_time_index"])
        start = raw_time_index - int(lookback_days) + 1
        end = raw_time_index + 1
        if start < 0 or end > propagated_tensor.shape[0]:
            raise IndexError(
                f"sample {sample_index}: relation history lies outside tensor"
            )
        window = propagated_tensor[start:end, node_index, :]
        if window.shape != (int(lookback_days), feature_dim):
            raise RuntimeError(
                f"sample {sample_index}: unexpected window shape {window.shape}"
            )
        sequences[output_position] = window
    return sequences


# ============================================================
# 2. Reconstructed HGNN architectures
# ============================================================

class RelationTemporalBranch(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.feature_projection = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, relation_sequence):
        projected = self.feature_projection(relation_sequence)
        projected = self.input_norm(projected)
        projected = F.gelu(projected)
        projected = self.dropout(projected)
        _, hidden = self.gru(projected)
        representation = self.output_norm(hidden[-1])
        return self.dropout(representation)


class _HGNNBase(nn.Module):
    def __init__(
        self,
        frozen_self_model,
        hidden_dim,
        dropout,
        use_dynamic_gate=True,
        use_residual_anchor=True,
    ):
        super().__init__()
        self.self_model = frozen_self_model
        for parameter in self.self_model.parameters():
            parameter.requires_grad = False

        self.group_branch = RelationTemporalBranch(INPUT_DIM, hidden_dim, dropout)
        self.storage_branch = RelationTemporalBranch(INPUT_DIM, hidden_dim, dropout)
        self.use_dynamic_gate = bool(use_dynamic_gate)
        self.use_residual_anchor = bool(use_residual_anchor)

        if self.use_dynamic_gate:
            self.gate_layer = nn.Linear(hidden_dim * 3, 2)
            nn.init.zeros_(self.gate_layer.weight)
            nn.init.constant_(self.gate_layer.bias, INITIAL_GATE_BIAS)
        else:
            self.gate_layer = None

        self.group_delta_head = nn.Linear(hidden_dim, HAZARD_DAYS)
        self.storage_delta_head = nn.Linear(hidden_dim, HAZARD_DAYS)
        nn.init.zeros_(self.group_delta_head.weight)
        nn.init.zeros_(self.group_delta_head.bias)
        nn.init.zeros_(self.storage_delta_head.weight)
        nn.init.zeros_(self.storage_delta_head.bias)

    def forward(self, x_self, x_group, x_storage):
        with torch.no_grad():
            self_representation = self.self_model.encode(x_self)
            base_logits = self.self_model(x_self)

        group_representation = self.group_branch(x_group)
        storage_representation = self.storage_branch(x_storage)

        if self.use_dynamic_gate:
            gate_input = torch.cat(
                [
                    self_representation,
                    group_representation,
                    storage_representation,
                ],
                dim=1,
            )
            gates = torch.sigmoid(self.gate_layer(gate_input))
            gate_group = gates[:, 0:1]
            gate_storage = gates[:, 1:2]
        else:
            gate_group = torch.ones(
                (x_self.shape[0], 1), dtype=x_self.dtype, device=x_self.device
            )
            gate_storage = torch.ones_like(gate_group)

        delta_group = self.group_delta_head(group_representation)
        delta_storage = self.storage_delta_head(storage_representation)

        if self.use_residual_anchor:
            final_logits = (
                base_logits
                + gate_group * delta_group
                + gate_storage * delta_storage
            )
        else:
            # Gated relation-logit fusion with no Self-GRU logit addition.
            final_logits = (
                gate_group * delta_group
                + gate_storage * delta_storage
            )

        return {
            "final_logits": final_logits,
            "base_logits": base_logits,
            "gate_group": gate_group,
            "gate_storage": gate_storage,
            "delta_group": delta_group,
            "delta_storage": delta_storage,
        }


class HGNNGatedResidualModel(_HGNNBase):
    def __init__(self, frozen_self_model, hidden_dim, dropout):
        super().__init__(
            frozen_self_model,
            hidden_dim,
            dropout,
            use_dynamic_gate=True,
            use_residual_anchor=True,
        )


class HGNNResidualNoGateModel(_HGNNBase):
    def __init__(self, frozen_self_model, hidden_dim, dropout):
        super().__init__(
            frozen_self_model,
            hidden_dim,
            dropout,
            use_dynamic_gate=False,
            use_residual_anchor=True,
        )


class HGNNGatedNoResidualModel(_HGNNBase):
    def __init__(self, frozen_self_model, hidden_dim, dropout):
        super().__init__(
            frozen_self_model,
            hidden_dim,
            dropout,
            use_dynamic_gate=True,
            use_residual_anchor=False,
        )


def build_model(model_name, frozen_self_model, hidden_dim, dropout):
    if model_name == "HGNN-Gated-Residual":
        return HGNNGatedResidualModel(frozen_self_model, hidden_dim, dropout)
    if model_name == "HGNN-Residual-NoGate":
        return HGNNResidualNoGateModel(frozen_self_model, hidden_dim, dropout)
    if model_name == "HGNN-Gated-NoResidual":
        return HGNNGatedNoResidualModel(frozen_self_model, hidden_dim, dropout)
    raise ValueError(f"unknown Stage 3b model: {model_name}")


# ============================================================
# 3. Loss, prediction, and descriptive moving-block resampling
# ============================================================

def model_training_loss(
    model_name,
    outputs,
    hazard_targets,
    hazard_mask,
    y_three_day,
    positive_weights,
):
    final_logits = outputs["final_logits"]
    base_logits = outputs["base_logits"]
    hazard_loss = masked_hazard_loss(
        final_logits,
        hazard_targets,
        hazard_mask,
        positive_weights,
    )
    risk = hazards_to_three_day_risk_torch(torch.sigmoid(final_logits))
    ranking_loss = pairwise_ranking_loss(risk, y_three_day)

    if model_name == "HGNN-Residual-NoGate":
        gate_l1 = torch.zeros((), dtype=final_logits.dtype, device=final_logits.device)
    else:
        gate_l1 = (
            outputs["gate_group"].mean()
            + outputs["gate_storage"].mean()
        )
    delta_l2 = (
        outputs["delta_group"].pow(2).mean()
        + outputs["delta_storage"].pow(2).mean()
    )
    if model_name == "HGNN-Gated-NoResidual":
        anchor_loss = torch.zeros(
            (), dtype=final_logits.dtype, device=final_logits.device
        )
    else:
        anchor_loss = F.mse_loss(final_logits, base_logits.detach())
    total = (
        hazard_loss
        + RANKING_LOSS_WEIGHT * ranking_loss
        + GATE_L1_WEIGHT * gate_l1
        + DELTA_L2_WEIGHT * delta_l2
        + ANCHOR_LOSS_WEIGHT * anchor_loss
    )
    return {
        "total_loss": total,
        "hazard_loss": hazard_loss.detach(),
        "ranking_loss": ranking_loss.detach(),
        "gate_l1": gate_l1.detach(),
        "delta_l2": delta_l2.detach(),
        "anchor_loss": anchor_loss.detach(),
    }


def predict_model(model, X_self, X_group, X_storage, device):
    model.eval()
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(X_self)).float(),
        torch.from_numpy(np.asarray(X_group)).float(),
        torch.from_numpy(np.asarray(X_storage)).float(),
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    collected = {
        "hazards": [],
        "base_hazards": [],
        "risk": [],
        "base_risk": [],
        "gate_group": [],
        "gate_storage": [],
        "delta_group": [],
        "delta_storage": [],
    }
    with torch.no_grad():
        for xb, xg, xs in loader:
            outputs = model(xb.to(device), xg.to(device), xs.to(device))
            hazards = torch.sigmoid(outputs["final_logits"])
            base_hazards = torch.sigmoid(outputs["base_logits"])
            collected["hazards"].append(hazards.cpu().numpy())
            collected["base_hazards"].append(base_hazards.cpu().numpy())
            collected["risk"].append(
                hazards_to_three_day_risk_torch(hazards).cpu().numpy()
            )
            collected["base_risk"].append(
                hazards_to_three_day_risk_torch(base_hazards).cpu().numpy()
            )
            collected["gate_group"].append(
                outputs["gate_group"].cpu().numpy().reshape(-1)
            )
            collected["gate_storage"].append(
                outputs["gate_storage"].cpu().numpy().reshape(-1)
            )
            collected["delta_group"].append(outputs["delta_group"].cpu().numpy())
            collected["delta_storage"].append(outputs["delta_storage"].cpu().numpy())
    return {
        key: np.concatenate(value, axis=0)
        for key, value in collected.items()
    }


def moving_block_bootstrap_improvement(
    metadata,
    y_true,
    baseline_score,
    candidate_score,
    metric_fn,
    block_length=BOOTSTRAP_BLOCK_LENGTH,
    n_bootstrap=BOOTSTRAP_ITERATIONS,
    random_state=BOOTSTRAP_RANDOM_STATE,
):
    """Descriptive paired moving-block resampling over unique dates.

    The returned non-positive fraction is deliberately not named a p-value.
    It comes from the ordinary bootstrap distribution, not a null-centred
    resampling scheme.
    """
    metadata = metadata.reset_index(drop=True).copy()
    y_true = np.asarray(y_true, dtype=int)
    baseline_score = np.asarray(baseline_score, dtype=float)
    candidate_score = np.asarray(candidate_score, dtype=float)
    if not (
        len(metadata)
        == len(y_true)
        == len(baseline_score)
        == len(candidate_score)
    ):
        raise ValueError("metadata, labels, and scores must have equal length")

    date_values = pd.to_datetime(metadata["date"], errors="raise")
    unique_dates = pd.Index(sorted(date_values.unique()))
    number_of_dates = len(unique_dates)
    if number_of_dates < 2:
        raise ValueError("moving-block resampling requires at least two dates")
    block_length = int(max(1, min(int(block_length), number_of_dates)))
    number_of_blocks = int(math.ceil(number_of_dates / block_length))
    positions_by_date = {
        date: np.where(date_values.to_numpy() == np.datetime64(date))[0]
        for date in unique_dates
    }

    rng = np.random.default_rng(int(random_state))
    differences = []
    for _ in range(int(n_bootstrap)):
        starts = rng.integers(0, number_of_dates, size=number_of_blocks)
        sampled_date_positions = []
        for start in starts:
            sampled_date_positions.extend(
                (start + offset) % number_of_dates
                for offset in range(block_length)
            )
        sampled_date_positions = sampled_date_positions[:number_of_dates]
        sampled_indices = np.concatenate(
            [positions_by_date[unique_dates[position]] for position in sampled_date_positions]
        )
        y_boot = y_true[sampled_indices]
        if np.unique(y_boot).size < 2:
            continue
        baseline_metric = metric_fn(y_boot, baseline_score[sampled_indices])
        candidate_metric = metric_fn(y_boot, candidate_score[sampled_indices])
        if np.isfinite(baseline_metric) and np.isfinite(candidate_metric):
            differences.append(float(candidate_metric - baseline_metric))

    differences = np.asarray(differences, dtype=float)
    if differences.size == 0:
        return {
            "mean_improvement": np.nan,
            "ci_lower_95": np.nan,
            "ci_upper_95": np.nan,
            "probability_improvement_gt_0": np.nan,
            "nonpositive_resample_fraction": np.nan,
            "completed_iterations": 0,
            "bootstrap_valid_iterations": 0,
        }
    return {
        "mean_improvement": float(differences.mean()),
        "ci_lower_95": float(np.quantile(differences, 0.025)),
        "ci_upper_95": float(np.quantile(differences, 0.975)),
        "probability_improvement_gt_0": float((differences > 0).mean()),
        "nonpositive_resample_fraction": float(
            (1 + np.sum(differences <= 0)) / (differences.size + 1)
        ),
        "completed_iterations": int(differences.size),
        "bootstrap_valid_iterations": int(differences.size),
    }


# ============================================================
# 4. Training and Stage-3b outputs
# ============================================================

def _training_loader(arrays, seed):
    dataset = TensorDataset(
        torch.from_numpy(arrays["X_self_train"]).float(),
        torch.from_numpy(arrays["X_group_train"]).float(),
        torch.from_numpy(arrays["X_storage_train"]).float(),
        torch.from_numpy(arrays["y_hazard_train"]).float(),
        torch.from_numpy(arrays["hazard_mask_train"]).float(),
        torch.from_numpy(arrays["y_3_day_train"].astype(np.float32)).float(),
    )
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        generator=generator,
        drop_last=False,
    )


def _train_one_seed(model_name, seed, arrays, y_validation, device):
    set_global_seed(int(seed))
    frozen_self, self_checkpoint = load_frozen_self_model(seed, device)
    hidden_dim = int(self_checkpoint["config"]["hidden_dim"])
    model = build_model(
        model_name,
        frozen_self,
        hidden_dim,
        RELATION_DROPOUT,
    ).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    positive_weights_np = calculate_hazard_positive_weights(
        arrays["y_hazard_train"], arrays["hazard_mask_train"]
    )
    positive_weights = torch.tensor(
        positive_weights_np, dtype=torch.float32, device=device
    )
    loader = _training_loader(arrays, seed)

    best_ap = -np.inf
    best_epoch = 0
    best_state = None
    best_prediction = None
    best_metrics = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        model.self_model.eval()
        loss_sum = 0.0
        mask_sum = 0.0
        for xb, xg, xs, yb, mb, y3b in loader:
            xb, xg, xs = xb.to(device), xg.to(device), xs.to(device)
            yb, mb, y3b = yb.to(device), mb.to(device), y3b.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(xb, xg, xs)
            losses = model_training_loss(
                model_name, outputs, yb, mb, y3b, positive_weights
            )
            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(trainable, GRADIENT_CLIP_NORM)
            optimizer.step()
            current_mask = float(mb.sum().item())
            loss_sum += float(losses["total_loss"].item()) * current_mask
            mask_sum += current_mask

        validation_prediction = predict_model(
            model,
            arrays["X_self_validation"],
            arrays["X_group_validation"],
            arrays["X_storage_validation"],
            device,
        )
        metrics = evaluate_risk_predictions(
            y_validation, validation_prediction["risk"]
        )
        validation_ap = float(metrics["average_precision"])
        history.append(
            {
                "model": model_name,
                "seed": int(seed),
                "epoch": int(epoch),
                "training_total_loss": loss_sum / max(mask_sum, 1.0),
                "validation_average_precision": validation_ap,
                "validation_roc_auc": metrics["roc_auc"],
            }
        )
        if np.isfinite(validation_ap) and validation_ap > best_ap + 1e-8:
            best_ap = validation_ap
            best_epoch = int(epoch)
            best_state = copy.deepcopy(model.state_dict())
            best_prediction = {
                key: value.copy() for key, value in validation_prediction.items()
            }
            best_metrics = dict(metrics)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if (
            epoch >= MIN_EPOCHS
            and epochs_without_improvement >= EARLY_STOPPING_PATIENCE
        ):
            break

    if best_state is None:
        raise RuntimeError(f"no valid checkpoint for {model_name}, seed {seed}")
    model.load_state_dict(best_state)
    return {
        "model": model,
        "hidden_dim": hidden_dim,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "best_prediction": best_prediction,
        "history": pd.DataFrame(history),
        "positive_weights": positive_weights_np,
        "self_checkpoint_config": self_checkpoint["config"],
        "trainable_parameter_count": count_trainable_parameters(model),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    protocol, protocol_hash = load_locked_protocol()
    package = load_package()
    normalizer = load_stage2_normalization()
    if package["package_protocol_hash"] != protocol_hash:
        raise RuntimeError("Stage-1 package hash does not match locked protocol")

    seeds = [int(seed) for seed in protocol["random_seeds"]]
    train_indices = package["train_indices"]
    validation_indices = package["validation_indices"]
    metadata = package["metadata"].reset_index(drop=True)
    node_table = (
        metadata[["node_index", "node"]]
        .drop_duplicates()
        .sort_values("node_index")
    )
    nodes = node_table["node"].astype(str).tolist()
    raw_tensor, raw_dates, raw_paths = load_raw_temporal_tensor(nodes)
    raw_normalized = transform_all_node_tensor(raw_tensor, normalizer)
    X_self_all = transform_sample_tensor(package["X_self"], normalizer)

    H_group = validate_incidence_matrix(
        package["H_product_group"], "Product Group"
    )
    H_storage = validate_incidence_matrix(
        package["H_storage_location"], "Storage Location"
    )
    group_tensor = genuine_node_hyperedge_node_propagation(
        raw_normalized, H_group
    )
    storage_tensor = genuine_node_hyperedge_node_propagation(
        raw_normalized, H_storage
    )
    X_group_train = build_target_relation_sequences(
        metadata, group_tensor, train_indices
    )
    X_storage_train = build_target_relation_sequences(
        metadata, storage_tensor, train_indices
    )
    X_group_validation = build_target_relation_sequences(
        metadata, group_tensor, validation_indices
    )
    X_storage_validation = build_target_relation_sequences(
        metadata, storage_tensor, validation_indices
    )
    arrays = {
        "X_self_train": X_self_all[train_indices],
        "X_group_train": X_group_train,
        "X_storage_train": X_storage_train,
        "y_hazard_train": package["y_hazard"][train_indices],
        "hazard_mask_train": package["hazard_mask"][train_indices],
        "y_3_day_train": package["y_3_day"][train_indices],
        "X_self_validation": X_self_all[validation_indices],
        "X_group_validation": X_group_validation,
        "X_storage_validation": X_storage_validation,
    }
    y_validation = package["y_3_day"][validation_indices].astype(int)
    validation_metadata = metadata.iloc[validation_indices].copy().reset_index(drop=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    histories = []
    metrics_rows = []
    parameter_rows = []
    predictions_by_model = {name: [] for name in MODEL_NAMES}
    hazards_by_model = {name: [] for name in MODEL_NAMES}
    base_risk_by_seed = []

    for seed in seeds:
        for model_name in MODEL_NAMES:
            result = _train_one_seed(
                model_name, seed, arrays, y_validation, device
            )
            prediction = result["best_prediction"]
            metrics = result["best_metrics"]
            histories.append(result["history"])
            predictions_by_model[model_name].append(prediction["risk"])
            hazards_by_model[model_name].append(prediction["hazards"])
            if model_name == "HGNN-Gated-Residual":
                base_risk_by_seed.append(prediction["base_risk"])

            parameter_count = int(result["trainable_parameter_count"])
            metrics_rows.append(
                {
                    "seed": seed,
                    "model": model_name,
                    "best_epoch": result["best_epoch"],
                    "validation_average_precision": metrics["average_precision"],
                    "validation_roc_auc": metrics["roc_auc"],
                    "validation_precision_at_top10": metrics["precision_at_top10"],
                    "validation_recall_at_top10": metrics["recall_at_top10"],
                    "validation_lift_at_top10": metrics["lift_at_top10"],
                    "trainable_parameters": parameter_count,
                }
            )
            parameter_rows.append(
                {
                    "seed": seed,
                    "model": model_name,
                    "hidden_dim": result["hidden_dim"],
                    "trainable_parameters": parameter_count,
                }
            )

            model_directory = CHECKPOINT_DIR / model_name.replace("-", "_")
            model_directory.mkdir(parents=True, exist_ok=True)
            checkpoint = {
                "stage": "Stage 3b",
                "provenance": "reference implementation",
                "model": model_name,
                "seed": seed,
                "hidden_dim": result["hidden_dim"],
                "best_epoch": result["best_epoch"],
                "validation_metrics": metrics,
                "trainable_parameter_count": parameter_count,
                "positive_weights": result["positive_weights"].tolist(),
                "self_checkpoint_config": result["self_checkpoint_config"],
                "state_dict": result["model"].state_dict(),
            }
            torch.save(checkpoint, model_directory / f"seed_{seed}.pt")

    full_counts = {
        row["seed"]: row["trainable_parameters"]
        for row in parameter_rows
        if row["model"] == "HGNN-Gated-Residual"
    }
    for row in parameter_rows:
        reference = full_counts[row["seed"]]
        relative = abs(row["trainable_parameters"] - reference) / max(reference, 1)
        row["full_model_trainable_parameters"] = reference
        row["relative_difference_vs_full"] = relative
        row["within_5pct_parameter_match"] = relative <= 0.05

    ensemble = {
        name: np.mean(np.stack(values, axis=0), axis=0)
        for name, values in predictions_by_model.items()
    }
    ensemble_hazards = {
        name: np.mean(np.stack(values, axis=0), axis=0)
        for name, values in hazards_by_model.items()
    }
    self_ensemble = np.mean(np.stack(base_risk_by_seed, axis=0), axis=0)

    comparison_rows = [
        {
            "model": "Stage 2 Self-GRU validation ensemble",
            **evaluate_risk_predictions(y_validation, self_ensemble),
        }
    ]
    for name in MODEL_NAMES:
        comparison_rows.append(
            {
                "model": name + " validation ensemble",
                **evaluate_risk_predictions(y_validation, ensemble[name]),
            }
        )

    bootstrap_rows = []
    for offset, baseline in enumerate(
        ["Stage 2 Self-GRU", "HGNN-Residual-NoGate", "HGNN-Gated-NoResidual"]
    ):
        baseline_score = (
            self_ensemble if baseline == "Stage 2 Self-GRU" else ensemble[baseline]
        )
        bootstrap_rows.append(
            {
                "split": "validation",
                "candidate": "HGNN-Gated-Residual",
                "baseline": baseline,
                "metric": "average_precision",
                "block_length_dates": BOOTSTRAP_BLOCK_LENGTH,
                **moving_block_bootstrap_improvement(
                    validation_metadata,
                    y_validation,
                    baseline_score,
                    ensemble["HGNN-Gated-Residual"],
                    safe_average_precision,
                    block_length=BOOTSTRAP_BLOCK_LENGTH,
                    n_bootstrap=BOOTSTRAP_ITERATIONS,
                    random_state=BOOTSTRAP_RANDOM_STATE + offset,
                ),
            }
        )

    prediction_df = validation_metadata.copy()
    for column in prediction_df.columns:
        if np.issubdtype(prediction_df[column].dtype, np.datetime64):
            prediction_df[column] = pd.to_datetime(
                prediction_df[column]
            ).dt.strftime("%Y-%m-%d")
    prediction_df.insert(0, "package_sample_index", validation_indices)
    prediction_df["next_3_day_onset_label"] = y_validation
    for name in MODEL_NAMES:
        prefix = name.lower().replace("-", "_")
        prediction_df[f"{prefix}_risk"] = ensemble[name]
        for day in range(HAZARD_DAYS):
            prediction_df[f"{prefix}_hazard_day{day + 1}"] = ensemble_hazards[name][:, day]

    pd.DataFrame(metrics_rows).to_csv(
        OUT_DIR / "hgnn_seed_metrics_validation.csv", index=False
    )
    pd.DataFrame(parameter_rows).to_csv(
        OUT_DIR / "hgnn_parameter_match_summary.csv", index=False
    )
    pd.DataFrame(comparison_rows).to_csv(
        OUT_DIR / "hgnn_validation_model_comparison.csv", index=False
    )
    pd.DataFrame(bootstrap_rows).to_csv(
        OUT_DIR / "hgnn_validation_moving_block_bootstrap.csv", index=False
    )
    pd.concat(histories, ignore_index=True).to_csv(
        OUT_DIR / "hgnn_training_history.csv", index=False
    )
    prediction_df.to_csv(
        PREDICTION_DIR / "validation_hgnn_ablation_ensemble_predictions.csv",
        index=False,
    )

    design_manifest = {
        "stage": "Stage 3b incidence-HGNN development",
        "provenance": "reference implementation",
        "created_without_locked_test_evaluation": True,
        "locked_test_policy": (
            "Training and selection use train/validation partitions only. "
            "The historical held-out period had been examined in an earlier stage."
        ),
        "protocol_sha256": protocol_hash,
        "models": MODEL_NAMES,
        "message_passing": {
            "type": "focal-excluded incidence node-hyperedge-node",
            "operator": (
                "P_ij = 1[i!=j]/m_i * sum_e H_ie H_je/(d_e-1); Z=PX"
            ),
            "self_feature_exclusion": "exact zero diagonal",
            "singleton_hyperedge_policy": "ignored; zero context if no eligible edge",
            "learned_after_propagation": "relation-specific Linear-LN-GELU-GRU",
        },
        "relation_structure": {
            "product_group": incidence_structure_summary(H_group),
            "storage_location": incidence_structure_summary(H_storage),
        },
        "training": {
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "maximum_epochs": MAX_EPOCHS,
            "minimum_epochs": MIN_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
            "relation_dropout": RELATION_DROPOUT,
            "gate_l1_weight": GATE_L1_WEIGHT,
            "delta_l2_weight": DELTA_L2_WEIGHT,
            "anchor_loss_weight": ANCHOR_LOSS_WEIGHT,
            "ranking_loss_weight": RANKING_LOSS_WEIGHT,
            "selection_metric": "validation average precision",
            "random_seeds": seeds,
        },
        "raw_input_files": {key: str(path) for key, path in raw_paths.items()},
        "raw_date_count": len(raw_dates),
    }
    save_json(design_manifest, OUT_DIR / "stage3b_design_manifest.json")

    audit_rows = [
        {
            "check_name": "protocol_matches_stage1_package",
            "status": "PASS",
            "details": protocol_hash,
        },
        {
            "check_name": "relational_operators_exclude_focal_features",
            "status": "PASS" if (
                focal_exclusion_invariants_hold(H_group)
                and focal_exclusion_invariants_hold(H_storage)
            ) else "FAIL",
            "details": (
                "zero diagonal; active rows sum to one; isolated rows are zero"
            ),
        },
        {
            "check_name": "all_model_seed_checkpoints_created",
            "status": "PASS",
            "details": len(seeds) * len(MODEL_NAMES),
        },
        {
            "check_name": "no_test_metric_or_prediction_created",
            "status": "PASS",
            "details": "Stage 3b selects on validation only",
        },
        {
            "check_name": "reconstruction_provenance_recorded",
            "status": "PASS",
            "details": "not claimed as byte-identical historical code",
        },
    ]
    audit_df = pd.DataFrame(audit_rows)
    if not audit_df["status"].eq("PASS").all():
        raise RuntimeError(audit_df.to_string(index=False))
    audit_df.to_csv(OUT_DIR / "stage3b_audit_report.csv", index=False)


# ============================================================
# 5. Synthetic self-test
# ============================================================

def run_self_test():
    rng = np.random.default_rng(42)
    H = np.asarray(
        [
            [1, 0],
            [1, 1],
            [0, 1],
        ],
        dtype=np.float32,
    )
    X = rng.normal(size=(5, 3, INPUT_DIM)).astype(np.float32)
    propagated = genuine_node_hyperedge_node_propagation(X, H)
    operator = focal_excluded_incidence_operator(H)
    expected = np.einsum("nm,tmf->tnf", operator, X)
    if not np.allclose(propagated, expected, atol=1e-6):
        raise AssertionError("incidence propagation does not match matrix operator")
    if not np.allclose(np.diag(operator), 0.0):
        raise AssertionError("focal-excluded incidence operator has a nonzero diagonal")
    if not np.allclose(operator.sum(axis=1), 1.0):
        raise AssertionError("eligible focal-excluded operator rows must sum to one")
    perturbed = X.copy()
    perturbed[:, 1, :] += 1000.0
    perturbed_context = genuine_node_hyperedge_node_propagation(perturbed, H)
    if not np.allclose(perturbed_context[:, 1, :], propagated[:, 1, :]):
        raise AssertionError("focal features leaked into their relational context")

    singleton_H = np.eye(2, dtype=np.float32)
    singleton_context = genuine_node_hyperedge_node_propagation(
        rng.normal(size=(3, 2, INPUT_DIM)).astype(np.float32),
        singleton_H,
    )
    if not np.allclose(singleton_context, 0.0):
        raise AssertionError("singleton hyperedges must produce zero context")

    self_model = SelfGRUHazardModel(INPUT_DIM, 64, 0.2)
    batch = torch.randn(4, LOOKBACK_DAYS, INPUT_DIM)
    counts = {}
    for name in MODEL_NAMES:
        model = build_model(name, copy.deepcopy(self_model), 64, 0.2)
        output = model(batch, batch, batch)
        if output["final_logits"].shape != (4, HAZARD_DAYS):
            raise AssertionError(f"{name}: wrong logit shape")
        if any(p.requires_grad for p in model.self_model.parameters()):
            raise AssertionError(f"{name}: Self-GRU is not frozen")
        counts[name] = count_trainable_parameters(model)
    if counts["HGNN-Gated-Residual"] != 52360:
        raise AssertionError(f"unexpected full-model count: {counts}")
    if max(
        abs(value - counts["HGNN-Gated-Residual"])
        / counts["HGNN-Gated-Residual"]
        for value in counts.values()
    ) > 0.05:
        raise AssertionError(f"structural controls are not capacity matched: {counts}")

    dates = pd.date_range("2023-01-01", periods=12, freq="D")
    metadata = pd.DataFrame({"date": np.repeat(dates, 2)})
    labels = np.tile([0, 1], 12)
    baseline = rng.random(24)
    candidate = np.clip(baseline + 0.05 * labels, 0, 1)
    summary = moving_block_bootstrap_improvement(
        metadata,
        labels,
        baseline,
        candidate,
        safe_average_precision,
        block_length=4,
        n_bootstrap=100,
        random_state=7,
    )
    required = {
        "mean_improvement",
        "ci_lower_95",
        "ci_upper_95",
        "nonpositive_resample_fraction",
        "completed_iterations",
    }
    if not required.issubset(summary):
        raise AssertionError(f"moving-block summary missing fields: {summary}")

    print("SYNTHETIC SELF-TEST: PASS")
    print("trainable_parameter_counts=", counts)
    print("moving_block_completed_iterations=", summary["completed_iterations"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run synthetic operator/model checks without study data",
    )
    arguments = parser.parse_args()
    if arguments.self_test:
        run_self_test()
    else:
        main()
