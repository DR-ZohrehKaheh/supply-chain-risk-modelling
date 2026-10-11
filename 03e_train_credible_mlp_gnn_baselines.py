# Project supervisor: Dr. Zohreh Kaheh.
# Undergraduate project collaborator: Yuyang Zhang.
# Initial student coding was developed with supervisory teaching and improvement guidance.
# -*- coding: utf-8 -*-

from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import os
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
# 0. Paths and frozen settings
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
STAGE3B_SCRIPT = BASE_DIR / "03b_train_genuine_hgnn_baselines.py"

STAGE3B_DIR = BASE_DIR / "hgnn_ablation_results"
STAGE3B_VALIDATION_PREDICTIONS = (
    STAGE3B_DIR
    / "predictions"
    / "validation_hgnn_ablation_ensemble_predictions.csv"
)

OUT_DIR = BASE_DIR / "credible_baseline_results"
CHECKPOINT_DIR = OUT_DIR / "checkpoints"
PREDICTION_DIR = OUT_DIR / "predictions"

OUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAMES = [
    "Temporal-MLP",
    "Clique-GCN-Gated-Residual",
    "Clique-GraphSAGE-Gated-Residual",
]

FULL_HGNN_NAME = "HGNN-Gated-Residual"
FULL_HGNN_VALIDATION_COLUMN = "hgnn_gated_residual_risk"

PARAMETER_MATCH_TOLERANCE = 0.05
BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_BLOCK_LENGTH = 7
BOOTSTRAP_RANDOM_STATE = 20260813


# ============================================================
# 1. Utilities
# ============================================================

def load_stage3b():
    if not STAGE3B_SCRIPT.exists():
        raise FileNotFoundError(STAGE3B_SCRIPT)

    spec = importlib.util.spec_from_file_location(
        "stage3b_hgnn",
        STAGE3B_SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_trainable_parameters(model):
    return int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
    )


def add_audit_check(rows, name, passed, details):
    rows.append(
        {
            "check_name": name,
            "status": "PASS" if passed else "FAIL",
            "details": str(details),
        }
    )


def ensemble_mean(arrays):
    return np.mean(np.stack(arrays, axis=0), axis=0)


# ============================================================
# 2. Pairwise reductions of each hypergraph relation
# ============================================================

def clique_binary_adjacency(H, include_self):
    """Return a binary pairwise clique graph induced by incidence matrix H."""
    H = np.asarray(H, dtype=np.float32)
    adjacency = (H @ H.T > 0).astype(np.float32)

    if include_self:
        np.fill_diagonal(adjacency, 1.0)
    else:
        np.fill_diagonal(adjacency, 0.0)

    return adjacency


def symmetric_gcn_operator(H):
    """Self-excluded D^-1/2 A D^-1/2 clique-GCN operator."""
    adjacency = clique_binary_adjacency(H, include_self=False)
    degree = adjacency.sum(axis=1)

    inv_sqrt_degree = np.zeros_like(degree, dtype=np.float32)
    nonzero = degree > 0
    inv_sqrt_degree[nonzero] = 1.0 / np.sqrt(degree[nonzero])

    operator = (
        inv_sqrt_degree[:, None]
        * adjacency
        * inv_sqrt_degree[None, :]
    )
    return operator.astype(np.float32)


def row_mean_neighbor_operator(H):
    """Pairwise mean-neighbour operator for GraphSAGE (focal node excluded)."""
    adjacency = clique_binary_adjacency(H, include_self=False)
    degree = adjacency.sum(axis=1)

    inv_degree = np.zeros_like(degree, dtype=np.float32)
    nonzero = degree > 0
    inv_degree[nonzero] = 1.0 / degree[nonzero]

    operator = inv_degree[:, None] * adjacency
    return operator.astype(np.float32)


def propagate_node_tensor(X_node, operator):
    """Apply a fixed node-to-node operator at every time step."""
    X_node = np.asarray(X_node, dtype=np.float32)
    operator = np.asarray(operator, dtype=np.float32)

    if X_node.ndim != 3:
        raise RuntimeError("X_node must have shape [T,N,F].")
    if operator.shape != (X_node.shape[1], X_node.shape[1]):
        raise RuntimeError(
            f"Operator shape {operator.shape} is incompatible with "
            f"node tensor {X_node.shape}."
        )

    return np.einsum(
        "ij,tjf->tif",
        operator,
        X_node,
        optimize=True,
    ).astype(np.float32)


def build_target_sequences(
    metadata,
    node_tensor,
    selected_indices,
    lookback_days,
):
    """Extract target-node temporal sequences with any feature dimension."""
    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    feature_dim = int(node_tensor.shape[2])

    X = np.zeros(
        (len(selected_indices), lookback_days, feature_dim),
        dtype=np.float32,
    )

    for output_position, sample_index in enumerate(selected_indices):
        row = metadata.iloc[int(sample_index)]
        node_index = int(row["node_index"])
        raw_time_index = int(row["raw_time_index"])

        start = raw_time_index - lookback_days + 1
        end = raw_time_index + 1

        if start < 0 or end > node_tensor.shape[0]:
            raise RuntimeError(
                "Requested baseline sequence lies outside the development "
                "temporal tensor."
            )

        X[output_position] = node_tensor[
            start:end,
            node_index,
            :,
        ]

    return X


# ============================================================
# 3. Baseline model definitions
# ============================================================

class TemporalMLPHazardModel(nn.Module):
    """Structure-unaware MLP over the same flattened focal-node history."""

    def __init__(self, lookback_days, input_dim, width, dropout, hazard_days):
        super().__init__()
        flat_dim = int(lookback_days * input_dim)

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, hazard_days),
        )

    def forward(self, x_self, x_group=None, x_storage=None):
        logits = self.network(x_self)
        zeros = torch.zeros(
            (x_self.shape[0], 1),
            dtype=x_self.dtype,
            device=x_self.device,
        )
        return {
            "final_logits": logits,
            "base_logits": logits.detach(),
            "gate_group": zeros,
            "gate_storage": zeros,
            "delta_group": torch.zeros_like(logits),
            "delta_storage": torch.zeros_like(logits),
        }


class RelationTemporalBranch(nn.Module):
    """Learned feature map plus GRU for one pairwise relation sequence."""

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

    def forward(self, x):
        z = self.feature_projection(x)
        z = self.input_norm(z)
        z = F.gelu(z)
        z = self.dropout(z)

        _, hidden = self.gru(z)
        representation = hidden[-1]
        representation = self.output_norm(representation)
        return self.dropout(representation)


class PairwiseGatedResidualModel(nn.Module):
    """Frozen Self-GRU plus two gated pairwise-relation residual branches."""

    def __init__(
        self,
        frozen_self_model,
        relation_input_dim,
        hidden_dim,
        dropout,
        input_dim,
        hazard_days,
        initial_gate_bias,
    ):
        super().__init__()

        self.self_model = frozen_self_model
        for parameter in self.self_model.parameters():
            parameter.requires_grad = False

        self.group_branch = RelationTemporalBranch(
            relation_input_dim,
            hidden_dim,
            dropout,
        )
        self.storage_branch = RelationTemporalBranch(
            relation_input_dim,
            hidden_dim,
            dropout,
        )

        self.gate_layer = nn.Linear(hidden_dim * 3, 2)
        self.group_delta_head = nn.Linear(hidden_dim, hazard_days)
        self.storage_delta_head = nn.Linear(hidden_dim, hazard_days)

        nn.init.zeros_(self.group_delta_head.weight)
        nn.init.zeros_(self.group_delta_head.bias)
        nn.init.zeros_(self.storage_delta_head.weight)
        nn.init.zeros_(self.storage_delta_head.bias)
        nn.init.zeros_(self.gate_layer.weight)
        nn.init.constant_(self.gate_layer.bias, initial_gate_bias)

        self.expected_self_input_dim = int(input_dim)

    def forward(self, x_self, x_group, x_storage):
        with torch.no_grad():
            self_representation = self.self_model.encode(x_self)
            base_logits = self.self_model(x_self)

        group_representation = self.group_branch(x_group)
        storage_representation = self.storage_branch(x_storage)

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

        delta_group = self.group_delta_head(group_representation)
        delta_storage = self.storage_delta_head(storage_representation)

        final_logits = (
            base_logits
            + gate_group * delta_group
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


def choose_parameter_matched_mlp_width(
    target_parameter_count,
    lookback_days,
    input_dim,
    dropout,
    hazard_days,
):
    candidates = []

    for width in range(8, 1025):
        model = TemporalMLPHazardModel(
            lookback_days,
            input_dim,
            width,
            dropout,
            hazard_days,
        )
        count = count_trainable_parameters(model)
        relative_difference = abs(count - target_parameter_count) / max(
            target_parameter_count,
            1,
        )
        candidates.append((relative_difference, width, count))

    relative_difference, width, count = min(candidates)

    if relative_difference > PARAMETER_MATCH_TOLERANCE:
        raise RuntimeError(
            "Unable to parameter-match Temporal-MLP within 5 percent: "
            f"target={target_parameter_count}, best={count}, "
            f"width={width}, relative_difference={relative_difference:.6f}"
        )

    return int(width), int(count), float(relative_difference)


def build_model(
    model_name,
    s3b,
    seed,
    device,
    mlp_width=None,
):
    frozen_self_model, self_checkpoint = s3b.load_frozen_self_model(
        seed,
        device,
    )
    hidden_dim = int(self_checkpoint["config"]["hidden_dim"])

    if model_name == "Temporal-MLP":
        if mlp_width is None:
            raise ValueError("Temporal-MLP requires mlp_width.")
        model = TemporalMLPHazardModel(
            s3b.LOOKBACK_DAYS,
            s3b.INPUT_DIM,
            mlp_width,
            s3b.RELATION_DROPOUT,
            s3b.HAZARD_DAYS,
        )
        relation_input_dim = 0

    elif model_name == "Clique-GCN-Gated-Residual":
        model = PairwiseGatedResidualModel(
            frozen_self_model=frozen_self_model,
            relation_input_dim=s3b.INPUT_DIM,
            hidden_dim=hidden_dim,
            dropout=s3b.RELATION_DROPOUT,
            input_dim=s3b.INPUT_DIM,
            hazard_days=s3b.HAZARD_DAYS,
            initial_gate_bias=s3b.INITIAL_GATE_BIAS,
        )
        relation_input_dim = s3b.INPUT_DIM

    elif model_name == "Clique-GraphSAGE-Gated-Residual":
        model = PairwiseGatedResidualModel(
            frozen_self_model=frozen_self_model,
            relation_input_dim=s3b.INPUT_DIM,
            hidden_dim=hidden_dim,
            dropout=s3b.RELATION_DROPOUT,
            input_dim=s3b.INPUT_DIM,
            hazard_days=s3b.HAZARD_DAYS,
            initial_gate_bias=s3b.INITIAL_GATE_BIAS,
        )
        relation_input_dim = s3b.INPUT_DIM

    else:
        raise ValueError(f"Unknown baseline model: {model_name}")

    return (
        model.to(device),
        hidden_dim,
        relation_input_dim,
        self_checkpoint["config"],
    )


# ============================================================
# 4. Training and prediction
# ============================================================

def build_loader(arrays, seed, s3b):
    dataset = TensorDataset(
        torch.from_numpy(arrays["X_self_train"]).float(),
        torch.from_numpy(arrays["X_group_train"]).float(),
        torch.from_numpy(arrays["X_storage_train"]).float(),
        torch.from_numpy(arrays["y_hazard_train"]).float(),
        torch.from_numpy(arrays["hazard_mask_train"]).float(),
        torch.from_numpy(
            arrays["y_3_day_train"].astype(np.float32)
        ).float(),
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=s3b.BATCH_SIZE,
        shuffle=True,
        num_workers=s3b.NUM_WORKERS,
        generator=generator,
        drop_last=False,
    )


def training_loss(
    model_name,
    outputs,
    hazard_targets,
    hazard_mask,
    y_three_day,
    positive_weights,
    s3b,
):
    if model_name == "Temporal-MLP":
        final_logits = outputs["final_logits"]
        hazard_loss = s3b.masked_hazard_loss(
            final_logits,
            hazard_targets,
            hazard_mask,
            positive_weights,
        )
        risk = s3b.hazards_to_three_day_risk_torch(
            torch.sigmoid(final_logits)
        )
        ranking_loss = s3b.pairwise_ranking_loss(risk, y_three_day)
        return hazard_loss + s3b.RANKING_LOSS_WEIGHT * ranking_loss

    losses = s3b.model_training_loss(
        "HGNN-Gated-Residual",
        outputs,
        hazard_targets,
        hazard_mask,
        y_three_day,
        positive_weights,
    )
    return losses["total_loss"]


def train_one_seed(
    model_name,
    seed,
    arrays,
    y_validation,
    s3b,
    device,
    mlp_width,
):
    set_global_seed(seed)

    model, hidden_dim, relation_input_dim, self_config = build_model(
        model_name,
        s3b,
        seed,
        device,
        mlp_width=mlp_width,
    )

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=s3b.LEARNING_RATE,
        weight_decay=s3b.WEIGHT_DECAY,
    )

    positive_weights_np = s3b.calculate_hazard_positive_weights(
        arrays["y_hazard_train"],
        arrays["hazard_mask_train"],
    )
    positive_weights = torch.tensor(
        positive_weights_np,
        dtype=torch.float32,
        device=device,
    )

    loader = build_loader(arrays, seed, s3b)

    best_ap = -np.inf
    best_epoch = 0
    best_state = None
    best_prediction = None
    best_metrics = None
    epochs_without_improvement = 0
    history_rows = []

    for epoch in range(1, s3b.MAX_EPOCHS + 1):
        model.train()
        if hasattr(model, "self_model"):
            model.self_model.eval()

        total_loss_sum = 0.0
        total_mask_sum = 0.0

        for (
            batch_self,
            batch_group,
            batch_storage,
            batch_y,
            batch_mask,
            batch_y_three_day,
        ) in loader:
            batch_self = batch_self.to(device)
            batch_group = batch_group.to(device)
            batch_storage = batch_storage.to(device)
            batch_y = batch_y.to(device)
            batch_mask = batch_mask.to(device)
            batch_y_three_day = batch_y_three_day.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch_self, batch_group, batch_storage)
            loss = training_loss(
                model_name,
                outputs,
                batch_y,
                batch_mask,
                batch_y_three_day,
                positive_weights,
                s3b,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                max_norm=s3b.GRADIENT_CLIP_NORM,
            )
            optimizer.step()

            mask_sum = float(batch_mask.sum().item())
            total_loss_sum += float(loss.item()) * mask_sum
            total_mask_sum += mask_sum

        validation_prediction = s3b.predict_model(
            model,
            arrays["X_self_validation"],
            arrays["X_group_validation"],
            arrays["X_storage_validation"],
            device,
        )
        validation_metrics = s3b.evaluate_risk_predictions(
            y_validation,
            validation_prediction["risk"],
        )
        validation_ap = float(validation_metrics["average_precision"])

        history_rows.append(
            {
                "model": model_name,
                "seed": seed,
                "epoch": epoch,
                "training_total_loss": (
                    total_loss_sum / max(total_mask_sum, 1.0)
                ),
                "validation_average_precision": validation_ap,
                "validation_roc_auc": validation_metrics["roc_auc"],
                "validation_recall_at_top10": validation_metrics[
                    "recall_at_top10"
                ],
            }
        )

        if np.isfinite(validation_ap) and validation_ap > best_ap + 1e-8:
            best_ap = validation_ap
            best_epoch = int(epoch)
            best_state = copy.deepcopy(model.state_dict())
            best_prediction = {
                key: value.copy()
                for key, value in validation_prediction.items()
            }
            best_metrics = dict(validation_metrics)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (
            epoch >= s3b.MIN_EPOCHS
            and epochs_without_improvement >= s3b.EARLY_STOPPING_PATIENCE
        ):
            break

    if best_state is None:
        raise RuntimeError(
            f"No valid validation checkpoint for {model_name}, seed={seed}."
        )

    model.load_state_dict(best_state)

    return {
        "model": model,
        "hidden_dim": hidden_dim,
        "relation_input_dim": relation_input_dim,
        "mlp_width": mlp_width if model_name == "Temporal-MLP" else None,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "best_prediction": best_prediction,
        "history": pd.DataFrame(history_rows),
        "positive_weights": positive_weights_np,
        "trainable_parameter_count": count_trainable_parameters(model),
        "self_checkpoint_config": self_config,
    }


# ============================================================
# 5. Main: development-only training and validation
# ============================================================

def main():
    print("=" * 72)
    print("STAGE 3e: CREDIBLE MLP + PAIRWISE-GNN BASELINES")
    print("TRAIN + VALIDATION ONLY; HELD-OUT TEST NOT EVALUATED")
    print("=" * 72)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nDevice:", device)

    if device.type == "cpu":
        torch.set_num_threads(int(os.environ.get("PORTFOLIO_TORCH_THREADS", "1")))

    s3b = load_stage3b()
    protocol, protocol_hash = s3b.load_locked_protocol()
    package = s3b.load_package()
    normalizer = s3b.load_stage2_normalization()

    if protocol_hash != package["package_protocol_hash"]:
        raise RuntimeError(
            "Current protocol hash does not match the Stage-1 package."
        )

    if not STAGE3B_VALIDATION_PREDICTIONS.exists():
        raise FileNotFoundError(
            "Run Stage 3b successfully before Stage 3e. Missing: "
            f"{STAGE3B_VALIDATION_PREDICTIONS}"
        )

    seeds = [int(seed) for seed in protocol["random_seeds"]]
    train_indices = package["train_indices"].astype(np.int64)
    validation_indices = package["validation_indices"].astype(np.int64)
    test_indices = package["test_indices"].astype(np.int64)

    print("\nTrain samples:", len(train_indices))
    print("Validation samples:", len(validation_indices))
    print("Held-out test samples NOT USED:", len(test_indices))
    print("Seeds:", seeds)

    node_table = (
        package["metadata"][["node_index", "node"]]
        .drop_duplicates()
        .sort_values("node_index")
    )
    nodes = node_table["node"].astype(str).tolist()
    if len(nodes) != 41:
        raise RuntimeError(f"Expected 41 nodes, got {len(nodes)}.")

    raw_tensor, raw_dates, raw_paths = s3b.load_raw_temporal_tensor(nodes)
    development_indices = np.concatenate(
        [train_indices, validation_indices]
    )
    max_development_time = int(
        package["metadata"]
        .iloc[development_indices]["raw_time_index"]
        .max()
    )
    min_test_time = int(
        package["metadata"].iloc[test_indices]["raw_time_index"].min()
    )

    raw_development = raw_tensor[: max_development_time + 1].copy()
    raw_development_normalised = s3b.transform_all_node_tensor(
        raw_development,
        normalizer,
    )
    # Transform only the development samples required by this stage. The
    # held-out feature windows are not materialised into Stage-3e arrays.
    X_self_train = s3b.transform_sample_tensor(
        package["X_self"][train_indices],
        normalizer,
    )
    X_self_validation = s3b.transform_sample_tensor(
        package["X_self"][validation_indices],
        normalizer,
    )

    H_group = s3b.validate_incidence_matrix(
        package["H_product_group"],
        "Product Group",
    )
    H_storage = s3b.validate_incidence_matrix(
        package["H_storage_location"],
        "Storage Location",
    )

    # Pairwise GCN relation tensors.
    gcn_group_operator = symmetric_gcn_operator(H_group)
    gcn_storage_operator = symmetric_gcn_operator(H_storage)
    gcn_group_tensor = propagate_node_tensor(
        raw_development_normalised,
        gcn_group_operator,
    )
    gcn_storage_tensor = propagate_node_tensor(
        raw_development_normalised,
        gcn_storage_operator,
    )

    # Mean GraphSAGE relation tensors contain neighbour context only.  The
    # frozen Self-GRU is the sole path for the focal product's own history.
    sage_group_neighbour_tensor = propagate_node_tensor(
        raw_development_normalised,
        row_mean_neighbor_operator(H_group),
    )
    sage_storage_neighbour_tensor = propagate_node_tensor(
        raw_development_normalised,
        row_mean_neighbor_operator(H_storage),
    )
    sage_group_tensor = sage_group_neighbour_tensor
    sage_storage_tensor = sage_storage_neighbour_tensor

    relation_sequences = {
        "Clique-GCN-Gated-Residual": {
            "group_train": build_target_sequences(
                package["metadata"],
                gcn_group_tensor,
                train_indices,
                s3b.LOOKBACK_DAYS,
            ),
            "storage_train": build_target_sequences(
                package["metadata"],
                gcn_storage_tensor,
                train_indices,
                s3b.LOOKBACK_DAYS,
            ),
            "group_validation": build_target_sequences(
                package["metadata"],
                gcn_group_tensor,
                validation_indices,
                s3b.LOOKBACK_DAYS,
            ),
            "storage_validation": build_target_sequences(
                package["metadata"],
                gcn_storage_tensor,
                validation_indices,
                s3b.LOOKBACK_DAYS,
            ),
        },
        "Clique-GraphSAGE-Gated-Residual": {
            "group_train": build_target_sequences(
                package["metadata"],
                sage_group_tensor,
                train_indices,
                s3b.LOOKBACK_DAYS,
            ),
            "storage_train": build_target_sequences(
                package["metadata"],
                sage_storage_tensor,
                train_indices,
                s3b.LOOKBACK_DAYS,
            ),
            "group_validation": build_target_sequences(
                package["metadata"],
                sage_group_tensor,
                validation_indices,
                s3b.LOOKBACK_DAYS,
            ),
            "storage_validation": build_target_sequences(
                package["metadata"],
                sage_storage_tensor,
                validation_indices,
                s3b.LOOKBACK_DAYS,
            ),
        },
    }

    y_validation = package["y_3_day"][validation_indices].astype(int)
    validation_metadata = (
        package["metadata"]
        .iloc[validation_indices]
        .copy()
        .reset_index(drop=True)
    )

    stage3b_validation = pd.read_csv(STAGE3B_VALIDATION_PREDICTIONS)
    if FULL_HGNN_VALIDATION_COLUMN not in stage3b_validation.columns:
        raise KeyError(
            f"Missing Stage-3b column: {FULL_HGNN_VALIDATION_COLUMN}"
        )
    if len(stage3b_validation) != len(validation_indices):
        raise RuntimeError("Stage-3b validation prediction length mismatch.")

    if (
        "sample_id" in stage3b_validation.columns
        and "sample_id" in validation_metadata.columns
    ):
        saved_ids = stage3b_validation["sample_id"].astype(str).values
        expected_ids = validation_metadata["sample_id"].astype(str).values
        if not np.array_equal(saved_ids, expected_ids):
            raise RuntimeError(
                "Stage-3b validation predictions are not aligned with the "
                "current validation sample order."
            )

    if "next_3_day_onset_label" in stage3b_validation.columns:
        saved_labels = stage3b_validation[
            "next_3_day_onset_label"
        ].astype(int).values
        if not np.array_equal(saved_labels, y_validation):
            raise RuntimeError(
                "Stage-3b validation labels do not match the current package."
            )

    full_hgnn_validation_risk = stage3b_validation[
        FULL_HGNN_VALIDATION_COLUMN
    ].astype(float).values

    # Determine the matched MLP width separately for each seed because the
    # locked Self-GRU hidden dimension is read from its frozen checkpoint.
    mlp_width_by_seed = {}
    full_hgnn_parameter_count_by_seed = {}
    mlp_parameter_count_by_seed = {}
    mlp_parameter_difference_by_seed = {}

    for seed in seeds:
        frozen_self, self_checkpoint = s3b.load_frozen_self_model(seed, device)
        hidden_dim = int(self_checkpoint["config"]["hidden_dim"])
        reference_hgnn = s3b.HGNNGatedResidualModel(
            frozen_self,
            hidden_dim,
            s3b.RELATION_DROPOUT,
        ).to(device)
        reference_count = s3b.count_trainable_parameters(reference_hgnn)
        width, mlp_count, relative_difference = (
            choose_parameter_matched_mlp_width(
                reference_count,
                s3b.LOOKBACK_DAYS,
                s3b.INPUT_DIM,
                s3b.RELATION_DROPOUT,
                s3b.HAZARD_DAYS,
            )
        )
        mlp_width_by_seed[seed] = width
        full_hgnn_parameter_count_by_seed[seed] = reference_count
        mlp_parameter_count_by_seed[seed] = mlp_count
        mlp_parameter_difference_by_seed[seed] = relative_difference
        del reference_hgnn

    if len(set(full_hgnn_parameter_count_by_seed.values())) != 1:
        raise RuntimeError(
            "The frozen full-HGNN trainable parameter count varies across "
            "seeds; a single parameter-matched MLP architecture cannot be "
            "defined without revising the frozen design."
        )

    if len(set(mlp_width_by_seed.values())) != 1:
        raise RuntimeError(
            "The mechanically selected MLP width varies across seeds. "
            "Stage 3e requires one fixed MLP architecture for all seeds."
        )

    design_manifest = {
        "stage": "Stage 3e credible MLP and pairwise-GNN baselines",
        "created_without_held_out_test_evaluation": True,
        "protocol_sha256": protocol_hash,
        "stage3b_script_sha256": sha256_file(STAGE3B_SCRIPT),
        "models": MODEL_NAMES,
        "temporal_mlp": {
            "input": "flattened 14 x 8 focal-node history",
            "parameter_match_target": FULL_HGNN_NAME,
            "parameter_match_tolerance": PARAMETER_MATCH_TOLERANCE,
            "width_by_seed": mlp_width_by_seed,
        },
        "pairwise_gcn": {
            "reduction": "binary clique expansion per relation",
            "operator": "D^-1/2 A D^-1/2 X with diag(A)=0",
            "self_feature_exclusion": "exact zero diagonal",
            "fusion": "frozen Self-GRU gated residual",
            "equivalence_caveat": (
                "Clique projection discards hyperedge identity and weights; "
                "operator differences are recorded rather than assumed."
            ),
        },
        "pairwise_graphsage": {
            "reduction": "binary clique expansion per relation",
            "aggregation": "mean pairwise neighbours; focal node excluded",
            "self_feature_exclusion": "exact zero diagonal",
            "fusion": "frozen Self-GRU gated residual",
        },
        "training": {
            "selection_split": "validation",
            "selection_metric": "average_precision",
            "learning_rate": s3b.LEARNING_RATE,
            "weight_decay": s3b.WEIGHT_DECAY,
            "max_epochs": s3b.MAX_EPOCHS,
            "early_stopping_patience": s3b.EARLY_STOPPING_PATIENCE,
            "random_seeds": seeds,
        },
        "bootstrap": {
            "split": "validation only",
            "type": "paired moving-block bootstrap over dates",
            "block_length_dates": BOOTSTRAP_BLOCK_LENGTH,
            "iterations": BOOTSTRAP_ITERATIONS,
        },
        "held_out_test_policy": (
            "No test prediction or test performance is calculated in Stage 3e."
        ),
    }
    save_json(design_manifest, OUT_DIR / "stage3e_design_manifest.json")

    all_history = []
    seed_metric_rows = []
    parameter_rows = []
    predictions_by_model = {name: [] for name in MODEL_NAMES}
    hazards_by_model = {name: [] for name in MODEL_NAMES}

    for seed in seeds:
        print("\n" + "=" * 72)
        print("SEED:", seed)
        print("=" * 72)

        for model_name in MODEL_NAMES:
            print("\nTraining:", model_name)

            if model_name == "Temporal-MLP":
                # Relation tensors are ignored by the MLP. Reusing self tensors
                # avoids a separate data-loader interface.
                group_train = X_self_train
                storage_train = X_self_train
                group_validation = X_self_validation
                storage_validation = X_self_validation
            else:
                relation = relation_sequences[model_name]
                group_train = relation["group_train"]
                storage_train = relation["storage_train"]
                group_validation = relation["group_validation"]
                storage_validation = relation["storage_validation"]

            arrays = {
                "X_self_train": X_self_train,
                "X_group_train": group_train,
                "X_storage_train": storage_train,
                "y_hazard_train": package["y_hazard"][train_indices],
                "hazard_mask_train": package["hazard_mask"][train_indices],
                "y_3_day_train": package["y_3_day"][train_indices],
                "X_self_validation": X_self_validation,
                "X_group_validation": group_validation,
                "X_storage_validation": storage_validation,
            }

            result = train_one_seed(
                model_name,
                seed,
                arrays,
                y_validation,
                s3b,
                device,
                mlp_width_by_seed[seed],
            )

            metrics = result["best_metrics"]
            prediction = result["best_prediction"]
            parameter_count = int(result["trainable_parameter_count"])
            reference_count = int(full_hgnn_parameter_count_by_seed[seed])

            seed_metric_rows.append(
                {
                    "seed": seed,
                    "model": model_name,
                    "best_epoch": result["best_epoch"],
                    "validation_average_precision": metrics[
                        "average_precision"
                    ],
                    "validation_roc_auc": metrics["roc_auc"],
                    "validation_precision_at_top10": metrics[
                        "precision_at_top10"
                    ],
                    "validation_recall_at_top10": metrics[
                        "recall_at_top10"
                    ],
                    "validation_lift_at_top10": metrics["lift_at_top10"],
                    "trainable_parameters": parameter_count,
                }
            )

            parameter_rows.append(
                {
                    "seed": seed,
                    "model": model_name,
                    "hidden_dim": result["hidden_dim"],
                    "relation_input_dim": result["relation_input_dim"],
                    "mlp_width": result["mlp_width"],
                    "trainable_parameters": parameter_count,
                    "full_hgnn_trainable_parameters": reference_count,
                    "relative_difference_vs_full_hgnn": abs(
                        parameter_count - reference_count
                    ) / max(reference_count, 1),
                }
            )

            all_history.append(result["history"])
            predictions_by_model[model_name].append(prediction["risk"])
            hazards_by_model[model_name].append(prediction["hazards"])

            model_directory = CHECKPOINT_DIR / model_name.replace("-", "_")
            model_directory.mkdir(parents=True, exist_ok=True)
            checkpoint_path = model_directory / f"seed_{seed}.pt"

            checkpoint = {
                "stage": "Stage 3e",
                "model": model_name,
                "seed": int(seed),
                "hidden_dim": int(result["hidden_dim"]),
                "relation_input_dim": int(result["relation_input_dim"]),
                "mlp_width": result["mlp_width"],
                "best_epoch": int(result["best_epoch"]),
                "validation_metrics": metrics,
                "trainable_parameter_count": parameter_count,
                "full_hgnn_trainable_parameter_count": reference_count,
                "positive_weights": result["positive_weights"].tolist(),
                "self_checkpoint_config": result["self_checkpoint_config"],
                "state_dict": result["model"].state_dict(),
            }
            torch.save(checkpoint, checkpoint_path)

            print(
                f"Validation AP={metrics['average_precision']:.6f}; "
                f"ROC-AUC={metrics['roc_auc']:.6f}; "
                f"parameters={parameter_count}; "
                f"best epoch={result['best_epoch']}"
            )

    seed_metrics = pd.DataFrame(seed_metric_rows)
    parameter_report = pd.DataFrame(parameter_rows)
    training_history = pd.concat(all_history, ignore_index=True)

    ensemble_scores = {
        model_name: ensemble_mean(predictions_by_model[model_name])
        for model_name in MODEL_NAMES
    }
    ensemble_hazards = {
        model_name: ensemble_mean(hazards_by_model[model_name])
        for model_name in MODEL_NAMES
    }

    comparison_rows = [
        {
            "model": FULL_HGNN_NAME + " validation ensemble",
            **s3b.evaluate_risk_predictions(
                y_validation,
                full_hgnn_validation_risk,
            ),
        }
    ]
    for model_name in MODEL_NAMES:
        comparison_rows.append(
            {
                "model": model_name + " validation ensemble",
                **s3b.evaluate_risk_predictions(
                    y_validation,
                    ensemble_scores[model_name],
                ),
            }
        )
    comparison_df = pd.DataFrame(comparison_rows)

    prediction_df = validation_metadata.copy()
    for column in prediction_df.columns:
        if np.issubdtype(prediction_df[column].dtype, np.datetime64):
            prediction_df[column] = pd.to_datetime(
                prediction_df[column]
            ).astype(str)
    prediction_df.insert(0, "package_sample_index", validation_indices)
    prediction_df["y_3_day"] = y_validation
    prediction_df["hgnn_gated_residual_risk"] = full_hgnn_validation_risk

    for model_name in MODEL_NAMES:
        prefix = model_name.lower().replace("-", "_")
        prediction_df[f"{prefix}_risk"] = ensemble_scores[model_name]
        for day in range(s3b.HAZARD_DAYS):
            prediction_df[f"{prefix}_hazard_day{day + 1}"] = (
                ensemble_hazards[model_name][:, day]
            )

    bootstrap_rows = []
    for offset, model_name in enumerate(MODEL_NAMES):
        result = s3b.moving_block_bootstrap_improvement(
            validation_metadata,
            y_validation,
            ensemble_scores[model_name],
            full_hgnn_validation_risk,
            s3b.safe_average_precision,
            block_length=BOOTSTRAP_BLOCK_LENGTH,
            n_bootstrap=BOOTSTRAP_ITERATIONS,
            random_state=BOOTSTRAP_RANDOM_STATE + offset,
        )
        bootstrap_rows.append(
            {
                "split": "validation",
                "candidate": FULL_HGNN_NAME,
                "baseline": model_name,
                "metric": "average_precision",
                "block_length_dates": BOOTSTRAP_BLOCK_LENGTH,
                **result,
            }
        )
    bootstrap_df = pd.DataFrame(bootstrap_rows)

    seed_metrics.to_csv(
        OUT_DIR / "credible_baseline_seed_metrics_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    parameter_report.to_csv(
        OUT_DIR / "credible_baseline_parameter_report.csv",
        index=False,
        encoding="utf-8-sig",
    )
    training_history.to_csv(
        OUT_DIR / "credible_baseline_training_history.csv",
        index=False,
        encoding="utf-8-sig",
    )
    comparison_df.to_csv(
        OUT_DIR / "credible_baseline_validation_model_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bootstrap_df.to_csv(
        OUT_DIR / "credible_baseline_validation_moving_block_bootstrap.csv",
        index=False,
        encoding="utf-8-sig",
    )
    prediction_df.to_csv(
        PREDICTION_DIR
        / "validation_credible_baseline_ensemble_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    audit_rows = []
    add_audit_check(
        audit_rows,
        "protocol_matches_stage1_package",
        protocol_hash == package["package_protocol_hash"],
        {
            "protocol": protocol_hash,
            "package": package["package_protocol_hash"],
        },
    )
    add_audit_check(
        audit_rows,
        "development_horizon_excludes_test_origins",
        max_development_time < min_test_time,
        {
            "max_development_raw_time_index": max_development_time,
            "min_test_raw_time_index": min_test_time,
        },
    )
    add_audit_check(
        audit_rows,
        "all_models_trained_for_all_locked_seeds",
        len(seed_metrics) == len(seeds) * len(MODEL_NAMES),
        {
            "rows": len(seed_metrics),
            "expected": len(seeds) * len(MODEL_NAMES),
        },
    )
    add_audit_check(
        audit_rows,
        "temporal_mlp_parameter_matched_within_5pct",
        all(
            difference <= PARAMETER_MATCH_TOLERANCE
            for difference in mlp_parameter_difference_by_seed.values()
        ),
        mlp_parameter_difference_by_seed,
    )
    group_operator_difference = float(
        np.max(
            np.abs(
                gcn_group_operator
                - s3b.focal_excluded_incidence_operator(H_group)
            )
        )
    )
    storage_operator_difference = float(
        np.max(
            np.abs(
                gcn_storage_operator
                - s3b.focal_excluded_incidence_operator(H_storage)
            )
        )
    )
    add_audit_check(
        audit_rows,
        "pairwise_and_hgnn_operator_differences_recorded",
        np.isfinite(group_operator_difference)
        and np.isfinite(storage_operator_difference),
        {
            "product_group_max_abs_difference": group_operator_difference,
            "storage_location_max_abs_difference": (
                storage_operator_difference
            ),
            "interpretation": (
                "Both operators exclude the focal node; a nonzero value "
                "quantifies the effect of clique reduction and normalisation."
            ),
        },
    )

    test_prediction_files = list(PREDICTION_DIR.glob("*test*"))
    add_audit_check(
        audit_rows,
        "no_test_prediction_file_created",
        len(test_prediction_files) == 0,
        [path.name for path in test_prediction_files],
    )
    add_audit_check(
        audit_rows,
        "held_out_test_performance_not_calculated",
        True,
        (
            "Stage 3e selects checkpoints with validation AUPRC and creates "
            "no held-out test predictions or test metrics."
        ),
    )

    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(
        OUT_DIR / "stage3e_audit_report.csv",
        index=False,
        encoding="utf-8-sig",
    )

    failed = audit_df[audit_df["status"].eq("FAIL")]

    print("\n" + "=" * 72)
    print("VALIDATION MODEL COMPARISON")
    print("=" * 72)
    print(comparison_df.to_string(index=False))

    print("\n" + "=" * 72)
    print("VALIDATION PAIRED MOVING-BLOCK BOOTSTRAP")
    print("=" * 72)
    print(bootstrap_df.to_string(index=False))

    print("\n" + "=" * 72)
    print("STAGE 3e AUDIT")
    print("=" * 72)
    print(audit_df.to_string(index=False))

    if len(failed) > 0:
        raise RuntimeError(
            "Stage 3e audit failed. Inspect stage3e_audit_report.csv."
        )

    print("\nResults saved to:", OUT_DIR)
    print(
        "\nIMPORTANT: Do not evaluate the held-out test split yet. "
        "Inspect validation outputs first, then freeze this script and all "
        "selected checkpoints in a separate stage."
    )


if __name__ == "__main__":
    main()
