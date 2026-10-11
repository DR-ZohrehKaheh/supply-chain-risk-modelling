# Project supervisor: Dr. Zohreh Kaheh.
# Undergraduate project collaborator: Yuyang Zhang.
# Initial student coding was developed with supervisory teaching and improvement guidance.
# -*- coding: utf-8 -*-

from pathlib import Path
import copy
import hashlib
import json
import math
import os
import random
import warnings

import numpy as np
import pandas as pd

import torch
import os
torch.set_num_threads(int(os.environ.get("PORTFOLIO_TORCH_THREADS", "1")))
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")


# ============================================================
# 0. Paths and locked settings
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

PROTOCOL_FILE = (
    BASE_DIR
    / "00_raht_protocol_locked.json"
)

PACKAGE_DIR = (
    BASE_DIR
    / "raht_training_package"
)

SELF_DIR = (
    BASE_DIR
    / "self_hazard_baseline_results"
)

SELF_PREDICTION_DIR = (
    SELF_DIR
    / "predictions"
)

SELF_CHECKPOINT_DIR = (
    SELF_DIR
    / "checkpoints"
)

OUT_DIR = (
    BASE_DIR
    / "raht_residual_hazard_results"
)

PREDICTION_DIR = (
    OUT_DIR
    / "predictions"
)

CHECKPOINT_DIR = (
    OUT_DIR
    / "checkpoints"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PREDICTION_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# Locked model settings
# ------------------------------------------------------------

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

# Fixed RAHT residual settings.
# These are not tuned on test data.
RAHT_LEARNING_RATE = 0.0003
RAHT_WEIGHT_DECAY = 0.001
GRAPH_DROPOUT = 0.20
GATE_L1_WEIGHT = 0.001
DELTA_L2_WEIGHT = 0.0001
ANCHOR_LOSS_WEIGHT = 0.001
RANKING_LOSS_WEIGHT = 0.05

# Negative gate bias keeps graph correction conservative at start.
INITIAL_GATE_BIAS = -2.0


TEMPORAL_KEYS = [
    "u_delivery",
    "u_factory",
    "u_production",
    "u_sales",
    "w_delivery",
    "w_factory",
    "w_production",
    "w_sales",
]


# ============================================================
# 1. Utilities
# ============================================================

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        torch.use_deterministic_algorithms(
            True,
            warn_only=True
        )
    except Exception:
        pass

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as file:
        while True:
            block = file.read(1024 * 1024)
            if not block:
                break
            h.update(block)

    return h.hexdigest()


def sha256_integer_array(values):
    values = np.asarray(
        values,
        dtype=np.int64
    )
    return hashlib.sha256(
        values.tobytes()
    ).hexdigest()


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            obj,
            file,
            indent=2,
            ensure_ascii=False
        )


def safe_torch_load(path, device):
    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=False
        )
    except TypeError:
        return torch.load(
            path,
            map_location=device
        )


def add_audit_check(rows, check_name, passed, details):
    rows.append({
        "check_name": check_name,
        "status": "PASS" if passed else "FAIL",
        "details": str(details),
    })


# ============================================================
# 2. Metrics
# ============================================================

def safe_average_precision(y_true, score):
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score).astype(float)

    if len(np.unique(y_true)) < 2:
        return np.nan

    return float(
        average_precision_score(
            y_true,
            score
        )
    )


def safe_roc_auc(y_true, score):
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score).astype(float)

    if len(np.unique(y_true)) < 2:
        return np.nan

    return float(
        roc_auc_score(
            y_true,
            score
        )
    )


def topk_metrics(
    y_true,
    score,
    top_rate=TOP_ALERT_RATE
):
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score).astype(float)

    sample_count = len(y_true)
    k = max(
        1,
        int(
            math.floor(
                sample_count
                * top_rate
            )
        )
    )

    ranked = np.argsort(score)[::-1]
    top_indices = ranked[:k]

    selected_positive = int(
        y_true[top_indices].sum()
    )
    total_positive = int(y_true.sum())

    precision = float(
        selected_positive
        / k
    )

    if total_positive > 0:
        recall = float(
            selected_positive
            / total_positive
        )
    else:
        recall = np.nan

    prevalence = float(y_true.mean())
    lift = float(
        precision
        / max(prevalence, 1e-12)
    )

    return {
        "precision_at_top10": precision,
        "recall_at_top10": recall,
        "lift_at_top10": lift,
        "top_alert_count": int(k),
    }


def evaluate_risk_predictions(y_true, risk_score):
    ap = safe_average_precision(
        y_true,
        risk_score
    )

    auc = safe_roc_auc(
        y_true,
        risk_score
    )

    top = topk_metrics(
        y_true,
        risk_score
    )

    return {
        "average_precision": ap,
        "roc_auc": auc,
        "precision_at_top10": top["precision_at_top10"],
        "recall_at_top10": top["recall_at_top10"],
        "lift_at_top10": top["lift_at_top10"],
        "top_alert_count": top["top_alert_count"],
    }


def hazards_to_three_day_risk_np(hazards):
    survival = np.prod(
        1.0 - hazards,
        axis=1
    )
    risk = 1.0 - survival
    return np.clip(risk, 0.0, 1.0)


def hazards_to_three_day_risk_torch(hazards):
    survival = torch.prod(
        1.0 - hazards,
        dim=1
    )
    return torch.clamp(
        1.0 - survival,
        min=0.0,
        max=1.0
    )


# ============================================================
# 3. Load locked files
# ============================================================

def load_locked_protocol():
    if not PROTOCOL_FILE.exists():
        raise FileNotFoundError(PROTOCOL_FILE)

    with open(PROTOCOL_FILE, "r", encoding="utf-8") as file:
        protocol = json.load(file)

    protocol_hash = sha256_file(PROTOCOL_FILE)

    if protocol["project_name"] != "RAHT-RH-CRC":
        raise RuntimeError("Unexpected project name.")

    if protocol["primary_metric"] != "average_precision":
        raise RuntimeError("Primary metric must be average_precision.")

    expected_relations = {
        "product_group",
        "storage_location",
    }

    if set(protocol["primary_relations"]) != expected_relations:
        raise RuntimeError(
            "Primary relations must be product_group and storage_location."
        )

    return protocol, protocol_hash


def load_package():
    required = [
        "X_self.npy",
        "y_hazard.npy",
        "hazard_at_risk_mask.npy",
        "y_3_day.npy",
        "sample_metadata.csv",
        "split_indices.npz",
        "H_product_group.npy",
        "H_storage_location.npy",
        "protocol_sha256.txt",
    ]

    missing = [
        name
        for name in required
        if not (PACKAGE_DIR / name).exists()
    ]

    if missing:
        raise FileNotFoundError(
            f"Missing package files: {missing}"
        )

    X_self = np.load(
        PACKAGE_DIR / "X_self.npy",
        allow_pickle=False
    )

    y_hazard = np.load(
        PACKAGE_DIR / "y_hazard.npy",
        allow_pickle=False
    )

    hazard_mask = np.load(
        PACKAGE_DIR / "hazard_at_risk_mask.npy",
        allow_pickle=False
    )

    y_3_day = np.load(
        PACKAGE_DIR / "y_3_day.npy",
        allow_pickle=False
    )

    H_product_group = np.load(
        PACKAGE_DIR / "H_product_group.npy",
        allow_pickle=False
    )

    H_storage = np.load(
        PACKAGE_DIR / "H_storage_location.npy",
        allow_pickle=False
    )

    metadata = pd.read_csv(
        PACKAGE_DIR / "sample_metadata.csv"
    )

    for col in [
        "date",
        "prediction_origin_date",
        "sequence_start_date",
        "sequence_end_date",
    ]:
        if col in metadata.columns:
            metadata[col] = pd.to_datetime(
                metadata[col],
                errors="coerce"
            )

    split_file = np.load(
        PACKAGE_DIR / "split_indices.npz",
        allow_pickle=False
    )

    train_indices = split_file["train_indices"].astype(np.int64)
    validation_indices = split_file["validation_indices"].astype(np.int64)
    test_indices = split_file["test_indices"].astype(np.int64)

    with open(
        PACKAGE_DIR / "protocol_sha256.txt",
        "r",
        encoding="utf-8"
    ) as file:
        package_protocol_hash = file.read().strip()

    return {
        "X_self": X_self,
        "y_hazard": y_hazard,
        "hazard_mask": hazard_mask,
        "y_3_day": y_3_day,
        "metadata": metadata,
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "test_indices": test_indices,
        "H_product_group": H_product_group,
        "H_storage_location": H_storage,
        "package_protocol_hash": package_protocol_hash,
    }


def load_stage2_normalization():
    path = SELF_DIR / "train_only_normalization.json"

    if not path.exists():
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as file:
        record = json.load(file)

    mean = np.asarray(
        record["mean_on_log1p_scale"],
        dtype=np.float32
    )

    std = np.asarray(
        record["std_on_log1p_scale"],
        dtype=np.float32
    )

    if mean.shape[0] != INPUT_DIM:
        raise RuntimeError("Unexpected normalization mean length.")

    if std.shape[0] != INPUT_DIM:
        raise RuntimeError("Unexpected normalization std length.")

    return {
        "record": record,
        "mean": mean,
        "std": std,
    }


def transform_with_stage2_normalizer(X_raw, normalizer):
    X = X_raw.astype(np.float32)

    if (X < 0).any():
        raise RuntimeError("Negative operational values detected.")

    X = np.log1p(X)

    mean = normalizer["mean"][None, None, :]
    std = normalizer["std"][None, None, :]

    return ((X - mean) / std).astype(np.float32)


# ============================================================
# 4. Reconstruct raw all-node temporal tensor
# ============================================================

def compact_name(value):
    return (
        str(value)
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace(".", "")
    )


def find_supplygraph_directory():
    candidates = [
        BASE_DIR
        / "SupplyGraph-main"
        / "SupplyGraph-main"
        / "Raw Dataset"
        / "Homogenoeus",

        BASE_DIR
        / "SupplyGraph-main"
        / "Raw Dataset"
        / "Homogenoeus",

        BASE_DIR
        / "SupplyGraph-main"
        / "SupplyGraph-main"
        / "Raw Dataset"
        / "Homogeneous",

        BASE_DIR
        / "SupplyGraph-main"
        / "Raw Dataset"
        / "Homogeneous",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError("Cannot find SupplyGraph raw directory.")


def find_temporal_file(sg_dir, folder_name, keyword):
    folder = sg_dir / "Temporal Data" / folder_name

    if not folder.exists():
        raise FileNotFoundError(folder)

    for path in folder.glob("*.csv"):
        if keyword in compact_name(path.name):
            return path

    raise FileNotFoundError(
        f"Cannot find temporal file {keyword} in {folder}"
    )


def read_wide_csv(path):
    df = pd.read_csv(path)
    df.columns = [
        str(c).strip()
        for c in df.columns
    ]

    first_col = df.columns[0]
    df = df.rename(columns={first_col: "Date"})
    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df = (
        df
        .dropna(subset=["Date"])
        .sort_values("Date")
        .drop_duplicates(subset=["Date"], keep="first")
        .reset_index(drop=True)
    )

    return df


def load_raw_temporal_tensor(nodes):
    sg_dir = find_supplygraph_directory()

    paths = {
        "u_delivery": find_temporal_file(sg_dir, "Unit", "delivery"),
        "u_factory": find_temporal_file(sg_dir, "Unit", "factory"),
        "u_production": find_temporal_file(sg_dir, "Unit", "production"),
        "u_sales": find_temporal_file(sg_dir, "Unit", "sales"),
        "w_delivery": find_temporal_file(sg_dir, "Weight", "delivery"),
        "w_factory": find_temporal_file(sg_dir, "Weight", "factory"),
        "w_production": find_temporal_file(sg_dir, "Weight", "production"),
        "w_sales": find_temporal_file(sg_dir, "Weight", "sales"),
    }

    tables = {}

    for key in TEMPORAL_KEYS:
        tables[key] = read_wide_csv(paths[key])

    common_dates = None

    for key in TEMPORAL_KEYS:
        dates = set(tables[key]["Date"])
        if common_dates is None:
            common_dates = dates
        else:
            common_dates = common_dates.intersection(dates)

    common_dates = sorted(common_dates)

    if len(common_dates) == 0:
        raise RuntimeError("No common raw dates.")

    all_matrices = []

    for key in TEMPORAL_KEYS:
        df = tables[key]

        missing_nodes = [
            node
            for node in nodes
            if node not in df.columns
        ]

        if missing_nodes:
            raise RuntimeError(
                f"{key} missing nodes: {missing_nodes}"
            )

        selected = (
            df[df["Date"].isin(common_dates)]
            .sort_values("Date")
            .loc[:, ["Date"] + nodes]
            .reset_index(drop=True)
        )

        matrix = (
            selected[nodes]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
            .values
            .astype(np.float32)
        )

        all_matrices.append(matrix)

    # Shape: [T, N, F]
    raw_tensor = np.stack(
        all_matrices,
        axis=2
    )

    raw_dates = pd.DatetimeIndex(common_dates)

    return raw_tensor, raw_dates, paths


def build_relation_weight_matrix(H):
    """
    Convert a binary incidence matrix into a row-normalized
    hypergraph neighbor-averaging operator.

    Self is excluded.

    H shape:
        [N, E]

    W shape:
        [N, N]
    """

    H = np.asarray(H, dtype=np.float32)

    number_of_nodes = H.shape[0]
    number_of_edges = H.shape[1]

    W = np.zeros(
        (number_of_nodes, number_of_nodes),
        dtype=np.float32
    )

    incident_count = np.zeros(
        number_of_nodes,
        dtype=np.float32
    )

    for edge_index in range(number_of_edges):
        members = np.where(H[:, edge_index] > 0.5)[0]

        if len(members) < 2:
            continue

        for i in members:
            other_members = [
                j
                for j in members
                if j != i
            ]

            if len(other_members) == 0:
                continue

            weight = 1.0 / float(len(other_members))

            for j in other_members:
                W[i, j] += weight

            incident_count[i] += 1.0

    for i in range(number_of_nodes):
        if incident_count[i] > 0:
            W[i, :] = W[i, :] / incident_count[i]

    return W


def build_graph_sequence_tensor(
    metadata,
    raw_tensor,
    W,
    lookback_days=LOOKBACK_DAYS
):
    """
    Build one relation-specific graph temporal tensor.

    For each sample i,t:
        X_graph[i] =
        weighted average of other nodes' raw sequences
        over the same 14-day window.
    """

    sample_count = len(metadata)

    X_graph = np.zeros(
        (sample_count, lookback_days, INPUT_DIM),
        dtype=np.float32
    )

    for row_id, row in metadata.iterrows():
        node_index = int(row["node_index"])
        raw_time_index = int(row["raw_time_index"])

        start = raw_time_index - lookback_days + 1
        end = raw_time_index + 1

        if start < 0:
            raise RuntimeError("Insufficient history in graph tensor.")

        window = raw_tensor[start:end, :, :]

        if window.shape != (
            lookback_days,
            W.shape[0],
            INPUT_DIM
        ):
            raise RuntimeError(
                f"Unexpected graph window shape: {window.shape}"
            )

        weights = W[node_index, :]

        X_graph[row_id] = np.einsum(
            "n,tnf->tf",
            weights,
            window
        )

    return X_graph


# ============================================================
# 5. Self model and RAHT model
# ============================================================

class SelfGRUHazardModel(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        dropout
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.hazard_head = nn.Linear(hidden_dim, HAZARD_DAYS)

    def encode(self, x):
        _, hidden = self.gru(x)
        representation = hidden[-1]
        representation = self.layer_norm(representation)
        return representation

    def forward(self, x):
        representation = self.encode(x)
        representation = self.dropout(representation)
        hazard_logits = self.hazard_head(representation)
        return hazard_logits


class RAHTResidualHazardModel(nn.Module):
    def __init__(
        self,
        frozen_self_model,
        hidden_dim,
        graph_dropout
    ):
        super().__init__()

        self.self_model = frozen_self_model

        for param in self.self_model.parameters():
            param.requires_grad = False

        self.group_gru = nn.GRU(
            input_size=INPUT_DIM,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.storage_gru = nn.GRU(
            input_size=INPUT_DIM,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.group_norm = nn.LayerNorm(hidden_dim)
        self.storage_norm = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(graph_dropout)

        self.gate_layer = nn.Linear(
            hidden_dim * 3,
            2
        )

        self.group_delta_head = nn.Linear(
            hidden_dim,
            HAZARD_DAYS
        )

        self.storage_delta_head = nn.Linear(
            hidden_dim,
            HAZARD_DAYS
        )

        self.reset_residual_initialization()

    def reset_residual_initialization(self):
        nn.init.zeros_(self.group_delta_head.weight)
        nn.init.zeros_(self.group_delta_head.bias)

        nn.init.zeros_(self.storage_delta_head.weight)
        nn.init.zeros_(self.storage_delta_head.bias)

        nn.init.zeros_(self.gate_layer.weight)
        nn.init.constant_(
            self.gate_layer.bias,
            INITIAL_GATE_BIAS
        )

    def encode_graph(self, x, gru, norm):
        _, hidden = gru(x)
        representation = hidden[-1]
        representation = norm(representation)
        representation = self.dropout(representation)
        return representation

    def forward(
        self,
        x_self,
        x_group,
        x_storage
    ):
        with torch.no_grad():
            self_representation = self.self_model.encode(x_self)
            base_logits = self.self_model(x_self)

        group_representation = self.encode_graph(
            x_group,
            self.group_gru,
            self.group_norm
        )

        storage_representation = self.encode_graph(
            x_storage,
            self.storage_gru,
            self.storage_norm
        )

        gate_input = torch.cat(
            [
                self_representation,
                group_representation,
                storage_representation,
            ],
            dim=1
        )

        gates = torch.sigmoid(
            self.gate_layer(gate_input)
        )

        gate_group = gates[:, 0:1]
        gate_storage = gates[:, 1:2]

        delta_group = self.group_delta_head(
            group_representation
        )

        delta_storage = self.storage_delta_head(
            storage_representation
        )

        final_logits = (
            base_logits
            +
            gate_group * delta_group
            +
            gate_storage * delta_storage
        )

        return {
            "final_logits": final_logits,
            "base_logits": base_logits,
            "gate_group": gate_group,
            "gate_storage": gate_storage,
            "delta_group": delta_group,
            "delta_storage": delta_storage,
        }


# ============================================================
# 6. Losses
# ============================================================

def calculate_hazard_positive_weights(
    y_hazard,
    hazard_mask
):
    y_hazard = np.asarray(y_hazard, dtype=np.float32)
    hazard_mask = np.asarray(hazard_mask, dtype=np.float32)

    weights = []

    for day_index in range(HAZARD_DAYS):
        mask = hazard_mask[:, day_index]
        targets = y_hazard[:, day_index]

        positives = float((targets * mask).sum())
        negatives = float(((1.0 - targets) * mask).sum())

        if positives <= 0:
            raise RuntimeError("No positive hazard labels.")

        weights.append(
            negatives / positives
        )

    return np.asarray(weights, dtype=np.float32)


def masked_hazard_loss(
    logits,
    targets,
    mask,
    positive_weights
):
    elementwise = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=positive_weights,
        reduction="none",
    )

    masked = elementwise * mask

    denominator = torch.clamp(
        mask.sum(),
        min=1.0
    )

    return masked.sum() / denominator


def pairwise_ranking_loss(risk, y_three_day):
    y_three_day = y_three_day.view(-1)

    positive_risk = risk[y_three_day > 0.5]
    negative_risk = risk[y_three_day < 0.5]

    if (
        positive_risk.numel() == 0
        or
        negative_risk.numel() == 0
    ):
        return torch.tensor(
            0.0,
            dtype=risk.dtype,
            device=risk.device
        )

    diff = (
        negative_risk[:, None]
        -
        positive_risk[None, :]
    )

    return F.softplus(diff).mean()


def raht_training_loss(
    outputs,
    hazard_targets,
    hazard_mask,
    y_three_day,
    positive_weights
):
    final_logits = outputs["final_logits"]
    base_logits = outputs["base_logits"]

    hazard_loss = masked_hazard_loss(
        final_logits,
        hazard_targets,
        hazard_mask,
        positive_weights
    )

    hazards = torch.sigmoid(final_logits)
    risk = hazards_to_three_day_risk_torch(hazards)

    ranking_loss = pairwise_ranking_loss(
        risk,
        y_three_day
    )

    gate_l1 = (
        outputs["gate_group"].mean()
        +
        outputs["gate_storage"].mean()
    )

    delta_l2 = (
        outputs["delta_group"].pow(2).mean()
        +
        outputs["delta_storage"].pow(2).mean()
    )

    anchor_loss = F.mse_loss(
        final_logits,
        base_logits.detach()
    )

    total = (
        hazard_loss
        +
        RANKING_LOSS_WEIGHT * ranking_loss
        +
        GATE_L1_WEIGHT * gate_l1
        +
        DELTA_L2_WEIGHT * delta_l2
        +
        ANCHOR_LOSS_WEIGHT * anchor_loss
    )

    return {
        "total_loss": total,
        "hazard_loss": hazard_loss.detach(),
        "ranking_loss": ranking_loss.detach(),
        "gate_l1": gate_l1.detach(),
        "delta_l2": delta_l2.detach(),
        "anchor_loss": anchor_loss.detach(),
    }


# ============================================================
# 7. Data loaders and prediction
# ============================================================

def build_training_loader(
    X_self,
    X_group,
    X_storage,
    y_hazard,
    hazard_mask,
    y_three_day,
    seed
):
    dataset = TensorDataset(
        torch.from_numpy(X_self).float(),
        torch.from_numpy(X_group).float(),
        torch.from_numpy(X_storage).float(),
        torch.from_numpy(y_hazard).float(),
        torch.from_numpy(hazard_mask).float(),
        torch.from_numpy(y_three_day.astype(np.float32)).float(),
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        generator=generator,
        drop_last=False,
    )


def predict_raht(
    model,
    X_self,
    X_group,
    X_storage,
    device
):
    model.eval()

    dataset = TensorDataset(
        torch.from_numpy(X_self).float(),
        torch.from_numpy(X_group).float(),
        torch.from_numpy(X_storage).float(),
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    all_final_hazards = []
    all_base_hazards = []
    all_final_risk = []
    all_base_risk = []
    all_gate_group = []
    all_gate_storage = []
    all_delta_group = []
    all_delta_storage = []

    with torch.no_grad():
        for batch_self, batch_group, batch_storage in loader:
            batch_self = batch_self.to(device)
            batch_group = batch_group.to(device)
            batch_storage = batch_storage.to(device)

            outputs = model(
                batch_self,
                batch_group,
                batch_storage
            )

            final_hazards = torch.sigmoid(
                outputs["final_logits"]
            )

            base_hazards = torch.sigmoid(
                outputs["base_logits"]
            )

            final_risk = hazards_to_three_day_risk_torch(
                final_hazards
            )

            base_risk = hazards_to_three_day_risk_torch(
                base_hazards
            )

            all_final_hazards.append(
                final_hazards.cpu().numpy()
            )

            all_base_hazards.append(
                base_hazards.cpu().numpy()
            )

            all_final_risk.append(
                final_risk.cpu().numpy()
            )

            all_base_risk.append(
                base_risk.cpu().numpy()
            )

            all_gate_group.append(
                outputs["gate_group"].cpu().numpy()
            )

            all_gate_storage.append(
                outputs["gate_storage"].cpu().numpy()
            )

            all_delta_group.append(
                outputs["delta_group"].cpu().numpy()
            )

            all_delta_storage.append(
                outputs["delta_storage"].cpu().numpy()
            )

    final_hazards = np.concatenate(
        all_final_hazards,
        axis=0
    )

    base_hazards = np.concatenate(
        all_base_hazards,
        axis=0
    )

    final_risk = np.concatenate(
        all_final_risk,
        axis=0
    )

    base_risk = np.concatenate(
        all_base_risk,
        axis=0
    )

    gate_group = np.concatenate(
        all_gate_group,
        axis=0
    ).reshape(-1)

    gate_storage = np.concatenate(
        all_gate_storage,
        axis=0
    ).reshape(-1)

    delta_group = np.concatenate(
        all_delta_group,
        axis=0
    )

    delta_storage = np.concatenate(
        all_delta_storage,
        axis=0
    )

    return {
        "final_hazards": final_hazards,
        "base_hazards": base_hazards,
        "final_risk": final_risk,
        "base_risk": base_risk,
        "gate_group": gate_group,
        "gate_storage": gate_storage,
        "delta_group": delta_group,
        "delta_storage": delta_storage,
    }


# ============================================================
# 8. Load frozen self checkpoint
# ============================================================

def load_frozen_self_model(seed, device):
    checkpoint_path = (
        SELF_CHECKPOINT_DIR
        / f"self_hazard_seed_{seed}.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    checkpoint = safe_torch_load(
        checkpoint_path,
        device
    )

    config = checkpoint["config"]

    model = SelfGRUHazardModel(
        input_dim=INPUT_DIM,
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config["dropout"]),
    ).to(device)

    model.load_state_dict(
        checkpoint["state_dict"]
    )

    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    return model, checkpoint


# ============================================================
# 9. Train one RAHT seed
# ============================================================

def train_one_raht_seed(
    seed,
    package,
    arrays,
    y_3_day_validation,
    device
):
    set_global_seed(seed)

    frozen_self_model, self_checkpoint = load_frozen_self_model(
        seed,
        device
    )

    hidden_dim = int(
        self_checkpoint["config"]["hidden_dim"]
    )

    model = RAHTResidualHazardModel(
        frozen_self_model=frozen_self_model,
        hidden_dim=hidden_dim,
        graph_dropout=GRAPH_DROPOUT,
    ).to(device)

    trainable_parameters = [
        param
        for param in model.parameters()
        if param.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=RAHT_LEARNING_RATE,
        weight_decay=RAHT_WEIGHT_DECAY,
    )

    positive_weights_np = calculate_hazard_positive_weights(
        arrays["y_hazard_train"],
        arrays["hazard_mask_train"]
    )

    positive_weights = torch.tensor(
        positive_weights_np,
        dtype=torch.float32,
        device=device
    )

    loader = build_training_loader(
        arrays["X_self_train"],
        arrays["X_group_train"],
        arrays["X_storage_train"],
        arrays["y_hazard_train"],
        arrays["hazard_mask_train"],
        arrays["y_3_day_train"],
        seed,
    )

    best_ap = -np.inf
    best_epoch = 0
    best_state = None
    best_prediction = None
    best_metrics = None
    epochs_without_improvement = 0
    history_rows = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()

        total_loss_sum = 0.0
        total_mask_sum = 0.0

        hazard_loss_sum = 0.0
        ranking_loss_sum = 0.0
        gate_l1_sum = 0.0
        anchor_loss_sum = 0.0

        for (
            batch_self,
            batch_group,
            batch_storage,
            batch_y,
            batch_mask,
            batch_y3,
        ) in loader:
            batch_self = batch_self.to(device)
            batch_group = batch_group.to(device)
            batch_storage = batch_storage.to(device)
            batch_y = batch_y.to(device)
            batch_mask = batch_mask.to(device)
            batch_y3 = batch_y3.to(device)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(
                batch_self,
                batch_group,
                batch_storage
            )

            losses = raht_training_loss(
                outputs,
                batch_y,
                batch_mask,
                batch_y3,
                positive_weights
            )

            loss = losses["total_loss"]
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                max_norm=GRADIENT_CLIP_NORM
            )

            optimizer.step()

            mask_sum = float(batch_mask.sum().item())

            total_loss_sum += float(loss.item()) * mask_sum
            total_mask_sum += mask_sum

            hazard_loss_sum += float(losses["hazard_loss"].item()) * mask_sum
            ranking_loss_sum += float(losses["ranking_loss"].item()) * mask_sum
            gate_l1_sum += float(losses["gate_l1"].item()) * mask_sum
            anchor_loss_sum += float(losses["anchor_loss"].item()) * mask_sum

        denom = max(total_mask_sum, 1.0)

        validation_prediction = predict_raht(
            model,
            arrays["X_self_validation"],
            arrays["X_group_validation"],
            arrays["X_storage_validation"],
            device,
        )

        validation_metrics = evaluate_risk_predictions(
            y_3_day_validation,
            validation_prediction["final_risk"]
        )

        validation_ap = validation_metrics["average_precision"]

        history_rows.append({
            "seed": seed,
            "epoch": epoch,
            "training_total_loss": total_loss_sum / denom,
            "training_hazard_loss": hazard_loss_sum / denom,
            "training_ranking_loss": ranking_loss_sum / denom,
            "training_gate_l1": gate_l1_sum / denom,
            "training_anchor_loss": anchor_loss_sum / denom,
            "validation_average_precision": validation_ap,
            "validation_roc_auc": validation_metrics["roc_auc"],
            "validation_recall_at_top10": validation_metrics["recall_at_top10"],
            "mean_gate_group": float(validation_prediction["gate_group"].mean()),
            "mean_gate_storage": float(validation_prediction["gate_storage"].mean()),
        })

        if (
            np.isfinite(validation_ap)
            and
            validation_ap > best_ap + 1e-8
        ):
            best_ap = float(validation_ap)
            best_epoch = int(epoch)
            best_state = copy.deepcopy(
                model.state_dict()
            )
            best_prediction = {
                key: value.copy()
                for key, value in validation_prediction.items()
            }
            best_metrics = dict(validation_metrics)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (
            epoch >= MIN_EPOCHS
            and
            epochs_without_improvement >= EARLY_STOPPING_PATIENCE
        ):
            break

    if best_state is None:
        raise RuntimeError("No valid RAHT checkpoint found.")

    model.load_state_dict(best_state)

    return {
        "model": model,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "best_prediction": best_prediction,
        "history": pd.DataFrame(history_rows),
        "positive_weights": positive_weights_np,
        "self_checkpoint_config": self_checkpoint["config"],
    }


# ============================================================
# 10. Validation bootstrap comparison
# ============================================================

def date_block_bootstrap_improvement(
    validation_metadata,
    y_true,
    baseline_score,
    candidate_score,
    metric_fn,
    n_bootstrap=2000,
    random_state=42
):
    validation_metadata = (
        validation_metadata
        .reset_index(drop=True)
        .copy()
    )

    y_true = np.asarray(y_true).astype(int)
    baseline_score = np.asarray(baseline_score).astype(float)
    candidate_score = np.asarray(candidate_score).astype(float)

    dates = (
        pd.to_datetime(validation_metadata["date"])
        .drop_duplicates()
        .tolist()
    )

    date_to_positions = {}

    date_values = pd.to_datetime(
        validation_metadata["date"]
    ).values

    for date in dates:
        date_to_positions[date] = np.where(
            date_values == np.datetime64(date)
        )[0]

    rng = np.random.default_rng(random_state)
    differences = []

    for _ in range(n_bootstrap):
        sampled_dates = rng.choice(
            dates,
            size=len(dates),
            replace=True
        )

        sampled_indices = np.concatenate([
            date_to_positions[date]
            for date in sampled_dates
        ])

        y_boot = y_true[sampled_indices]

        if len(np.unique(y_boot)) < 2:
            continue

        base_metric = metric_fn(
            y_boot,
            baseline_score[sampled_indices]
        )

        candidate_metric = metric_fn(
            y_boot,
            candidate_score[sampled_indices]
        )

        if (
            np.isfinite(base_metric)
            and
            np.isfinite(candidate_metric)
        ):
            differences.append(
                candidate_metric - base_metric
            )

    differences = np.asarray(differences, dtype=float)

    if len(differences) == 0:
        return {
            "mean_improvement": np.nan,
            "ci_lower_95": np.nan,
            "ci_upper_95": np.nan,
            "probability_improvement_gt_0": np.nan,
            "nonpositive_resample_fraction": np.nan,
        }

    nonpositive_fraction = (
        1
        +
        np.sum(differences <= 0)
    ) / (
        len(differences)
        +
        1
    )

    return {
        "mean_improvement": float(differences.mean()),
        "ci_lower_95": float(np.quantile(differences, 0.025)),
        "ci_upper_95": float(np.quantile(differences, 0.975)),
        "probability_improvement_gt_0": float((differences > 0).mean()),
        "nonpositive_resample_fraction": float(nonpositive_fraction),
    }


def recall_top10_metric(y, score):
    return topk_metrics(
        y,
        score
    )["recall_at_top10"]


# ============================================================
# 11. Main training
# ============================================================

def main():
    print("==============================================")
    print("STAGE 3: TRAIN RAHT-RH RESIDUAL HAZARD MODEL")
    print("==============================================")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\nDevice:", device)

    if device.type == "cpu":
        torch.set_num_threads(int(os.environ.get("PORTFOLIO_TORCH_THREADS", "1")))

    protocol, protocol_hash = load_locked_protocol()
    package = load_package()
    normalizer = load_stage2_normalization()

    if protocol_hash != package["package_protocol_hash"]:
        raise RuntimeError(
            "Current protocol hash does not match Stage 1 package."
        )

    locked_seeds = [
        int(seed)
        for seed in protocol["random_seeds"]
    ]

    print("\n========== LOCKED INPUTS ==========")
    print("Protocol SHA256:", protocol_hash)
    print("X_self shape:", package["X_self"].shape)
    print("Train samples:", len(package["train_indices"]))
    print("Validation samples:", len(package["validation_indices"]))
    print("Test samples locked:", len(package["test_indices"]))
    print("Primary relations:", protocol["primary_relations"])
    print("Locked seeds:", locked_seeds)

    # --------------------------------------------------------
    # Node order
    # --------------------------------------------------------

    node_table = (
        package["metadata"][
            [
                "node_index",
                "node",
            ]
        ]
        .drop_duplicates()
        .sort_values("node_index")
    )

    nodes = node_table["node"].astype(str).tolist()

    if len(nodes) != 41:
        raise RuntimeError("Expected 41 nodes.")

    # --------------------------------------------------------
    # Build graph sequence tensors
    # --------------------------------------------------------

    print("\n========== BUILD GRAPH TEMPORAL TENSORS ==========")

    raw_tensor, raw_dates, raw_paths = load_raw_temporal_tensor(nodes)

    print("Raw temporal tensor shape:", raw_tensor.shape)
    print("Raw dates:", raw_dates.min(), "to", raw_dates.max())

    W_group = build_relation_weight_matrix(
        package["H_product_group"]
    )

    W_storage = build_relation_weight_matrix(
        package["H_storage_location"]
    )

    print("Product Group W shape:", W_group.shape)
    print("Storage W shape:", W_storage.shape)
    print("Product Group row-sum nonzero:", int((W_group.sum(axis=1) > 0).sum()))
    print("Storage row-sum nonzero:", int((W_storage.sum(axis=1) > 0).sum()))

    X_group_raw = build_graph_sequence_tensor(
        package["metadata"],
        raw_tensor,
        W_group,
        LOOKBACK_DAYS
    )

    X_storage_raw = build_graph_sequence_tensor(
        package["metadata"],
        raw_tensor,
        W_storage,
        LOOKBACK_DAYS
    )

    X_self_norm = transform_with_stage2_normalizer(
        package["X_self"],
        normalizer
    )

    X_group_norm = transform_with_stage2_normalizer(
        X_group_raw,
        normalizer
    )

    X_storage_norm = transform_with_stage2_normalizer(
        X_storage_raw,
        normalizer
    )

    print("X_group shape:", X_group_norm.shape)
    print("X_storage shape:", X_storage_norm.shape)

    # --------------------------------------------------------
    # Official splits
    # --------------------------------------------------------

    train_indices = package["train_indices"]
    validation_indices = package["validation_indices"]

    arrays = {
        "X_self_train": X_self_norm[train_indices],
        "X_group_train": X_group_norm[train_indices],
        "X_storage_train": X_storage_norm[train_indices],
        "y_hazard_train": package["y_hazard"][train_indices],
        "hazard_mask_train": package["hazard_mask"][train_indices],
        "y_3_day_train": package["y_3_day"][train_indices],

        "X_self_validation": X_self_norm[validation_indices],
        "X_group_validation": X_group_norm[validation_indices],
        "X_storage_validation": X_storage_norm[validation_indices],
    }

    y_3_day_validation = package["y_3_day"][validation_indices]

    validation_metadata = (
        package["metadata"]
        .iloc[validation_indices]
        .copy()
        .reset_index(drop=True)
    )

    print("\n========== RAHT FIXED CONFIGURATION ==========")
    print("learning_rate:", RAHT_LEARNING_RATE)
    print("weight_decay:", RAHT_WEIGHT_DECAY)
    print("graph_dropout:", GRAPH_DROPOUT)
    print("gate_l1_weight:", GATE_L1_WEIGHT)
    print("delta_l2_weight:", DELTA_L2_WEIGHT)
    print("anchor_loss_weight:", ANCHOR_LOSS_WEIGHT)
    print("ranking_loss_weight:", RANKING_LOSS_WEIGHT)
    print("initial_gate_bias:", INITIAL_GATE_BIAS)

    raht_config = {
        "method": "RAHT-RH",
        "learning_rate": RAHT_LEARNING_RATE,
        "weight_decay": RAHT_WEIGHT_DECAY,
        "graph_dropout": GRAPH_DROPOUT,
        "gate_l1_weight": GATE_L1_WEIGHT,
        "delta_l2_weight": DELTA_L2_WEIGHT,
        "anchor_loss_weight": ANCHOR_LOSS_WEIGHT,
        "ranking_loss_weight": RANKING_LOSS_WEIGHT,
        "initial_gate_bias": INITIAL_GATE_BIAS,
        "primary_relations": [
            "product_group",
            "storage_location",
        ],
        "self_backbone": "Frozen Stage 2 Self-GRU Hazard Baseline",
    }

    save_json(
        raht_config,
        OUT_DIR / "raht_fixed_config.json"
    )

    # --------------------------------------------------------
    # Train one RAHT model per locked seed
    # --------------------------------------------------------

    seed_metric_rows = []
    all_history = []
    all_predictions = []
    checkpoint_records = []

    all_raht_risks = []
    all_raht_hazards = []
    all_base_risks = []
    all_gate_group = []
    all_gate_storage = []

    print("\n==============================================")
    print("FINAL TEN-SEED RAHT VALIDATION DEVELOPMENT")
    print("==============================================")

    for seed in locked_seeds:
        print("\n----------------------------------------------")
        print("Training RAHT seed:", seed)

        result = train_one_raht_seed(
            seed,
            package,
            arrays,
            y_3_day_validation,
            device
        )

        prediction = result["best_prediction"]
        metrics = result["best_metrics"]

        all_raht_risks.append(
            prediction["final_risk"]
        )

        all_raht_hazards.append(
            prediction["final_hazards"]
        )

        all_base_risks.append(
            prediction["base_risk"]
        )

        all_gate_group.append(
            prediction["gate_group"]
        )

        all_gate_storage.append(
            prediction["gate_storage"]
        )

        seed_metric_rows.append({
            "seed": seed,
            "best_epoch": result["best_epoch"],
            "validation_average_precision": metrics["average_precision"],
            "validation_roc_auc": metrics["roc_auc"],
            "validation_precision_at_top10": metrics["precision_at_top10"],
            "validation_recall_at_top10": metrics["recall_at_top10"],
            "validation_lift_at_top10": metrics["lift_at_top10"],
            "mean_gate_group": float(prediction["gate_group"].mean()),
            "mean_gate_storage": float(prediction["gate_storage"].mean()),
            "mean_abs_delta_group": float(np.abs(prediction["delta_group"]).mean()),
            "mean_abs_delta_storage": float(np.abs(prediction["delta_storage"]).mean()),
        })

        history = result["history"].copy()
        all_history.append(history)

        pred_df = validation_metadata[
            [
                "sample_id",
                "date",
                "node",
                "node_index",
                "split",
                "event_day_1_to_3",
                "next_3_day_onset_label",
            ]
        ].copy()

        pred_df["seed"] = seed
        pred_df["raht_hazard_day1"] = prediction["final_hazards"][:, 0]
        pred_df["raht_hazard_day2"] = prediction["final_hazards"][:, 1]
        pred_df["raht_hazard_day3"] = prediction["final_hazards"][:, 2]
        pred_df["raht_three_day_risk"] = prediction["final_risk"]
        pred_df["base_three_day_risk_from_frozen_self"] = prediction["base_risk"]
        pred_df["gate_product_group"] = prediction["gate_group"]
        pred_df["gate_storage_location"] = prediction["gate_storage"]
        pred_df["delta_group_day1"] = prediction["delta_group"][:, 0]
        pred_df["delta_group_day2"] = prediction["delta_group"][:, 1]
        pred_df["delta_group_day3"] = prediction["delta_group"][:, 2]
        pred_df["delta_storage_day1"] = prediction["delta_storage"][:, 0]
        pred_df["delta_storage_day2"] = prediction["delta_storage"][:, 1]
        pred_df["delta_storage_day3"] = prediction["delta_storage"][:, 2]

        pred_path = (
            PREDICTION_DIR
            / f"validation_raht_seed_{seed}_predictions.csv"
        )

        pred_df.to_csv(
            pred_path,
            index=False,
            encoding="utf-8-sig"
        )

        all_predictions.append(pred_df)

        checkpoint_path = (
            CHECKPOINT_DIR
            / f"raht_residual_seed_{seed}.pt"
        )

        checkpoint = {
            "stage": "Stage 3 RAHT-RH Residual Hazard Model",
            "method": "RAHT-RH",
            "seed": int(seed),
            "raht_config": raht_config,
            "best_epoch": int(result["best_epoch"]),
            "validation_metrics": metrics,
            "positive_weights": result["positive_weights"].tolist(),
            "self_checkpoint_config": result["self_checkpoint_config"],
            "state_dict": result["model"].state_dict(),
        }

        torch.save(
            checkpoint,
            checkpoint_path
        )

        checkpoint_records.append({
            "seed": seed,
            "checkpoint_filename": checkpoint_path.name,
            "checkpoint_sha256": sha256_file(checkpoint_path),
        })

        print("Validation AP:", f"{metrics['average_precision']:.6f}")
        print("Validation Recall@Top10:", f"{metrics['recall_at_top10']:.6f}")
        print("Mean Product Group gate:", f"{prediction['gate_group'].mean():.6f}")
        print("Mean Storage gate:", f"{prediction['gate_storage'].mean():.6f}")
        print("Best epoch:", result["best_epoch"])

    # --------------------------------------------------------
    # Ensemble
    # --------------------------------------------------------

    raht_risk_stack = np.stack(
        all_raht_risks,
        axis=0
    )

    raht_hazard_stack = np.stack(
        all_raht_hazards,
        axis=0
    )

    base_risk_stack = np.stack(
        all_base_risks,
        axis=0
    )

    gate_group_stack = np.stack(
        all_gate_group,
        axis=0
    )

    gate_storage_stack = np.stack(
        all_gate_storage,
        axis=0
    )

    raht_ensemble_risk = raht_risk_stack.mean(axis=0)
    raht_ensemble_hazards = raht_hazard_stack.mean(axis=0)
    frozen_self_ensemble_risk = base_risk_stack.mean(axis=0)
    gate_group_ensemble = gate_group_stack.mean(axis=0)
    gate_storage_ensemble = gate_storage_stack.mean(axis=0)

    raht_ensemble_metrics = evaluate_risk_predictions(
        y_3_day_validation,
        raht_ensemble_risk
    )

    frozen_self_metrics = evaluate_risk_predictions(
        y_3_day_validation,
        frozen_self_ensemble_risk
    )

    # Load official Stage 2 saved ensemble for comparison.
    self_ensemble_path = (
        SELF_PREDICTION_DIR
        / "validation_ensemble_predictions.csv"
    )

    if not self_ensemble_path.exists():
        raise FileNotFoundError(self_ensemble_path)

    self_ensemble_saved = pd.read_csv(
        self_ensemble_path
    )

    saved_self_risk = (
        self_ensemble_saved[
            "ensemble_three_day_risk"
        ]
        .astype(float)
        .values
    )

    saved_self_metrics = evaluate_risk_predictions(
        y_3_day_validation,
        saved_self_risk
    )

    # --------------------------------------------------------
    # Bootstrap comparison on validation
    # --------------------------------------------------------

    bootstrap_rows = []

    bootstrap_rows.append({
        "comparison": "RAHT-RH ensemble vs Stage 2 Self-GRU ensemble",
        "metric": "average_precision",
        **date_block_bootstrap_improvement(
            validation_metadata,
            y_3_day_validation,
            saved_self_risk,
            raht_ensemble_risk,
            safe_average_precision,
        )
    })

    bootstrap_rows.append({
        "comparison": "RAHT-RH ensemble vs Stage 2 Self-GRU ensemble",
        "metric": "recall_at_top10",
        **date_block_bootstrap_improvement(
            validation_metadata,
            y_3_day_validation,
            saved_self_risk,
            raht_ensemble_risk,
            recall_top10_metric,
        )
    })

    bootstrap_df = pd.DataFrame(bootstrap_rows)

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    seed_metrics = pd.DataFrame(seed_metric_rows)
    training_history = pd.concat(all_history, ignore_index=True)
    checkpoint_manifest = pd.DataFrame(checkpoint_records)

    seed_metrics.to_csv(
        OUT_DIR / "raht_seed_metrics_validation.csv",
        index=False,
        encoding="utf-8-sig"
    )

    training_history.to_csv(
        OUT_DIR / "raht_training_history_all_seeds.csv",
        index=False,
        encoding="utf-8-sig"
    )

    checkpoint_manifest.to_csv(
        OUT_DIR / "raht_checkpoint_manifest.csv",
        index=False,
        encoding="utf-8-sig"
    )

    bootstrap_df.to_csv(
        OUT_DIR / "raht_vs_self_validation_bootstrap.csv",
        index=False,
        encoding="utf-8-sig"
    )

    ensemble_pred_df = validation_metadata[
        [
            "sample_id",
            "date",
            "node",
            "node_index",
            "split",
            "event_day_1_to_3",
            "next_3_day_onset_label",
        ]
    ].copy()

    ensemble_pred_df["self_stage2_ensemble_risk"] = saved_self_risk
    ensemble_pred_df["raht_ensemble_hazard_day1"] = raht_ensemble_hazards[:, 0]
    ensemble_pred_df["raht_ensemble_hazard_day2"] = raht_ensemble_hazards[:, 1]
    ensemble_pred_df["raht_ensemble_hazard_day3"] = raht_ensemble_hazards[:, 2]
    ensemble_pred_df["raht_ensemble_three_day_risk"] = raht_ensemble_risk
    ensemble_pred_df["frozen_self_recomputed_risk"] = frozen_self_ensemble_risk
    ensemble_pred_df["gate_product_group_mean"] = gate_group_ensemble
    ensemble_pred_df["gate_storage_location_mean"] = gate_storage_ensemble
    ensemble_pred_df["ensemble_seed_count"] = len(locked_seeds)

    ensemble_pred_df.to_csv(
        PREDICTION_DIR / "validation_raht_ensemble_predictions.csv",
        index=False,
        encoding="utf-8-sig"
    )

    comparison_rows = [
        {
            "model": "Stage 2 Self-GRU ensemble",
            **saved_self_metrics,
        },
        {
            "model": "Frozen self branch recomputed inside RAHT",
            **frozen_self_metrics,
        },
        {
            "model": "RAHT-RH ensemble",
            **raht_ensemble_metrics,
        },
    ]

    comparison_df = pd.DataFrame(comparison_rows)

    comparison_df.to_csv(
        OUT_DIR / "raht_vs_self_validation_comparison.csv",
        index=False,
        encoding="utf-8-sig"
    )

    gate_analysis = pd.DataFrame([
        {
            "group": "all_validation_samples",
            "sample_count": len(ensemble_pred_df),
            "mean_gate_product_group": float(gate_group_ensemble.mean()),
            "mean_gate_storage_location": float(gate_storage_ensemble.mean()),
            "median_gate_product_group": float(np.median(gate_group_ensemble)),
            "median_gate_storage_location": float(np.median(gate_storage_ensemble)),
        },
        {
            "group": "positive_3day_samples",
            "sample_count": int((y_3_day_validation == 1).sum()),
            "mean_gate_product_group": float(gate_group_ensemble[y_3_day_validation == 1].mean()),
            "mean_gate_storage_location": float(gate_storage_ensemble[y_3_day_validation == 1].mean()),
            "median_gate_product_group": float(np.median(gate_group_ensemble[y_3_day_validation == 1])),
            "median_gate_storage_location": float(np.median(gate_storage_ensemble[y_3_day_validation == 1])),
        },
        {
            "group": "negative_3day_samples",
            "sample_count": int((y_3_day_validation == 0).sum()),
            "mean_gate_product_group": float(gate_group_ensemble[y_3_day_validation == 0].mean()),
            "mean_gate_storage_location": float(gate_storage_ensemble[y_3_day_validation == 0].mean()),
            "median_gate_product_group": float(np.median(gate_group_ensemble[y_3_day_validation == 0])),
            "median_gate_storage_location": float(np.median(gate_storage_ensemble[y_3_day_validation == 0])),
        },
    ])

    gate_analysis.to_csv(
        OUT_DIR / "gate_analysis_validation.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Stage 3 audit
    # --------------------------------------------------------

    audit_rows = []

    add_audit_check(
        audit_rows,
        "current_protocol_matches_stage1_package",
        protocol_hash == package["package_protocol_hash"],
        f"current={protocol_hash}; package={package['package_protocol_hash']}",
    )

    add_audit_check(
        audit_rows,
        "primary_hypergraphs_loaded",
        (
            package["H_product_group"].shape[0] == 41
            and
            package["H_storage_location"].shape[0] == 41
        ),
        {
            "H_product_group": package["H_product_group"].shape,
            "H_storage_location": package["H_storage_location"].shape,
        },
    )

    add_audit_check(
        audit_rows,
        "all_ten_stage2_self_checkpoints_exist",
        all(
            (
                SELF_CHECKPOINT_DIR
                / f"self_hazard_seed_{seed}.pt"
            ).exists()
            for seed in locked_seeds
        ),
        locked_seeds,
    )

    add_audit_check(
        audit_rows,
        "all_ten_locked_raht_seeds_trained",
        set(seed_metrics["seed"].astype(int)) == set(locked_seeds),
        seed_metrics["seed"].astype(int).tolist(),
    )

    add_audit_check(
        audit_rows,
        "ensemble_uses_all_ten_locked_seeds",
        set(ensemble_pred_df["ensemble_seed_count"].astype(int).unique()) == {10},
        ensemble_pred_df["ensemble_seed_count"].astype(int).unique().tolist(),
    )

    add_audit_check(
        audit_rows,
        "validation_prediction_count_matches_validation_split",
        len(ensemble_pred_df) == len(validation_indices),
        f"predictions={len(ensemble_pred_df)}; validation={len(validation_indices)}",
    )

    add_audit_check(
        audit_rows,
        "raht_predictions_are_finite",
        (
            np.isfinite(raht_ensemble_risk).all()
            and
            np.isfinite(raht_ensemble_hazards).all()
        ),
        "RAHT ensemble risk and hazards are finite.",
    )

    add_audit_check(
        audit_rows,
        "raht_probabilities_are_inside_zero_one_interval",
        (
            (raht_ensemble_risk >= 0.0).all()
            and
            (raht_ensemble_risk <= 1.0).all()
            and
            (raht_ensemble_hazards >= 0.0).all()
            and
            (raht_ensemble_hazards <= 1.0).all()
        ),
        "All RAHT risks and hazards are in [0, 1].",
    )

    add_audit_check(
        audit_rows,
        "gate_values_are_inside_zero_one_interval",
        (
            (gate_group_ensemble >= 0.0).all()
            and
            (gate_group_ensemble <= 1.0).all()
            and
            (gate_storage_ensemble >= 0.0).all()
            and
            (gate_storage_ensemble <= 1.0).all()
        ),
        "All dynamic gate values are in [0, 1].",
    )

    test_prediction_files = list(
        PREDICTION_DIR.glob("*test*")
    )

    add_audit_check(
        audit_rows,
        "no_test_prediction_file_was_created",
        len(test_prediction_files) == 0,
        [path.name for path in test_prediction_files],
    )

    add_audit_check(
        audit_rows,
        "test_predictive_performance_not_calculated",
        True,
        (
            "Test split was not transformed, predicted, scored, "
            "or used for early stopping or checkpoint selection."
        ),
    )

    audit = pd.DataFrame(audit_rows)
    failed = audit[audit["status"] == "FAIL"].copy()

    audit.to_csv(
        OUT_DIR / "stage_3_audit_report.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n==============================================")
    print("STAGE 3 STRICT AUDIT")
    print("==============================================")
    print(audit.to_string(index=False))

    print("\nAudit summary:")
    print("PASS:", int((audit["status"] == "PASS").sum()))
    print("FAIL:", int((audit["status"] == "FAIL").sum()))

    if len(failed) > 0:
        print("\nFailed checks:")
        print(failed.to_string(index=False))
        raise RuntimeError("Stage 3 audit failed.")

    print("\n==============================================")
    print("STAGE 3 APPROVED: RAHT-RH VALIDATION MODEL TRAINED")
    print("==============================================")

    print("\nValidation model comparison:")
    print(comparison_df.to_string(index=False))

    print("\nValidation bootstrap comparison:")
    print(bootstrap_df.to_string(index=False))

    print("\nGate analysis:")
    print(gate_analysis.to_string(index=False))

    print("\nRAHT seed metrics:")
    print(seed_metrics.to_string(index=False))

    print("\nResults saved to:")
    print(OUT_DIR)

    print("\nImportant files:")
    print(" - raht_seed_metrics_validation.csv")
    print(" - raht_vs_self_validation_comparison.csv")
    print(" - raht_vs_self_validation_bootstrap.csv")
    print(" - gate_analysis_validation.csv")
    print(" - raht_training_history_all_seeds.csv")
    print(" - raht_checkpoint_manifest.csv")
    print(" - stage_3_audit_report.csv")
    print(" - predictions/validation_raht_ensemble_predictions.csv")
    print(" - checkpoints/raht_residual_seed_*.pt")

    print("\nIMPORTANT:")
    print("No locked-test prediction or locked-test performance was calculated.")


if __name__ == "__main__":
    main()
