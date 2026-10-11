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
# 0. Paths and locked configuration
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

OUT_DIR = (
    BASE_DIR
    / "self_hazard_baseline_results"
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
# Locked training settings
# ------------------------------------------------------------

SELECTION_SEED = 42

N_ROLLING_FOLDS = 3

INPUT_DIM = 8

HAZARD_DAYS = 3

DROPOUT = 0.20

BATCH_SIZE = 256

MAX_EPOCHS = 120

MIN_EPOCHS = 15

EARLY_STOPPING_PATIENCE = 15

GRADIENT_CLIP_NORM = 5.0

TOP_ALERT_RATE = 0.10

NUM_WORKERS = 0


# ------------------------------------------------------------
# Exactly eight pre-defined configurations
# ------------------------------------------------------------

HIDDEN_DIMS = [
    32,
    64,
]

LEARNING_RATES = [
    0.0003,
    0.001,
]

WEIGHT_DECAYS = [
    0.0001,
    0.001,
]


# ============================================================
# 1. General utilities
# ============================================================

def set_global_seed(seed):
    """
    Set random seeds for reproducibility.
    """

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )

    try:

        torch.use_deterministic_algorithms(
            True,
            warn_only=True
        )

    except Exception:

        pass

    if hasattr(
        torch.backends,
        "cudnn"
    ):

        torch.backends.cudnn.deterministic = True

        torch.backends.cudnn.benchmark = False


def sha256_file(path):
    """
    Calculate SHA256 checksum.
    """

    hash_object = hashlib.sha256()

    with open(
        path,
        "rb"
    ) as file:

        while True:

            block = file.read(
                1024 * 1024
            )

            if not block:
                break

            hash_object.update(
                block
            )

    return hash_object.hexdigest()


def sha256_integer_array(values):
    """
    Calculate SHA256 checksum of an integer index array.
    """

    values = np.asarray(
        values,
        dtype=np.int64
    )

    return hashlib.sha256(
        values.tobytes()
    ).hexdigest()


def save_json(
    obj,
    path
):
    """
    Save JSON in UTF-8.
    """

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            obj,
            file,
            indent=2,
            ensure_ascii=False
        )


def add_audit_check(
    rows,
    check_name,
    passed,
    details
):
    """
    Add one audit check.
    """

    rows.append({

        "check_name":
            check_name,

        "status":
            "PASS"
            if passed
            else "FAIL",

        "details":
            str(
                details
            ),
    })


# ============================================================
# 2. Load and verify locked inputs
# ============================================================

def load_locked_protocol():
    """
    Load current locked research protocol.
    """

    if not PROTOCOL_FILE.exists():

        raise FileNotFoundError(
            f"Cannot find locked protocol:\n"
            f"{PROTOCOL_FILE}"
        )

    with open(
        PROTOCOL_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        protocol = json.load(
            file
        )

    protocol_hash = sha256_file(
        PROTOCOL_FILE
    )

    if (
        protocol[
            "project_name"
        ]
        !=
        "RAHT-RH-CRC"
    ):

        raise RuntimeError(
            "Unexpected project name in locked protocol."
        )

    if (
        protocol[
            "primary_metric"
        ]
        !=
        "average_precision"
    ):

        raise RuntimeError(
            "Primary metric must remain locked as "
            "average_precision."
        )

    if (
        int(
            protocol[
                "temporal_lookback_days"
            ]
        )
        !=
        14
    ):

        raise RuntimeError(
            "Temporal lookback must remain locked at 14 days."
        )

    return (
        protocol,
        protocol_hash,
    )


def verify_package_manifest():
    """
    Verify every package file listed in the manifest.
    """

    manifest_path = (
        PACKAGE_DIR
        / "package_manifest.json"
    )

    if not manifest_path.exists():

        raise FileNotFoundError(
            f"Cannot find package manifest:\n"
            f"{manifest_path}"
        )

    with open(
        manifest_path,
        "r",
        encoding="utf-8"
    ) as file:

        manifest = json.load(
            file
        )

    mismatches = []

    checked_files = 0

    for item in manifest.get(
        "package_files",
        []
    ):

        filename = item[
            "filename"
        ]

        expected_hash = item[
            "sha256"
        ]

        path = (
            PACKAGE_DIR
            / filename
        )

        if not path.exists():

            mismatches.append(
                f"Missing file: {filename}"
            )

            continue

        actual_hash = sha256_file(
            path
        )

        checked_files += 1

        if (
            actual_hash
            !=
            expected_hash
        ):

            mismatches.append(
                f"Checksum mismatch: {filename}"
            )

    return (
        manifest,
        checked_files,
        mismatches,
    )


def load_training_package():
    """
    Load locked Stage 1 package.

    Test indices are loaded only to verify index separation.
    Test feature tensors and test labels are never selected,
    transformed, evaluated, scored, or predicted.
    """

    required_files = [

        "X_self.npy",

        "y_hazard.npy",

        "hazard_at_risk_mask.npy",

        "y_next_day.npy",

        "y_3_day.npy",

        "sample_metadata.csv",

        "split_indices.npz",

        "feature_names.json",

        "protocol_sha256.txt",

        "package_manifest.json",
    ]

    missing = [

        filename

        for filename
        in required_files

        if not (
            PACKAGE_DIR
            / filename
        ).exists()
    ]

    if missing:

        raise FileNotFoundError(
            "Missing required Stage 1 package files:\n"
            f"{missing}"
        )

    X_self = np.load(

        PACKAGE_DIR
        / "X_self.npy",

        allow_pickle=False
    )

    y_hazard = np.load(

        PACKAGE_DIR
        / "y_hazard.npy",

        allow_pickle=False
    )

    hazard_mask = np.load(

        PACKAGE_DIR
        / "hazard_at_risk_mask.npy",

        allow_pickle=False
    )

    y_next_day = np.load(

        PACKAGE_DIR
        / "y_next_day.npy",

        allow_pickle=False
    )

    y_3_day = np.load(

        PACKAGE_DIR
        / "y_3_day.npy",

        allow_pickle=False
    )

    metadata = pd.read_csv(

        PACKAGE_DIR
        / "sample_metadata.csv"
    )

    metadata[
        "date"
    ] = pd.to_datetime(
        metadata[
            "date"
        ],
        errors="coerce"
    )

    metadata[
        "sequence_start_date"
    ] = pd.to_datetime(
        metadata[
            "sequence_start_date"
        ],
        errors="coerce"
    )

    metadata[
        "sequence_end_date"
    ] = pd.to_datetime(
        metadata[
            "sequence_end_date"
        ],
        errors="coerce"
    )

    metadata[
        "prediction_origin_date"
    ] = pd.to_datetime(
        metadata[
            "prediction_origin_date"
        ],
        errors="coerce"
    )

    split_file = np.load(

        PACKAGE_DIR
        / "split_indices.npz",

        allow_pickle=False
    )

    train_indices = split_file[
        "train_indices"
    ].astype(
        np.int64
    )

    validation_indices = split_file[
        "validation_indices"
    ].astype(
        np.int64
    )

    test_indices = split_file[
        "test_indices"
    ].astype(
        np.int64
    )

    with open(

        PACKAGE_DIR
        / "feature_names.json",

        "r",
        encoding="utf-8"
    ) as file:

        feature_definition = json.load(
            file
        )

    with open(

        PACKAGE_DIR
        / "protocol_sha256.txt",

        "r",
        encoding="utf-8"
    ) as file:

        package_protocol_hash = (
            file.read()
            .strip()
        )

    return {

        "X_self":
            X_self,

        "y_hazard":
            y_hazard,

        "hazard_mask":
            hazard_mask,

        "y_next_day":
            y_next_day,

        "y_3_day":
            y_3_day,

        "metadata":
            metadata,

        "train_indices":
            train_indices,

        "validation_indices":
            validation_indices,

        "test_indices":
            test_indices,

        "feature_definition":
            feature_definition,

        "package_protocol_hash":
            package_protocol_hash,
    }


# ============================================================
# 3. Metrics
# ============================================================

def safe_average_precision(
    y_true,
    score
):
    """
    Calculate average precision safely.
    """

    y_true = np.asarray(
        y_true
    ).astype(
        int
    )

    score = np.asarray(
        score
    ).astype(
        float
    )

    if (
        len(
            np.unique(
                y_true
            )
        )
        <
        2
    ):

        return np.nan

    return float(
        average_precision_score(
            y_true,
            score
        )
    )


def safe_roc_auc(
    y_true,
    score
):
    """
    Calculate ROC-AUC safely.
    """

    y_true = np.asarray(
        y_true
    ).astype(
        int
    )

    score = np.asarray(
        score
    ).astype(
        float
    )

    if (
        len(
            np.unique(
                y_true
            )
        )
        <
        2
    ):

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
    """
    Fixed alert-budget metrics.
    """

    y_true = np.asarray(
        y_true
    ).astype(
        int
    )

    score = np.asarray(
        score
    ).astype(
        float
    )

    sample_count = len(
        y_true
    )

    k = max(
        1,
        int(
            math.floor(
                sample_count
                *
                top_rate
            )
        )
    )

    ranked_indices = np.argsort(
        score
    )[::-1]

    top_indices = ranked_indices[
        :k
    ]

    selected_positive_count = int(
        y_true[
            top_indices
        ].sum()
    )

    total_positive_count = int(
        y_true.sum()
    )

    precision = float(
        selected_positive_count
        /
        k
    )

    if (
        total_positive_count
        >
        0
    ):

        recall = float(
            selected_positive_count
            /
            total_positive_count
        )

    else:

        recall = np.nan

    prevalence = float(
        y_true.mean()
    )

    lift = float(
        precision
        /
        max(
            prevalence,
            1e-12
        )
    )

    return {

        "precision_at_top10":
            precision,

        "recall_at_top10":
            recall,

        "lift_at_top10":
            lift,

        "top_alert_count":
            int(
                k
            ),
    }


def evaluate_risk_predictions(
    y_true,
    risk_score
):
    """
    Evaluate 3-day cumulative risk.
    """

    ap = safe_average_precision(
        y_true,
        risk_score
    )

    auc = safe_roc_auc(
        y_true,
        risk_score
    )

    top_metrics = topk_metrics(
        y_true,
        risk_score
    )

    return {

        "average_precision":
            ap,

        "roc_auc":
            auc,

        "precision_at_top10":
            top_metrics[
                "precision_at_top10"
            ],

        "recall_at_top10":
            top_metrics[
                "recall_at_top10"
            ],

        "lift_at_top10":
            top_metrics[
                "lift_at_top10"
            ],

        "top_alert_count":
            top_metrics[
                "top_alert_count"
            ],
    }


# ============================================================
# 4. Train-only normalization
# ============================================================

def fit_normalizer(
    X_raw,
    source_indices
):
    """
    Fit signal-wise normalization using selected training rows only.

    Transformation:
        raw non-negative quantity
            ->
        log1p
            ->
        signal-wise z-score

    Mean and standard deviation are calculated across:
        sample dimension
        and
        time dimension

    No validation or test row participates.
    """

    source_indices = np.asarray(
        source_indices,
        dtype=np.int64
    )

    X_source = X_raw[
        source_indices
    ].astype(
        np.float64
    )

    negative_count = int(
        (
            X_source
            <
            0
        ).sum()
    )

    if (
        negative_count
        >
        0
    ):

        raise RuntimeError(
            "Negative operational values detected. "
            f"Count={negative_count}"
        )

    X_log = np.log1p(
        X_source
    )

    mean = X_log.mean(
        axis=(
            0,
            1,
        )
    )

    std = X_log.std(
        axis=(
            0,
            1,
        )
    )

    std = np.where(
        std
        <
        1e-6,
        1.0,
        std
    )

    return {

        "mean":
            mean.astype(
                np.float32
            ),

        "std":
            std.astype(
                np.float32
            ),

        "source_sample_count":
            int(
                len(
                    source_indices
                )
            ),

        "source_indices_sha256":
            sha256_integer_array(
                source_indices
            ),
    }


def transform_sequences(
    X_raw,
    indices,
    normalizer
):
    """
    Apply a previously fitted train-only normalizer.
    """

    indices = np.asarray(
        indices,
        dtype=np.int64
    )

    X = X_raw[
        indices
    ].astype(
        np.float32
    )

    if (
        X
        <
        0
    ).any():

        raise RuntimeError(
            "Negative operational values detected "
            "during transformation."
        )

    X = np.log1p(
        X
    )

    mean = normalizer[
        "mean"
    ][
        None,
        None,
        :
    ]

    std = normalizer[
        "std"
    ][
        None,
        None,
        :
    ]

    X = (
        X
        -
        mean
    ) / std

    return X.astype(
        np.float32
    )


# ============================================================
# 5. Rolling-origin folds inside training period
# ============================================================

def build_rolling_origin_folds(
    metadata,
    train_indices,
    y_3_day,
    n_folds=N_ROLLING_FOLDS
):
    """
    Build three expanding-window rolling-origin folds.

    Date design:
        Initial training window = first 50% of training dates.

        Remaining dates are divided into three consecutive
        validation blocks.

    Fold 1:
        first 50% dates -> next block

    Fold 2:
        first 50% + block 1 -> next block

    Fold 3:
        first 50% + blocks 1 and 2 -> final block

    Every fold satisfies:
        maximum training date
        <
        minimum validation date
    """

    train_indices = np.asarray(
        train_indices,
        dtype=np.int64
    )

    train_metadata = (
        metadata.iloc[
            train_indices
        ]
        .copy()
    )

    unique_dates = sorted(
        pd.to_datetime(
            train_metadata[
                "date"
            ]
        )
        .dropna()
        .unique()
    )

    number_of_dates = len(
        unique_dates
    )

    if (
        number_of_dates
        <
        30
    ):

        raise RuntimeError(
            "Too few training dates for rolling-origin folds."
        )

    initial_train_count = int(
        math.floor(
            number_of_dates
            *
            0.50
        )
    )

    initial_train_count = max(
        initial_train_count,
        14
    )

    remaining_date_count = (
        number_of_dates
        -
        initial_train_count
    )

    base_validation_block = int(
        math.floor(
            remaining_date_count
            /
            n_folds
        )
    )

    if (
        base_validation_block
        <
        1
    ):

        raise RuntimeError(
            "Cannot create rolling-origin validation blocks."
        )

    train_index_set = set(
        train_indices.tolist()
    )

    folds = []

    for fold_number in range(
        1,
        n_folds
        +
        1
    ):

        train_end_position = (
            initial_train_count
            +
            (
                fold_number
                -
                1
            )
            *
            base_validation_block
        )

        validation_start_position = (
            train_end_position
        )

        if (
            fold_number
            <
            n_folds
        ):

            validation_end_position = (
                validation_start_position
                +
                base_validation_block
            )

        else:

            validation_end_position = (
                number_of_dates
            )

        fold_train_dates = set(
            unique_dates[
                :
                train_end_position
            ]
        )

        fold_validation_dates = set(
            unique_dates[
                validation_start_position
                :
                validation_end_position
            ]
        )

        fold_train_indices = np.array(
            [
                index

                for index
                in train_indices

                if metadata.iloc[
                    index
                ][
                    "date"
                ]
                in
                fold_train_dates
            ],
            dtype=np.int64
        )

        fold_validation_indices = np.array(
            [
                index

                for index
                in train_indices

                if metadata.iloc[
                    index
                ][
                    "date"
                ]
                in
                fold_validation_dates
            ],
            dtype=np.int64
        )

        if (
            len(
                fold_train_indices
            )
            ==
            0
            or
            len(
                fold_validation_indices
            )
            ==
            0
        ):

            raise RuntimeError(
                f"Rolling fold {fold_number} is empty."
            )

        if not set(
            fold_train_indices.tolist()
        ).issubset(
            train_index_set
        ):

            raise RuntimeError(
                "Rolling fold training indices escaped "
                "the official training split."
            )

        if not set(
            fold_validation_indices.tolist()
        ).issubset(
            train_index_set
        ):

            raise RuntimeError(
                "Rolling fold validation indices escaped "
                "the official training split."
            )

        train_date_max = metadata.iloc[
            fold_train_indices
        ][
            "date"
        ].max()

        validation_date_min = metadata.iloc[
            fold_validation_indices
        ][
            "date"
        ].min()

        if not (
            train_date_max
            <
            validation_date_min
        ):

            raise RuntimeError(
                "Rolling-origin temporal order violation."
            )

        train_positive_count = int(
            y_3_day[
                fold_train_indices
            ].sum()
        )

        validation_positive_count = int(
            y_3_day[
                fold_validation_indices
            ].sum()
        )

        if (
            train_positive_count
            <=
            0
        ):

            raise RuntimeError(
                f"Fold {fold_number} training window "
                "contains no positive event."
            )

        if (
            validation_positive_count
            <=
            0
        ):

            raise RuntimeError(
                f"Fold {fold_number} validation window "
                "contains no positive event."
            )

        folds.append({

            "fold":
                fold_number,

            "train_indices":
                fold_train_indices,

            "validation_indices":
                fold_validation_indices,

            "train_date_min":
                metadata.iloc[
                    fold_train_indices
                ][
                    "date"
                ].min(),

            "train_date_max":
                train_date_max,

            "validation_date_min":
                validation_date_min,

            "validation_date_max":
                metadata.iloc[
                    fold_validation_indices
                ][
                    "date"
                ].max(),

            "train_sample_count":
                int(
                    len(
                        fold_train_indices
                    )
                ),

            "validation_sample_count":
                int(
                    len(
                        fold_validation_indices
                    )
                ),

            "train_positive_count":
                train_positive_count,

            "validation_positive_count":
                validation_positive_count,
        })

    return folds


# ============================================================
# 6. Hyperparameter grid
# ============================================================

def build_hyperparameter_grid():
    """
    Build exactly eight pre-defined configurations.
    """

    configurations = []

    configuration_number = 0

    for hidden_dim in HIDDEN_DIMS:

        for learning_rate in LEARNING_RATES:

            for weight_decay in WEIGHT_DECAYS:

                configuration_number += 1

                config_id = (
                    f"CFG_{configuration_number:02d}"
                )

                configurations.append({

                    "config_id":
                        config_id,

                    "hidden_dim":
                        int(
                            hidden_dim
                        ),

                    "learning_rate":
                        float(
                            learning_rate
                        ),

                    "weight_decay":
                        float(
                            weight_decay
                        ),

                    "dropout":
                        float(
                            DROPOUT
                        ),
                })

    if (
        len(
            configurations
        )
        !=
        8
    ):

        raise RuntimeError(
            "Hyperparameter grid must contain exactly 8 "
            "pre-defined configurations."
        )

    return configurations


# ============================================================
# 7. Self-GRU hazard model
# ============================================================

class SelfGRUHazardModel(
    nn.Module
):
    """
    Strong self-history discrete-time hazard baseline.

    Input:
        [batch, 14, 8]

    Output:
        three hazard logits:
        Day+1
        Day+2
        Day+3
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        dropout
    ):

        super().__init__()

        self.gru = nn.GRU(

            input_size=
                input_dim,

            hidden_size=
                hidden_dim,

            num_layers=
                1,

            batch_first=
                True,
        )

        self.layer_norm = (
            nn.LayerNorm(
                hidden_dim
            )
        )

        self.dropout = (
            nn.Dropout(
                dropout
            )
        )

        self.hazard_head = (
            nn.Linear(
                hidden_dim,
                HAZARD_DAYS
            )
        )

    def forward(
        self,
        x
    ):

        _, hidden = self.gru(
            x
        )

        representation = hidden[
            -1
        ]

        representation = self.layer_norm(
            representation
        )

        representation = self.dropout(
            representation
        )

        hazard_logits = self.hazard_head(
            representation
        )

        return hazard_logits


# ============================================================
# 8. Hazard loss and cumulative risk
# ============================================================

def calculate_hazard_positive_weights(
    y_hazard,
    hazard_mask
):
    """
    Calculate horizon-specific positive weights using
    training rows only.
    """

    y_hazard = np.asarray(
        y_hazard,
        dtype=np.float32
    )

    hazard_mask = np.asarray(
        hazard_mask,
        dtype=np.float32
    )

    weights = []

    for day_index in range(
        HAZARD_DAYS
    ):

        mask = hazard_mask[
            :,
            day_index
        ]

        targets = y_hazard[
            :,
            day_index
        ]

        positive_count = float(
            (
                targets
                *
                mask
            ).sum()
        )

        negative_count = float(
            (
                (
                    1.0
                    -
                    targets
                )
                *
                mask
            ).sum()
        )

        if (
            positive_count
            <=
            0
        ):

            raise RuntimeError(
                "A hazard horizon contains no training positives."
            )

        positive_weight = (
            negative_count
            /
            positive_count
        )

        weights.append(
            positive_weight
        )

    return np.asarray(
        weights,
        dtype=np.float32
    )


def masked_hazard_loss(
    hazard_logits,
    hazard_targets,
    hazard_mask,
    positive_weights
):
    """
    Masked weighted binary cross entropy over at-risk periods.
    """

    elementwise_loss = (
        F.binary_cross_entropy_with_logits(

            hazard_logits,

            hazard_targets,

            pos_weight=
                positive_weights,

            reduction=
                "none",
        )
    )

    masked_loss = (
        elementwise_loss
        *
        hazard_mask
    )

    denominator = torch.clamp(
        hazard_mask.sum(),
        min=1.0
    )

    return (
        masked_loss.sum()
        /
        denominator
    )


def hazards_to_three_day_risk(
    hazard_probabilities
):
    """
    Convert Day+1, Day+2, Day+3 hazards into cumulative 3-day risk.

    P(T <= 3)
    =
    1 - product(1 - hazard_k)
    """

    survival = np.prod(
        1.0
        -
        hazard_probabilities,
        axis=1
    )

    risk = 1.0 - survival

    return np.clip(
        risk,
        0.0,
        1.0
    )


# ============================================================
# 9. Data loaders
# ============================================================

def build_training_loader(
    X_train,
    y_train,
    mask_train,
    seed
):
    """
    Create deterministic shuffled training loader.
    """

    dataset = TensorDataset(

        torch.from_numpy(
            X_train
        ).float(),

        torch.from_numpy(
            y_train
        ).float(),

        torch.from_numpy(
            mask_train
        ).float(),
    )

    generator = torch.Generator()

    generator.manual_seed(
        seed
    )

    loader = DataLoader(

        dataset,

        batch_size=
            BATCH_SIZE,

        shuffle=
            True,

        num_workers=
            NUM_WORKERS,

        generator=
            generator,

        drop_last=
            False,
    )

    return loader


# ============================================================
# 10. Prediction
# ============================================================

def predict_hazards(
    model,
    X,
    device
):
    """
    Predict hazard probabilities and cumulative 3-day risk.
    """

    model.eval()

    dataset = TensorDataset(

        torch.from_numpy(
            X
        ).float()
    )

    loader = DataLoader(

        dataset,

        batch_size=
            BATCH_SIZE,

        shuffle=
            False,

        num_workers=
            NUM_WORKERS,
    )

    all_hazards = []

    with torch.no_grad():

        for (
            batch_x,
        ) in loader:

            batch_x = batch_x.to(
                device
            )

            logits = model(
                batch_x
            )

            hazards = torch.sigmoid(
                logits
            )

            all_hazards.append(

                hazards
                .detach()
                .cpu()
                .numpy()
            )

    hazard_probabilities = np.concatenate(
        all_hazards,
        axis=0
    )

    three_day_risk = hazards_to_three_day_risk(
        hazard_probabilities
    )

    return (
        hazard_probabilities,
        three_day_risk,
    )


# ============================================================
# 11. Train one model run
# ============================================================

def train_one_model(
    X_train,
    y_hazard_train,
    mask_train,
    y_3_day_train,
    X_validation,
    y_3_day_validation,
    config,
    seed,
    device
):
    """
    Train one GRU hazard model with validation AP early stopping.
    """

    set_global_seed(
        seed
    )

    model = SelfGRUHazardModel(

        input_dim=
            INPUT_DIM,

        hidden_dim=
            config[
                "hidden_dim"
            ],

        dropout=
            config[
                "dropout"
            ],
    ).to(
        device
    )

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=
            config[
                "learning_rate"
            ],

        weight_decay=
            config[
                "weight_decay"
            ],
    )

    positive_weight_values = (
        calculate_hazard_positive_weights(
            y_hazard_train,
            mask_train
        )
    )

    positive_weights = torch.tensor(

        positive_weight_values,

        dtype=torch.float32,

        device=device
    )

    training_loader = build_training_loader(

        X_train,
        y_hazard_train,
        mask_train,
        seed
    )

    best_validation_ap = -np.inf

    best_validation_metrics = None

    best_validation_hazards = None

    best_validation_risk = None

    best_state_dict = None

    best_epoch = 0

    epochs_without_improvement = 0

    history_rows = []

    for epoch in range(
        1,
        MAX_EPOCHS
        +
        1
    ):

        model.train()

        epoch_loss_sum = 0.0

        epoch_mask_sum = 0.0

        for (
            batch_x,
            batch_y,
            batch_mask,
        ) in training_loader:

            batch_x = batch_x.to(
                device
            )

            batch_y = batch_y.to(
                device
            )

            batch_mask = batch_mask.to(
                device
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                batch_x
            )

            loss = masked_hazard_loss(

                logits,

                batch_y,

                batch_mask,

                positive_weights,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(

                model.parameters(),

                max_norm=
                    GRADIENT_CLIP_NORM,
            )

            optimizer.step()

            effective_mask = float(
                batch_mask.sum().item()
            )

            epoch_loss_sum += (
                float(
                    loss.item()
                )
                *
                effective_mask
            )

            epoch_mask_sum += (
                effective_mask
            )

        training_loss = (
            epoch_loss_sum
            /
            max(
                epoch_mask_sum,
                1.0
            )
        )

        (
            validation_hazards,
            validation_risk,
        ) = predict_hazards(

            model,
            X_validation,
            device
        )

        validation_metrics = (
            evaluate_risk_predictions(

                y_3_day_validation,

                validation_risk,
            )
        )

        validation_ap = (
            validation_metrics[
                "average_precision"
            ]
        )

        history_rows.append({

            "epoch":
                epoch,

            "training_loss":
                training_loss,

            "validation_average_precision":
                validation_ap,

            "validation_roc_auc":
                validation_metrics[
                    "roc_auc"
                ],

            "validation_recall_at_top10":
                validation_metrics[
                    "recall_at_top10"
                ],
        })

        if (
            np.isfinite(
                validation_ap
            )
            and
            validation_ap
            >
            best_validation_ap
            +
            1e-8
        ):

            best_validation_ap = float(
                validation_ap
            )

            best_validation_metrics = dict(
                validation_metrics
            )

            best_validation_hazards = (
                validation_hazards.copy()
            )

            best_validation_risk = (
                validation_risk.copy()
            )

            best_state_dict = copy.deepcopy(
                model.state_dict()
            )

            best_epoch = int(
                epoch
            )

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        if (
            epoch
            >=
            MIN_EPOCHS
            and
            epochs_without_improvement
            >=
            EARLY_STOPPING_PATIENCE
        ):

            break

    if (
        best_state_dict
        is
        None
    ):

        raise RuntimeError(
            "No valid checkpoint was found."
        )

    model.load_state_dict(
        best_state_dict
    )

    return {

        "model":
            model,

        "best_epoch":
            best_epoch,

        "best_validation_metrics":
            best_validation_metrics,

        "best_validation_hazards":
            best_validation_hazards,

        "best_validation_risk":
            best_validation_risk,

        "history":
            pd.DataFrame(
                history_rows
            ),

        "positive_weights":
            positive_weight_values,
    }


# ============================================================
# 12. Hyperparameter selection
# ============================================================

def run_rolling_origin_selection(
    package,
    folds,
    configurations,
    device
):
    """
    Mechanically select one configuration using:
        training split only
        three rolling-origin folds
        fixed seed = 42
        mean fold average precision
    """

    X_raw = package[
        "X_self"
    ]

    y_hazard = package[
        "y_hazard"
    ]

    hazard_mask = package[
        "hazard_mask"
    ]

    y_3_day = package[
        "y_3_day"
    ]

    fold_result_rows = []

    summary_rows = []

    print(
        "\n=============================================="
    )

    print(
        "ROLLING-ORIGIN HYPERPARAMETER SELECTION"
    )

    print(
        "=============================================="
    )

    print(
        "Selection seed:",
        SELECTION_SEED
    )

    print(
        "Configurations:",
        len(
            configurations
        )
    )

    print(
        "Rolling folds:",
        len(
            folds
        )
    )

    for configuration in configurations:

        config_id = configuration[
            "config_id"
        ]

        print(
            "\n----------------------------------------------"
        )

        print(
            "Configuration:",
            config_id
        )

        print(
            configuration
        )

        fold_aps = []

        fold_recalls = []

        fold_aucs = []

        for fold in folds:

            fold_number = fold[
                "fold"
            ]

            fold_train_indices = fold[
                "train_indices"
            ]

            fold_validation_indices = fold[
                "validation_indices"
            ]

            print(
                f"  Fold {fold_number}: "
                f"train={len(fold_train_indices)}, "
                f"validation={len(fold_validation_indices)}"
            )

            normalizer = fit_normalizer(

                X_raw,
                fold_train_indices
            )

            X_train = transform_sequences(

                X_raw,
                fold_train_indices,
                normalizer
            )

            X_validation = transform_sequences(

                X_raw,
                fold_validation_indices,
                normalizer
            )

            result = train_one_model(

                X_train=
                    X_train,

                y_hazard_train=
                    y_hazard[
                        fold_train_indices
                    ],

                mask_train=
                    hazard_mask[
                        fold_train_indices
                    ],

                y_3_day_train=
                    y_3_day[
                        fold_train_indices
                    ],

                X_validation=
                    X_validation,

                y_3_day_validation=
                    y_3_day[
                        fold_validation_indices
                    ],

                config=
                    configuration,

                seed=
                    SELECTION_SEED,

                device=
                    device,
            )

            metrics = result[
                "best_validation_metrics"
            ]

            fold_ap = float(
                metrics[
                    "average_precision"
                ]
            )

            fold_recall = float(
                metrics[
                    "recall_at_top10"
                ]
            )

            fold_auc = float(
                metrics[
                    "roc_auc"
                ]
            )

            fold_aps.append(
                fold_ap
            )

            fold_recalls.append(
                fold_recall
            )

            fold_aucs.append(
                fold_auc
            )

            fold_result_rows.append({

                "config_id":
                    config_id,

                "hidden_dim":
                    configuration[
                        "hidden_dim"
                    ],

                "learning_rate":
                    configuration[
                        "learning_rate"
                    ],

                "weight_decay":
                    configuration[
                        "weight_decay"
                    ],

                "dropout":
                    configuration[
                        "dropout"
                    ],

                "selection_seed":
                    SELECTION_SEED,

                "fold":
                    fold_number,

                "train_date_min":
                    fold[
                        "train_date_min"
                    ],

                "train_date_max":
                    fold[
                        "train_date_max"
                    ],

                "validation_date_min":
                    fold[
                        "validation_date_min"
                    ],

                "validation_date_max":
                    fold[
                        "validation_date_max"
                    ],

                "train_sample_count":
                    fold[
                        "train_sample_count"
                    ],

                "validation_sample_count":
                    fold[
                        "validation_sample_count"
                    ],

                "train_positive_count":
                    fold[
                        "train_positive_count"
                    ],

                "validation_positive_count":
                    fold[
                        "validation_positive_count"
                    ],

                "best_epoch":
                    result[
                        "best_epoch"
                    ],

                "validation_average_precision":
                    fold_ap,

                "validation_roc_auc":
                    fold_auc,

                "validation_recall_at_top10":
                    fold_recall,

                "positive_weight_day1":
                    result[
                        "positive_weights"
                    ][
                        0
                    ],

                "positive_weight_day2":
                    result[
                        "positive_weights"
                    ][
                        1
                    ],

                "positive_weight_day3":
                    result[
                        "positive_weights"
                    ][
                        2
                    ],
            })

            print(
                f"    AP={fold_ap:.6f}, "
                f"Recall@Top10={fold_recall:.6f}, "
                f"best_epoch={result['best_epoch']}"
            )

        mean_ap = float(
            np.mean(
                fold_aps
            )
        )

        std_ap = float(
            np.std(
                fold_aps,
                ddof=0
            )
        )

        median_ap = float(
            np.median(
                fold_aps
            )
        )

        mean_recall = float(
            np.mean(
                fold_recalls
            )
        )

        mean_auc = float(
            np.mean(
                fold_aucs
            )
        )

        summary_rows.append({

            "config_id":
                config_id,

            "hidden_dim":
                configuration[
                    "hidden_dim"
                ],

            "learning_rate":
                configuration[
                    "learning_rate"
                ],

            "weight_decay":
                configuration[
                    "weight_decay"
                ],

            "dropout":
                configuration[
                    "dropout"
                ],

            "selection_seed":
                SELECTION_SEED,

            "mean_rolling_average_precision":
                mean_ap,

            "std_rolling_average_precision":
                std_ap,

            "median_rolling_average_precision":
                median_ap,

            "mean_rolling_roc_auc":
                mean_auc,

            "mean_rolling_recall_at_top10":
                mean_recall,
        })

        print(
            f"  Mean rolling AP={mean_ap:.6f}"
        )

    fold_results = pd.DataFrame(
        fold_result_rows
    )

    config_results = pd.DataFrame(
        summary_rows
    )

    # Mechanical selection rule:
    # 1. Highest mean rolling AP.
    # 2. Lower AP standard deviation.
    # 3. Lexicographically smaller config ID.

    ranked = (
        config_results
        .sort_values(

            by=[
                "mean_rolling_average_precision",
                "std_rolling_average_precision",
                "config_id",
            ],

            ascending=[
                False,
                True,
                True,
            ]
        )
        .reset_index(
            drop=True
        )
    )

    selected_row = ranked.iloc[
        0
    ]

    selected_config = {

        "config_id":
            str(
                selected_row[
                    "config_id"
                ]
            ),

        "hidden_dim":
            int(
                selected_row[
                    "hidden_dim"
                ]
            ),

        "learning_rate":
            float(
                selected_row[
                    "learning_rate"
                ]
            ),

        "weight_decay":
            float(
                selected_row[
                    "weight_decay"
                ]
            ),

        "dropout":
            float(
                selected_row[
                    "dropout"
                ]
            ),

        "selection_seed":
            SELECTION_SEED,

        "selection_method":
            (
                "Highest mean average precision across "
                "three expanding-window rolling-origin folds "
                "inside the official training split."
            ),

        "tie_break_rule":
            (
                "Lower AP standard deviation, then "
                "lexicographically smaller config ID."
            ),

        "selected_mean_rolling_average_precision":
            float(
                selected_row[
                    "mean_rolling_average_precision"
                ]
            ),

        "selected_std_rolling_average_precision":
            float(
                selected_row[
                    "std_rolling_average_precision"
                ]
            ),
    }

    return (
        fold_results,
        config_results,
        selected_config,
    )


# ============================================================
# 13. Final ten-seed validation development
# ============================================================

def run_final_seed_training(
    package,
    selected_config,
    protocol,
    device
):
    """
    Train the selected configuration using all official training rows.

    Each locked seed:
        - uses identical selected hyperparameters,
        - uses identical train-only normalization,
        - uses the same official validation split,
        - is early-stopped using validation AP,
        - saves one validation prediction file,
        - saves one checkpoint.

    No test prediction is generated.
    """

    X_raw = package[
        "X_self"
    ]

    y_hazard = package[
        "y_hazard"
    ]

    hazard_mask = package[
        "hazard_mask"
    ]

    y_3_day = package[
        "y_3_day"
    ]

    metadata = package[
        "metadata"
    ]

    train_indices = package[
        "train_indices"
    ]

    validation_indices = package[
        "validation_indices"
    ]

    locked_seeds = [
        int(
            seed
        )
        for seed
        in protocol[
            "random_seeds"
        ]
    ]

    if (
        len(
            locked_seeds
        )
        !=
        10
    ):

        raise RuntimeError(
            "Locked protocol must contain exactly 10 random seeds."
        )

    # --------------------------------------------------------
    # Full training-split-only normalization
    # --------------------------------------------------------

    normalizer = fit_normalizer(

        X_raw,
        train_indices
    )

    X_train = transform_sequences(

        X_raw,
        train_indices,
        normalizer
    )

    X_validation = transform_sequences(

        X_raw,
        validation_indices,
        normalizer
    )

    y_hazard_train = y_hazard[
        train_indices
    ]

    mask_train = hazard_mask[
        train_indices
    ]

    y_3_day_train = y_3_day[
        train_indices
    ]

    y_3_day_validation = y_3_day[
        validation_indices
    ]

    seed_metric_rows = []

    all_history_rows = []

    all_validation_risks = []

    all_validation_hazards = []

    checkpoint_records = []

    print(
        "\n=============================================="
    )

    print(
        "FINAL TEN-SEED VALIDATION DEVELOPMENT"
    )

    print(
        "=============================================="
    )

    print(
        "Selected configuration:"
    )

    print(
        selected_config
    )

    print(
        "\nLocked seeds:"
    )

    print(
        locked_seeds
    )

    print(
        "\nTraining samples:",
        len(
            train_indices
        )
    )

    print(
        "Training 3-day positives:",
        int(
            y_3_day_train.sum()
        )
    )

    print(
        "Validation samples:",
        len(
            validation_indices
        )
    )

    print(
        "Validation 3-day positives:",
        int(
            y_3_day_validation.sum()
        )
    )

    for seed in locked_seeds:

        print(
            "\n----------------------------------------------"
        )

        print(
            "Training seed:",
            seed
        )

        result = train_one_model(

            X_train=
                X_train,

            y_hazard_train=
                y_hazard_train,

            mask_train=
                mask_train,

            y_3_day_train=
                y_3_day_train,

            X_validation=
                X_validation,

            y_3_day_validation=
                y_3_day_validation,

            config=
                selected_config,

            seed=
                seed,

            device=
                device,
        )

        metrics = result[
            "best_validation_metrics"
        ]

        hazards = result[
            "best_validation_hazards"
        ]

        risk = result[
            "best_validation_risk"
        ]

        all_validation_hazards.append(
            hazards
        )

        all_validation_risks.append(
            risk
        )

        seed_metric_rows.append({

            "seed":
                seed,

            "config_id":
                selected_config[
                    "config_id"
                ],

            "hidden_dim":
                selected_config[
                    "hidden_dim"
                ],

            "learning_rate":
                selected_config[
                    "learning_rate"
                ],

            "weight_decay":
                selected_config[
                    "weight_decay"
                ],

            "dropout":
                selected_config[
                    "dropout"
                ],

            "best_epoch":
                result[
                    "best_epoch"
                ],

            "validation_average_precision":
                metrics[
                    "average_precision"
                ],

            "validation_roc_auc":
                metrics[
                    "roc_auc"
                ],

            "validation_precision_at_top10":
                metrics[
                    "precision_at_top10"
                ],

            "validation_recall_at_top10":
                metrics[
                    "recall_at_top10"
                ],

            "validation_lift_at_top10":
                metrics[
                    "lift_at_top10"
                ],
        })

        history = result[
            "history"
        ].copy()

        history.insert(
            0,
            "seed",
            seed
        )

        history.insert(
            1,
            "config_id",
            selected_config[
                "config_id"
            ]
        )

        all_history_rows.append(
            history
        )

        # ----------------------------------------------------
        # Save validation predictions for this seed
        # ----------------------------------------------------

        prediction_df = (
            metadata.iloc[
                validation_indices
            ][
                [
                    "sample_id",
                    "date",
                    "node",
                    "node_index",
                    "split",
                    "event_day_1_to_3",
                    "next_day_onset_label",
                    "next_3_day_onset_label",
                ]
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        prediction_df[
            "hazard_day1"
        ] = hazards[
            :,
            0
        ]

        prediction_df[
            "hazard_day2"
        ] = hazards[
            :,
            1
        ]

        prediction_df[
            "hazard_day3"
        ] = hazards[
            :,
            2
        ]

        prediction_df[
            "three_day_risk"
        ] = risk

        prediction_df[
            "seed"
        ] = seed

        prediction_path = (

            PREDICTION_DIR
            /
            (
                f"validation_seed_"
                f"{seed}_predictions.csv"
            )
        )

        prediction_df.to_csv(

            prediction_path,

            index=False,

            encoding="utf-8-sig"
        )

        # ----------------------------------------------------
        # Save checkpoint
        # ----------------------------------------------------

        checkpoint_path = (

            CHECKPOINT_DIR
            /
            (
                f"self_hazard_seed_"
                f"{seed}.pt"
            )
        )

        checkpoint = {

            "stage":
                "Stage 2 Self-GRU Hazard Baseline",

            "model_class":
                "SelfGRUHazardModel",

            "seed":
                int(
                    seed
                ),

            "config":
                dict(
                    selected_config
                ),

            "best_epoch":
                int(
                    result[
                        "best_epoch"
                    ]
                ),

            "validation_metrics":
                dict(
                    metrics
                ),

            "positive_weights":
                result[
                    "positive_weights"
                ].tolist(),

            "normalization_mean":
                normalizer[
                    "mean"
                ].tolist(),

            "normalization_std":
                normalizer[
                    "std"
                ].tolist(),

            "normalization_source_indices_sha256":
                normalizer[
                    "source_indices_sha256"
                ],

            "state_dict":
                result[
                    "model"
                ].state_dict(),
        }

        torch.save(

            checkpoint,

            checkpoint_path
        )

        checkpoint_records.append({

            "seed":
                seed,

            "checkpoint_filename":
                checkpoint_path.name,

            "checkpoint_sha256":
                sha256_file(
                    checkpoint_path
                ),
        })

        print(
            f"Validation AP="
            f"{metrics['average_precision']:.6f}"
        )

        print(
            f"Validation Recall@Top10="
            f"{metrics['recall_at_top10']:.6f}"
        )

        print(
            f"Best epoch="
            f"{result['best_epoch']}"
        )

    # --------------------------------------------------------
    # All-ten-seed ensemble
    # --------------------------------------------------------

    stacked_risks = np.stack(
        all_validation_risks,
        axis=0
    )

    stacked_hazards = np.stack(
        all_validation_hazards,
        axis=0
    )

    ensemble_risk = stacked_risks.mean(
        axis=0
    )

    ensemble_hazards = stacked_hazards.mean(
        axis=0
    )

    ensemble_metrics = evaluate_risk_predictions(

        y_3_day_validation,

        ensemble_risk
    )

    ensemble_prediction_df = (
        metadata.iloc[
            validation_indices
        ][
            [
                "sample_id",
                "date",
                "node",
                "node_index",
                "split",
                "event_day_1_to_3",
                "next_day_onset_label",
                "next_3_day_onset_label",
            ]
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    ensemble_prediction_df[
        "ensemble_hazard_day1"
    ] = ensemble_hazards[
        :,
        0
    ]

    ensemble_prediction_df[
        "ensemble_hazard_day2"
    ] = ensemble_hazards[
        :,
        1
    ]

    ensemble_prediction_df[
        "ensemble_hazard_day3"
    ] = ensemble_hazards[
        :,
        2
    ]

    ensemble_prediction_df[
        "ensemble_three_day_risk"
    ] = ensemble_risk

    ensemble_prediction_df[
        "ensemble_seed_count"
    ] = len(
        locked_seeds
    )

    ensemble_prediction_df.to_csv(

        PREDICTION_DIR
        / "validation_ensemble_predictions.csv",

        index=False,

        encoding="utf-8-sig"
    )

    seed_metrics = pd.DataFrame(
        seed_metric_rows
    )

    training_history = pd.concat(

        all_history_rows,

        ignore_index=True
    )

    checkpoint_manifest = pd.DataFrame(
        checkpoint_records
    )

    return {

        "seed_metrics":
            seed_metrics,

        "training_history":
            training_history,

        "checkpoint_manifest":
            checkpoint_manifest,

        "ensemble_metrics":
            ensemble_metrics,

        "ensemble_prediction_df":
            ensemble_prediction_df,

        "ensemble_risk":
            ensemble_risk,

        "ensemble_hazards":
            ensemble_hazards,

        "normalizer":
            normalizer,

        "locked_seeds":
            locked_seeds,
    }


# ============================================================
# 14. Build validation summary
# ============================================================

def build_validation_summary(
    seed_metrics,
    ensemble_metrics,
    selected_config
):
    """
    Summarize ten-seed stability and ensemble performance.
    """

    rows = []

    metric_mapping = {

        "average_precision":
            "validation_average_precision",

        "roc_auc":
            "validation_roc_auc",

        "precision_at_top10":
            "validation_precision_at_top10",

        "recall_at_top10":
            "validation_recall_at_top10",

        "lift_at_top10":
            "validation_lift_at_top10",
    }

    for (
        metric_name,
        seed_column
    ) in metric_mapping.items():

        values = (
            seed_metrics[
                seed_column
            ]
            .astype(float)
            .values
        )

        rows.append({

            "config_id":
                selected_config[
                    "config_id"
                ],

            "metric":
                metric_name,

            "single_seed_mean":
                float(
                    np.mean(
                        values
                    )
                ),

            "single_seed_std":
                float(
                    np.std(
                        values,
                        ddof=1
                    )
                ),

            "single_seed_median":
                float(
                    np.median(
                        values
                    )
                ),

            "single_seed_min":
                float(
                    np.min(
                        values
                    )
                ),

            "single_seed_max":
                float(
                    np.max(
                        values
                    )
                ),

            "ten_seed_ensemble_value":
                float(
                    ensemble_metrics[
                        metric_name
                    ]
                ),
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# 15. Stage 2 audit
# ============================================================

def run_stage_2_audit(
    protocol,
    current_protocol_hash,
    package,
    package_manifest,
    manifest_checked_files,
    manifest_mismatches,
    folds,
    configurations,
    config_results,
    selected_config,
    final_results
):
    """
    Strict Stage 2 audit.

    Any failed check stops approval.
    """

    audit_rows = []

    train_indices = package[
        "train_indices"
    ]

    validation_indices = package[
        "validation_indices"
    ]

    test_indices = package[
        "test_indices"
    ]

    metadata = package[
        "metadata"
    ]

    # --------------------------------------------------------
    # Protocol integrity
    # --------------------------------------------------------

    add_audit_check(

        audit_rows,

        "current_protocol_matches_stage1_package",

        current_protocol_hash
        ==
        package[
            "package_protocol_hash"
        ],

        (
            f"current={current_protocol_hash}; "
            f"package={package['package_protocol_hash']}"
        ),
    )

    add_audit_check(

        audit_rows,

        "package_manifest_protocol_hash_matches",

        package_manifest.get(
            "protocol_sha256"
        )
        ==
        current_protocol_hash,

        package_manifest.get(
            "protocol_sha256"
        ),
    )

    # --------------------------------------------------------
    # Package integrity
    # --------------------------------------------------------

    add_audit_check(

        audit_rows,

        "stage1_package_manifest_files_verified",

        len(
            manifest_mismatches
        )
        ==
        0,

        (
            f"checked_files={manifest_checked_files}; "
            f"mismatches={manifest_mismatches}"
        ),
    )

    # --------------------------------------------------------
    # Split separation
    # --------------------------------------------------------

    train_set = set(
        train_indices.tolist()
    )

    validation_set = set(
        validation_indices.tolist()
    )

    test_set = set(
        test_indices.tolist()
    )

    splits_are_disjoint = (

        train_set.isdisjoint(
            validation_set
        )

        and

        train_set.isdisjoint(
            test_set
        )

        and

        validation_set.isdisjoint(
            test_set
        )
    )

    add_audit_check(

        audit_rows,

        "train_validation_test_indices_are_disjoint",

        splits_are_disjoint,

        {
            "train":
                len(
                    train_set
                ),

            "validation":
                len(
                    validation_set
                ),

            "test":
                len(
                    test_set
                ),
        },
    )

    # --------------------------------------------------------
    # Official split labels
    # --------------------------------------------------------

    train_labels_correct = (

        metadata.iloc[
            train_indices
        ][
            "split"
        ]
        ==
        "train"
    ).all()

    validation_labels_correct = (

        metadata.iloc[
            validation_indices
        ][
            "split"
        ]
        ==
        "validation"
    ).all()

    test_labels_correct = (

        metadata.iloc[
            test_indices
        ][
            "split"
        ]
        ==
        "test"
    ).all()

    add_audit_check(

        audit_rows,

        "official_split_indices_match_metadata",

        (
            train_labels_correct
            and
            validation_labels_correct
            and
            test_labels_correct
        ),

        (
            f"train={train_labels_correct}, "
            f"validation={validation_labels_correct}, "
            f"test={test_labels_correct}"
        ),
    )

    # --------------------------------------------------------
    # Exactly eight configurations
    # --------------------------------------------------------

    add_audit_check(

        audit_rows,

        "exactly_eight_predefined_configurations",

        len(
            configurations
        )
        ==
        8,

        len(
            configurations
        ),
    )

    # --------------------------------------------------------
    # Rolling folds
    # --------------------------------------------------------

    rolling_folds_valid = True

    fold_details = []

    for fold in folds:

        fold_train_set = set(
            fold[
                "train_indices"
            ].tolist()
        )

        fold_validation_set = set(
            fold[
                "validation_indices"
            ].tolist()
        )

        subset_ok = (

            fold_train_set.issubset(
                train_set
            )

            and

            fold_validation_set.issubset(
                train_set
            )
        )

        no_test = (

            fold_train_set.isdisjoint(
                test_set
            )

            and

            fold_validation_set.isdisjoint(
                test_set
            )
        )

        chronological = (

            fold[
                "train_date_max"
            ]
            <
            fold[
                "validation_date_min"
            ]
        )

        if not (
            subset_ok
            and
            no_test
            and
            chronological
        ):

            rolling_folds_valid = False

        fold_details.append({

            "fold":
                fold[
                    "fold"
                ],

            "subset_ok":
                subset_ok,

            "no_test":
                no_test,

            "chronological":
                chronological,
        })

    add_audit_check(

        audit_rows,

        "all_rolling_origin_folds_are_train_only_and_chronological",

        rolling_folds_valid,

        fold_details,
    )

    # --------------------------------------------------------
    # Mechanical configuration selection
    # --------------------------------------------------------

    mechanically_ranked = (
        config_results
        .sort_values(

            by=[
                "mean_rolling_average_precision",
                "std_rolling_average_precision",
                "config_id",
            ],

            ascending=[
                False,
                True,
                True,
            ]
        )
        .reset_index(
            drop=True
        )
    )

    expected_selected_config = str(
        mechanically_ranked.iloc[
            0
        ][
            "config_id"
        ]
    )

    actual_selected_config = str(
        selected_config[
            "config_id"
        ]
    )

    add_audit_check(

        audit_rows,

        "selected_configuration_matches_locked_mechanical_rule",

        expected_selected_config
        ==
        actual_selected_config,

        (
            f"expected={expected_selected_config}; "
            f"actual={actual_selected_config}"
        ),
    )

    # --------------------------------------------------------
    # Train-only normalization
    # --------------------------------------------------------

    normalizer = final_results[
        "normalizer"
    ]

    expected_train_index_hash = (
        sha256_integer_array(
            train_indices
        )
    )

    add_audit_check(

        audit_rows,

        "final_normalization_fitted_on_official_train_indices_only",

        normalizer[
            "source_indices_sha256"
        ]
        ==
        expected_train_index_hash,

        (
            f"expected={expected_train_index_hash}; "
            f"actual="
            f"{normalizer['source_indices_sha256']}"
        ),
    )

    # --------------------------------------------------------
    # Locked ten seeds
    # --------------------------------------------------------

    protocol_seeds = [
        int(
            seed
        )
        for seed
        in protocol[
            "random_seeds"
        ]
    ]

    actual_seeds = final_results[
        "locked_seeds"
    ]

    add_audit_check(

        audit_rows,

        "all_ten_locked_random_seeds_used",

        (
            actual_seeds
            ==
            protocol_seeds
            and
            len(
                actual_seeds
            )
            ==
            10
        ),

        {
            "protocol_seeds":
                protocol_seeds,

            "actual_seeds":
                actual_seeds,
        },
    )

    # --------------------------------------------------------
    # Same selected configuration for every seed
    # --------------------------------------------------------

    seed_metrics = final_results[
        "seed_metrics"
    ]

    seed_config_ids = set(
        seed_metrics[
            "config_id"
        ].astype(
            str
        )
    )

    add_audit_check(

        audit_rows,

        "every_seed_uses_identical_selected_configuration",

        seed_config_ids
        ==
        {
            actual_selected_config
        },

        seed_config_ids,
    )

    # --------------------------------------------------------
    # Ensemble uses all ten seeds
    # --------------------------------------------------------

    ensemble_prediction_df = final_results[
        "ensemble_prediction_df"
    ]

    ensemble_seed_counts = set(
        ensemble_prediction_df[
            "ensemble_seed_count"
        ]
        .astype(
            int
        )
        .unique()
        .tolist()
    )

    add_audit_check(

        audit_rows,

        "ensemble_uses_all_ten_locked_seeds",

        ensemble_seed_counts
        ==
        {
            10
        },

        ensemble_seed_counts,
    )

    # --------------------------------------------------------
    # Validation prediction length
    # --------------------------------------------------------

    add_audit_check(

        audit_rows,

        "ensemble_prediction_count_matches_validation_split",

        len(
            ensemble_prediction_df
        )
        ==
        len(
            validation_indices
        ),

        (
            f"predictions={len(ensemble_prediction_df)}; "
            f"validation={len(validation_indices)}"
        ),
    )

    # --------------------------------------------------------
    # Prediction numerical safety
    # --------------------------------------------------------

    risk_values = final_results[
        "ensemble_risk"
    ]

    hazard_values = final_results[
        "ensemble_hazards"
    ]

    add_audit_check(

        audit_rows,

        "ensemble_predictions_are_finite",

        (
            np.isfinite(
                risk_values
            ).all()

            and

            np.isfinite(
                hazard_values
            ).all()
        ),

        (
            f"risk_finite="
            f"{np.isfinite(risk_values).all()}, "
            f"hazard_finite="
            f"{np.isfinite(hazard_values).all()}"
        ),
    )

    add_audit_check(

        audit_rows,

        "ensemble_probabilities_are_inside_zero_one_interval",

        (
            (
                risk_values
                >=
                0.0
            ).all()

            and

            (
                risk_values
                <=
                1.0
            ).all()

            and

            (
                hazard_values
                >=
                0.0
            ).all()

            and

            (
                hazard_values
                <=
                1.0
            ).all()
        ),

        "All ensemble hazards and cumulative risks must be in [0, 1].",
    )

    # --------------------------------------------------------
    # Positive labels available
    # --------------------------------------------------------

    validation_positive_count = int(
        package[
            "y_3_day"
        ][
            validation_indices
        ].sum()
    )

    add_audit_check(

        audit_rows,

        "validation_contains_positive_three_day_events",

        validation_positive_count
        >
        0,

        validation_positive_count,
    )

    # --------------------------------------------------------
    # No test prediction files
    # --------------------------------------------------------

    test_prediction_files = list(
        PREDICTION_DIR.glob(
            "*test*"
        )
    )

    add_audit_check(

        audit_rows,

        "no_test_prediction_file_was_created",

        len(
            test_prediction_files
        )
        ==
        0,

        [
            path.name
            for path
            in test_prediction_files
        ],
    )

    # --------------------------------------------------------
    # No test metric calculation
    # --------------------------------------------------------

    add_audit_check(

        audit_rows,

        "test_predictive_performance_not_calculated",

        True,

        (
            "Test indices were loaded only for split-integrity "
            "auditing. No test feature transformation, prediction, "
            "metric, early stopping, checkpoint selection, or "
            "hyperparameter selection was performed."
        ),
    )

    audit = pd.DataFrame(
        audit_rows
    )

    failed = audit[
        audit[
            "status"
        ]
        ==
        "FAIL"
    ].copy()

    return (
        audit,
        failed
    )


# ============================================================
# 16. Main
# ============================================================

def main():

    print(
        "=============================================="
    )

    print(
        "STAGE 2: TRAIN SELF-GRU HAZARD BASELINE"
    )

    print(
        "=============================================="
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else
        "cpu"
    )

    print(
        "\nDevice:",
        device
    )

    if (
        device.type
        ==
        "cpu"
    ):

        available_threads = (
            os.cpu_count()
            or
            1
        )

        torch.set_num_threads(int(os.environ.get("PORTFOLIO_TORCH_THREADS", "1")))

    # --------------------------------------------------------
    # Locked protocol
    # --------------------------------------------------------

    (
        protocol,
        protocol_hash,
    ) = load_locked_protocol()

    print(
        "\n========== LOCKED PROTOCOL =========="
    )

    print(
        "Project:",
        protocol[
            "project_name"
        ]
    )

    print(
        "Primary task:",
        protocol[
            "primary_task"
        ]
    )

    print(
        "Primary metric:",
        protocol[
            "primary_metric"
        ]
    )

    print(
        "Random seeds:",
        protocol[
            "random_seeds"
        ]
    )

    print(
        "Protocol SHA256:",
        protocol_hash
    )

    # --------------------------------------------------------
    # Verify Stage 1 package manifest
    # --------------------------------------------------------

    (
        package_manifest,
        manifest_checked_files,
        manifest_mismatches,
    ) = verify_package_manifest()

    print(
        "\n========== PACKAGE MANIFEST =========="
    )

    print(
        "Manifest files checked:",
        manifest_checked_files
    )

    print(
        "Checksum mismatches:",
        len(
            manifest_mismatches
        )
    )

    if manifest_mismatches:

        print(
            manifest_mismatches
        )

    # --------------------------------------------------------
    # Load package
    # --------------------------------------------------------

    package = load_training_package()

    X_self = package[
        "X_self"
    ]

    y_hazard = package[
        "y_hazard"
    ]

    hazard_mask = package[
        "hazard_mask"
    ]

    y_3_day = package[
        "y_3_day"
    ]

    metadata = package[
        "metadata"
    ]

    train_indices = package[
        "train_indices"
    ]

    validation_indices = package[
        "validation_indices"
    ]

    test_indices = package[
        "test_indices"
    ]

    print(
        "\n========== LOCKED TRAINING PACKAGE =========="
    )

    print(
        "X_self shape:",
        X_self.shape
    )

    print(
        "y_hazard shape:",
        y_hazard.shape
    )

    print(
        "hazard mask shape:",
        hazard_mask.shape
    )

    print(
        "Train samples:",
        len(
            train_indices
        )
    )

    print(
        "Validation samples:",
        len(
            validation_indices
        )
    )

    print(
        "Test samples locked:",
        len(
            test_indices
        )
    )

    print(
        "Train 3-day positives:",
        int(
            y_3_day[
                train_indices
            ].sum()
        )
    )

    print(
        "Validation 3-day positives:",
        int(
            y_3_day[
                validation_indices
            ].sum()
        )
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "No test feature tensor or test label is selected "
        "for model fitting, validation, prediction, or metrics."
    )

    # --------------------------------------------------------
    # Build three train-only rolling folds
    # --------------------------------------------------------

    folds = build_rolling_origin_folds(

        metadata=
            metadata,

        train_indices=
            train_indices,

        y_3_day=
            y_3_day,

        n_folds=
            N_ROLLING_FOLDS,
    )

    fold_definition_rows = []

    print(
        "\n========== ROLLING-ORIGIN FOLDS =========="
    )

    for fold in folds:

        print(
            f"Fold {fold['fold']}: "
            f"train "
            f"{fold['train_date_min'].date()} "
            f"to "
            f"{fold['train_date_max'].date()} "
            f"({fold['train_sample_count']} samples, "
            f"{fold['train_positive_count']} positives)"
        )

        print(
            f"         validation "
            f"{fold['validation_date_min'].date()} "
            f"to "
            f"{fold['validation_date_max'].date()} "
            f"({fold['validation_sample_count']} samples, "
            f"{fold['validation_positive_count']} positives)"
        )

        fold_definition_rows.append({

            "fold":
                fold[
                    "fold"
                ],

            "train_date_min":
                fold[
                    "train_date_min"
                ],

            "train_date_max":
                fold[
                    "train_date_max"
                ],

            "validation_date_min":
                fold[
                    "validation_date_min"
                ],

            "validation_date_max":
                fold[
                    "validation_date_max"
                ],

            "train_sample_count":
                fold[
                    "train_sample_count"
                ],

            "validation_sample_count":
                fold[
                    "validation_sample_count"
                ],

            "train_positive_count":
                fold[
                    "train_positive_count"
                ],

            "validation_positive_count":
                fold[
                    "validation_positive_count"
                ],
        })

    pd.DataFrame(
        fold_definition_rows
    ).to_csv(

        OUT_DIR
        / "rolling_origin_folds.csv",

        index=False,

        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Hyperparameter grid
    # --------------------------------------------------------

    configurations = (
        build_hyperparameter_grid()
    )

    print(
        "\n========== PREDEFINED CONFIGURATIONS =========="
    )

    for config in configurations:

        print(
            config
        )

    # --------------------------------------------------------
    # Rolling-origin model selection
    # --------------------------------------------------------

    (
        fold_results,
        config_results,
        selected_config,
    ) = run_rolling_origin_selection(

        package=
            package,

        folds=
            folds,

        configurations=
            configurations,

        device=
            device,
    )

    fold_results.to_csv(

        OUT_DIR
        / "rolling_origin_fold_results.csv",

        index=False,

        encoding="utf-8-sig"
    )

    config_results.to_csv(

        OUT_DIR
        / "rolling_origin_config_results.csv",

        index=False,

        encoding="utf-8-sig"
    )

    save_json(

        selected_config,

        OUT_DIR
        / "selected_self_hazard_config.json"
    )

    print(
        "\n=============================================="
    )

    print(
        "SELECTED SELF-HAZARD CONFIGURATION"
    )

    print(
        "=============================================="
    )

    print(
        selected_config
    )

    # --------------------------------------------------------
    # Train selected model across all ten locked seeds
    # --------------------------------------------------------

    final_results = run_final_seed_training(

        package=
            package,

        selected_config=
            selected_config,

        protocol=
            protocol,

        device=
            device,
    )

    seed_metrics = final_results[
        "seed_metrics"
    ]

    training_history = final_results[
        "training_history"
    ]

    checkpoint_manifest = final_results[
        "checkpoint_manifest"
    ]

    ensemble_metrics = final_results[
        "ensemble_metrics"
    ]

    validation_summary = (
        build_validation_summary(

            seed_metrics=
                seed_metrics,

            ensemble_metrics=
                ensemble_metrics,

            selected_config=
                selected_config,
        )
    )

    seed_metrics.to_csv(

        OUT_DIR
        / "self_hazard_seed_metrics_validation.csv",

        index=False,

        encoding="utf-8-sig"
    )

    training_history.to_csv(

        OUT_DIR
        / "training_history_all_seeds.csv",

        index=False,

        encoding="utf-8-sig"
    )

    checkpoint_manifest.to_csv(

        OUT_DIR
        / "checkpoint_manifest.csv",

        index=False,

        encoding="utf-8-sig"
    )

    validation_summary.to_csv(

        OUT_DIR
        / "self_hazard_validation_summary.csv",

        index=False,

        encoding="utf-8-sig"
    )

    pd.DataFrame([
        {

            "config_id":
                selected_config[
                    "config_id"
                ],

            "ensemble_seed_count":
                len(
                    final_results[
                        "locked_seeds"
                    ]
                ),

            "validation_average_precision":
                ensemble_metrics[
                    "average_precision"
                ],

            "validation_roc_auc":
                ensemble_metrics[
                    "roc_auc"
                ],

            "validation_precision_at_top10":
                ensemble_metrics[
                    "precision_at_top10"
                ],

            "validation_recall_at_top10":
                ensemble_metrics[
                    "recall_at_top10"
                ],

            "validation_lift_at_top10":
                ensemble_metrics[
                    "lift_at_top10"
                ],
        }
    ]).to_csv(

        OUT_DIR
        / "self_hazard_ensemble_validation_metrics.csv",

        index=False,

        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Save train-only normalization
    # --------------------------------------------------------

    signal_order = package[
        "feature_definition"
    ][
        "signal_order"
    ]

    normalizer = final_results[
        "normalizer"
    ]

    normalization_record = {

        "transformation":
            "log1p followed by signal-wise z-score",

        "normalization_source":
            "official training split only",

        "source_sample_count":
            normalizer[
                "source_sample_count"
            ],

        "source_indices_sha256":
            normalizer[
                "source_indices_sha256"
            ],

        "signal_order":
            signal_order,

        "mean_on_log1p_scale":
            normalizer[
                "mean"
            ].tolist(),

        "std_on_log1p_scale":
            normalizer[
                "std"
            ].tolist(),

        "validation_used_for_normalization":
            False,

        "test_used_for_normalization":
            False,
    }

    save_json(

        normalization_record,

        OUT_DIR
        / "train_only_normalization.json"
    )

    # --------------------------------------------------------
    # Stage 2 audit
    # --------------------------------------------------------

    (
        audit,
        failed_audit,
    ) = run_stage_2_audit(

        protocol=
            protocol,

        current_protocol_hash=
            protocol_hash,

        package=
            package,

        package_manifest=
            package_manifest,

        manifest_checked_files=
            manifest_checked_files,

        manifest_mismatches=
            manifest_mismatches,

        folds=
            folds,

        configurations=
            configurations,

        config_results=
            config_results,

        selected_config=
            selected_config,

        final_results=
            final_results,
    )

    audit.to_csv(

        OUT_DIR
        / "stage_2_audit_report.csv",

        index=False,

        encoding="utf-8-sig"
    )

    print(
        "\n=============================================="
    )

    print(
        "STAGE 2 STRICT AUDIT"
    )

    print(
        "=============================================="
    )

    print(
        audit.to_string(
            index=False
        )
    )

    pass_count = int(
        (
            audit[
                "status"
            ]
            ==
            "PASS"
        ).sum()
    )

    fail_count = int(
        (
            audit[
                "status"
            ]
            ==
            "FAIL"
        ).sum()
    )

    print(
        "\nAudit summary:"
    )

    print(
        "PASS:",
        pass_count
    )

    print(
        "FAIL:",
        fail_count
    )

    if (
        len(
            failed_audit
        )
        >
        0
    ):

        print(
            "\nFailed checks:"
        )

        print(
            failed_audit.to_string(
                index=False
            )
        )

        raise RuntimeError(
            "Stage 2 was not approved because "
            "strict audit checks failed."
        )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print(
        "\n=============================================="
    )

    print(
        "STAGE 2 APPROVED: SELF-HAZARD BASELINE TRAINED"
    )

    print(
        "=============================================="
    )

    print(
        "\nSelected configuration:"
    )

    print(
        selected_config
    )

    print(
        "\nTen-seed single-model validation summary:"
    )

    print(
        validation_summary.to_string(
            index=False
        )
    )

    print(
        "\nTen-seed ensemble validation metrics:"
    )

    for (
        metric_name,
        metric_value
    ) in ensemble_metrics.items():

        print(
            f"{metric_name}: "
            f"{metric_value:.6f}"
        )

    print(
        "\nResults saved to:"
    )

    print(
        OUT_DIR
    )

    print(
        "\nImportant files:"
    )

    print(
        " - rolling_origin_folds.csv"
    )

    print(
        " - rolling_origin_fold_results.csv"
    )

    print(
        " - rolling_origin_config_results.csv"
    )

    print(
        " - selected_self_hazard_config.json"
    )

    print(
        " - self_hazard_seed_metrics_validation.csv"
    )

    print(
        " - self_hazard_ensemble_validation_metrics.csv"
    )

    print(
        " - self_hazard_validation_summary.csv"
    )

    print(
        " - train_only_normalization.json"
    )

    print(
        " - training_history_all_seeds.csv"
    )

    print(
        " - checkpoint_manifest.csv"
    )

    print(
        " - stage_2_audit_report.csv"
    )

    print(
        " - predictions/validation_ensemble_predictions.csv"
    )

    print(
        " - checkpoints/self_hazard_seed_*.pt"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "No locked-test prediction or locked-test performance "
        "was calculated."
    )


if __name__ == "__main__":

    main()