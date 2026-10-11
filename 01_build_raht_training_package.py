# Project supervisor: Dr. Zohreh Kaheh.
# Undergraduate project collaborator: Yuyang Zhang.
# Initial student coding was developed with supervisory teaching and improvement guidance.
# -*- coding: utf-8 -*-

from pathlib import Path
import hashlib
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# 0. Paths and fixed configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

PROTOCOL_FILE = (
    BASE_DIR
    / "00_raht_protocol_locked.json"
)

EVENT_DIR = (
    BASE_DIR
    / "supplygraph_disruption_events"
)

ANOMALY_FILE = (
    EVENT_DIR
    / "node_day_anomaly_scores.csv"
)

OUT_DIR = (
    BASE_DIR
    / "raht_training_package"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


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


TEMPORAL_LABELS = {
    "u_delivery":
        "Unit Delivery to Distributor",

    "u_factory":
        "Unit Factory Issue Flow",

    "u_production":
        "Unit Production",

    "u_sales":
        "Unit Sales Order",

    "w_delivery":
        "Weight Delivery to Distributor",

    "w_factory":
        "Weight Factory Issue Flow",

    "w_production":
        "Weight Production",

    "w_sales":
        "Weight Sales Order",
}


RELATION_CONFIG = {

    "product_group": {
        "filename":
            "Edges (Product Group).csv",

        "group_column":
            "GroupCode",

        "role":
            "primary",
    },

    "storage_location": {
        "filename":
            "Edges (Storage Location).csv",

        "group_column":
            "Storage Location",

        "role":
            "primary",
    },

    "plant": {
        "filename":
            "Edges (Plant).csv",

        "group_column":
            "Plant",

        "role":
            "ablation",
    },

    "product_subgroup": {
        "filename":
            "Edges (Product Sub-Group).csv",

        "group_column":
            "SubGroupCode",

        "role":
            "ablation",
    },
}


# ============================================================
# 1. General utilities
# ============================================================

def sha256_file(path):
    """
    Calculate SHA256 checksum of a file.
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


def save_json(
    obj,
    path
):
    """
    Save JSON using UTF-8.
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


def compact_name(value):
    """
    Simplify filename for robust matching.
    """

    return (
        str(value)
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace(".", "")
    )


def find_supplygraph_directory():
    """
    Find the SupplyGraph Homogenoeus/Homogeneous directory.
    """

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

    raise FileNotFoundError(
        "Cannot find SupplyGraph raw dataset directory.\n"
        "Tried:\n"
        +
        "\n".join(
            str(path)
            for path in candidates
        )
    )


# ============================================================
# 2. Locked protocol verification
# ============================================================

def load_locked_protocol():
    """
    Load and validate the locked protocol.
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

    required_keys = [
        "project_name",
        "dataset",
        "primary_task",
        "disruption_definition",
        "temporal_lookback_days",
        "primary_relations",
        "primary_metric",
        "test_policy",
    ]

    missing = [
        key
        for key in required_keys
        if key not in protocol
    ]

    if missing:

        raise KeyError(
            "Locked protocol is missing required keys:\n"
            f"{missing}"
        )

    if int(
        protocol[
            "temporal_lookback_days"
        ]
    ) != 14:

        raise RuntimeError(
            "Protocol mismatch: temporal lookback "
            "must remain locked at 14 days."
        )

    expected_primary_relations = {
        "product_group",
        "storage_location",
    }

    actual_primary_relations = set(
        protocol[
            "primary_relations"
        ]
    )

    if (
        actual_primary_relations
        !=
        expected_primary_relations
    ):

        raise RuntimeError(
            "Protocol mismatch.\n"
            "Primary relations must remain locked as:\n"
            "product_group + storage_location"
        )

    protocol_hash = sha256_file(
        PROTOCOL_FILE
    )

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
        "Lookback days:",
        protocol[
            "temporal_lookback_days"
        ]
    )

    print(
        "Primary relations:",
        protocol[
            "primary_relations"
        ]
    )

    print(
        "Primary metric:",
        protocol[
            "primary_metric"
        ]
    )

    print(
        "Protocol SHA256:",
        protocol_hash
    )

    return (
        protocol,
        protocol_hash,
    )


# ============================================================
# 3. SupplyGraph temporal tables
# ============================================================

def find_temporal_file(
    sg_dir,
    folder_name,
    keyword
):
    """
    Find one temporal CSV robustly.
    """

    folder = (
        sg_dir
        / "Temporal Data"
        / folder_name
    )

    if not folder.exists():

        raise FileNotFoundError(
            f"Cannot find temporal folder:\n"
            f"{folder}"
        )

    files = list(
        folder.glob(
            "*.csv"
        )
    )

    for path in files:

        if (
            keyword
            in compact_name(
                path.name
            )
        ):

            return path

    raise FileNotFoundError(
        f"Cannot find temporal file "
        f"keyword='{keyword}' "
        f"in:\n{folder}"
    )


def read_wide_csv(path):
    """
    Read one SupplyGraph temporal CSV.
    """

    df = pd.read_csv(
        path
    )

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    first_column = (
        df.columns[0]
    )

    df = df.rename(
        columns={
            first_column:
                "Date"
        }
    )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df = (
        df
        .dropna(
            subset=[
                "Date"
            ]
        )
        .sort_values(
            "Date"
        )
        .drop_duplicates(
            subset=[
                "Date"
            ],
            keep="first"
        )
        .reset_index(
            drop=True
        )
    )

    return df


def load_temporal_tables(
    sg_dir
):
    """
    Read the eight raw SupplyGraph temporal tables.
    """

    paths = {

        "u_delivery":
            find_temporal_file(
                sg_dir,
                "Unit",
                "delivery"
            ),

        "u_factory":
            find_temporal_file(
                sg_dir,
                "Unit",
                "factory"
            ),

        "u_production":
            find_temporal_file(
                sg_dir,
                "Unit",
                "production"
            ),

        "u_sales":
            find_temporal_file(
                sg_dir,
                "Unit",
                "sales"
            ),

        "w_delivery":
            find_temporal_file(
                sg_dir,
                "Weight",
                "delivery"
            ),

        "w_factory":
            find_temporal_file(
                sg_dir,
                "Weight",
                "factory"
            ),

        "w_production":
            find_temporal_file(
                sg_dir,
                "Weight",
                "production"
            ),

        "w_sales":
            find_temporal_file(
                sg_dir,
                "Weight",
                "sales"
            ),
    }

    tables = {}

    print(
        "\n========== RAW TEMPORAL TABLES =========="
    )

    for key in TEMPORAL_KEYS:

        path = paths[
            key
        ]

        tables[
            key
        ] = read_wide_csv(
            path
        )

        print(
            f"{key}: "
            f"{path.name}, "
            f"shape="
            f"{tables[key].shape}"
        )

    return (
        tables,
        paths,
    )


# ============================================================
# 4. Anomaly/onset dataset
# ============================================================

def load_anomaly_dataset():
    """
    Read the previously locked disruption-onset proxy data.
    """

    if not ANOMALY_FILE.exists():

        raise FileNotFoundError(
            f"Cannot find:\n"
            f"{ANOMALY_FILE}"
        )

    data = pd.read_csv(
        ANOMALY_FILE
    )

    required_columns = [
        "date",
        "node",
        "split",
        "is_anomaly_state",
        "is_disruption_onset",
        "eligible_for_early_warning",
        "next_day_onset_label",
        "next_3_day_onset_label",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:

        raise KeyError(
            "Anomaly dataset is missing columns:\n"
            f"{missing_columns}"
        )

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce"
    )

    data["node"] = (
        data[
            "node"
        ]
        .astype(str)
        .str.strip()
    )

    data = (
        data
        .dropna(
            subset=[
                "date",
                "node",
            ]
        )
        .sort_values(
            [
                "date",
                "node",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    duplicate_keys = data.duplicated(
        subset=[
            "date",
            "node",
        ]
    ).sum()

    if duplicate_keys > 0:

        raise RuntimeError(
            "Duplicate node-date rows found in "
            "anomaly dataset:\n"
            f"{duplicate_keys}"
        )

    nodes = list(
        data[
            "node"
        ]
        .drop_duplicates()
    )

    onset_count = int(
        data[
            "is_disruption_onset"
        ].sum()
    )

    print(
        "\n========== DISRUPTION ONSET DATA =========="
    )

    print(
        "Node-days:",
        len(
            data
        )
    )

    print(
        "Nodes:",
        len(
            nodes
        )
    )

    print(
        "Qualified disruption onsets:",
        onset_count
    )

    print(
        "Date range:",
        data[
            "date"
        ].min(),
        "to",
        data[
            "date"
        ].max()
    )

    return (
        data,
        nodes,
        onset_count,
    )


# ============================================================
# 5. Align all eight temporal tables
# ============================================================

def align_temporal_tables(
    tables,
    nodes
):
    """
    Align all temporal tables by:
    - identical dates,
    - identical ordered 41-node columns.

    Pandas-renamed nodes such as:
        POP001L12P.1
    are deliberately preserved.
    """

    common_dates = None

    for key in TEMPORAL_KEYS:

        current_dates = set(
            tables[
                key
            ][
                "Date"
            ]
        )

        if common_dates is None:

            common_dates = (
                current_dates
            )

        else:

            common_dates = (
                common_dates
                .intersection(
                    current_dates
                )
            )

    common_dates = sorted(
        common_dates
    )

    if len(
        common_dates
    ) == 0:

        raise RuntimeError(
            "No common dates were found "
            "across eight temporal tables."
        )

    aligned = {}

    for key in TEMPORAL_KEYS:

        df = tables[
            key
        ].copy()

        duplicated_columns = (
            df.columns[
                df.columns.duplicated()
            ].tolist()
        )

        if duplicated_columns:

            raise RuntimeError(
                f"Duplicated columns remain "
                f"in {key}:\n"
                f"{duplicated_columns}"
            )

        missing_nodes = [
            node
            for node in nodes
            if node not in df.columns
        ]

        if missing_nodes:

            raise RuntimeError(
                f"Temporal table {key} "
                f"is missing expected nodes:\n"
                f"{missing_nodes}"
            )

        selected = (
            df[
                df[
                    "Date"
                ].isin(
                    common_dates
                )
            ]
            .sort_values(
                "Date"
            )
            .loc[
                :,
                [
                    "Date"
                ]
                +
                nodes
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        for node in nodes:

            selected[
                node
            ] = pd.to_numeric(
                selected[
                    node
                ],
                errors="coerce"
            )

            selected[
                node
            ] = (
                selected[
                    node
                ]
                .replace(
                    [
                        np.inf,
                        -np.inf,
                    ],
                    np.nan
                )
                .fillna(
                    0.0
                )
                .astype(
                    float
                )
            )

        aligned[
            key
        ] = selected

        print(
            f"Aligned {key}: "
            f"{selected.shape}"
        )

    reference_dates = aligned[
        TEMPORAL_KEYS[0]
    ][
        "Date"
    ].tolist()

    for key in TEMPORAL_KEYS[1:]:

        candidate_dates = aligned[
            key
        ][
            "Date"
        ].tolist()

        if (
            candidate_dates
            !=
            reference_dates
        ):

            raise RuntimeError(
                f"Date alignment failed for "
                f"{key}."
            )

    print(
        "\nCommon raw dates:",
        len(
            reference_dates
        )
    )

    print(
        "Raw date range:",
        reference_dates[0],
        "to",
        reference_dates[-1]
    )

    return aligned


# ============================================================
# 6. Build raw temporal matrices
# ============================================================

def build_temporal_matrices(
    tables,
    nodes
):
    """
    Convert eight temporal tables to T × N matrices.
    """

    matrices = {}

    for key in TEMPORAL_KEYS:

        matrices[
            key
        ] = (
            tables[
                key
            ][
                nodes
            ]
            .astype(
                float
            )
            .values
            .astype(
                np.float32
            )
        )

    dates = pd.DatetimeIndex(
        tables[
            TEMPORAL_KEYS[0]
        ][
            "Date"
        ]
    )

    return (
        matrices,
        dates,
    )


# ============================================================
# 7. Build 14-day self temporal tensors
# ============================================================

def build_raht_samples(
    anomaly,
    nodes,
    matrices,
    raw_dates,
    lookback_days
):
    """
    Construct leakage-safe RAHT samples.

    Prediction origin:
        current normal node-day t

    Input:
        previous 14 daily observations ending at t

    Target:
        first qualified disruption onset at:
            t+1
            t+2
            t+3

    Strict split-boundary policy:
        All three future target dates must remain inside the
        SAME chronological split as prediction origin t.

    Therefore:
        - Train labels never use Validation outcomes.
        - Validation labels never use Test outcomes.
        - Test remains completely locked.
    """

    node_to_index = {
        node: index
        for index, node in enumerate(nodes)
    }

    date_to_index = {
        pd.Timestamp(date): index
        for index, date in enumerate(raw_dates)
    }

    # --------------------------------------------------------
    # Verify that every calendar date belongs to exactly
    # one chronological split.
    # --------------------------------------------------------

    split_counts_by_date = (
        anomaly
        .groupby("date")["split"]
        .nunique()
    )

    if (
        split_counts_by_date.max()
        > 1
    ):
        raise RuntimeError(
            "At least one calendar date belongs to more than "
            "one chronological split."
        )

    date_to_split = (
        anomaly[
            [
                "date",
                "split",
            ]
        ]
        .drop_duplicates(
            subset=["date"]
        )
        .set_index("date")["split"]
        .to_dict()
    )

    # --------------------------------------------------------
    # Build qualified-onset matrix.
    # --------------------------------------------------------

    onset_pivot = (
        anomaly
        .pivot(
            index="date",
            columns="node",
            values="is_disruption_onset"
        )
        .reindex(
            index=raw_dates,
            columns=nodes
        )
        .fillna(0)
        .astype(int)
    )

    onset_matrix = (
        onset_pivot
        .values
        .astype(np.int64)
    )

    # --------------------------------------------------------
    # Candidate prediction origins.
    # --------------------------------------------------------

    usable = anomaly[
        (
            anomaly[
                "eligible_for_early_warning"
            ]
            == 1
        )
        &
        (
            anomaly[
                "next_3_day_onset_label"
            ].notna()
        )
    ].copy()

    usable = (
        usable
        .sort_values(
            [
                "date",
                "node",
            ]
        )
        .reset_index(drop=True)
    )

    X_list = []
    hazard_targets = []
    hazard_masks = []

    y_next_day = []
    y_3_day = []

    metadata_rows = []

    next_day_label_mismatches = 0
    next_3_day_label_mismatches = 0

    skipped_insufficient_history = 0
    skipped_missing_raw_date = 0
    skipped_insufficient_future = 0

    # New critical counter.
    skipped_cross_split_target_horizon = 0

    sample_counter = 0

    # --------------------------------------------------------
    # Construct each sample.
    # --------------------------------------------------------

    for _, row in usable.iterrows():

        date = pd.Timestamp(
            row["date"]
        )

        node = str(
            row["node"]
        ).strip()

        origin_split = str(
            row["split"]
        )

        if date not in date_to_index:

            skipped_missing_raw_date += 1
            continue

        raw_time_index = (
            date_to_index[
                date
            ]
        )

        node_index = (
            node_to_index[
                node
            ]
        )

        sequence_start = (
            raw_time_index
            -
            lookback_days
            +
            1
        )

        sequence_end = (
            raw_time_index
        )

        if sequence_start < 0:

            skipped_insufficient_history += 1
            continue

        # Need all three future raw observations.
        if (
            raw_time_index
            +
            3
            >=
            len(raw_dates)
        ):

            skipped_insufficient_future += 1
            continue

        # ----------------------------------------------------
        # Critical split-boundary leakage protection.
        # ----------------------------------------------------

        target_day_1_date = pd.Timestamp(
            raw_dates[
                raw_time_index + 1
            ]
        )

        target_day_2_date = pd.Timestamp(
            raw_dates[
                raw_time_index + 2
            ]
        )

        target_day_3_date = pd.Timestamp(
            raw_dates[
                raw_time_index + 3
            ]
        )

        future_target_dates = [
            target_day_1_date,
            target_day_2_date,
            target_day_3_date,
        ]

        future_target_splits = [
            date_to_split.get(
                future_date,
                None
            )
            for future_date
            in future_target_dates
        ]

        # Every future target day must:
        # 1. have a known split;
        # 2. remain in exactly the same split as origin t.
        target_horizon_is_split_safe = all(
            future_split
            ==
            origin_split
            for future_split
            in future_target_splits
        )

        if not target_horizon_is_split_safe:

            skipped_cross_split_target_horizon += 1
            continue

        # ----------------------------------------------------
        # Build 14 × 8 self-history tensor.
        # ----------------------------------------------------

        sequence_channels = []

        for key in TEMPORAL_KEYS:

            signal_sequence = (
                matrices[
                    key
                ][
                    sequence_start
                    :
                    sequence_end + 1,
                    node_index
                ]
            )

            sequence_channels.append(
                signal_sequence
            )

        X_sequence = np.stack(
            sequence_channels,
            axis=1
        )

        expected_shape = (
            lookback_days,
            len(TEMPORAL_KEYS),
        )

        if (
            X_sequence.shape
            !=
            expected_shape
        ):

            raise RuntimeError(
                "Unexpected temporal tensor "
                f"shape for {node}, {date}:\n"
                f"{X_sequence.shape}, "
                f"expected {expected_shape}"
            )

        # ----------------------------------------------------
        # Derive Day+1 / Day+2 / Day+3 first-onset targets.
        # ----------------------------------------------------

        future_onsets = onset_matrix[
            raw_time_index + 1
            :
            raw_time_index + 4,
            node_index
        ]

        positive_positions = np.where(
            future_onsets
            ==
            1
        )[0]

        if len(
            positive_positions
        ) > 0:

            event_day = int(
                positive_positions[0]
                +
                1
            )

        else:

            event_day = 0

        # ----------------------------------------------------
        # Discrete-time hazard target.
        # ----------------------------------------------------

        hazard_target = np.zeros(
            3,
            dtype=np.float32
        )

        hazard_mask = np.ones(
            3,
            dtype=np.float32
        )

        if event_day > 0:

            hazard_target[
                event_day - 1
            ] = 1.0

            # Once the first onset occurs, later periods
            # are no longer part of the risk set.
            if event_day < 3:

                hazard_mask[
                    event_day:
                ] = 0.0

        derived_next_day_label = int(
            event_day
            ==
            1
        )

        derived_next_3_day_label = int(
            event_day
            >
            0
        )

        saved_next_day_label = int(
            row[
                "next_day_onset_label"
            ]
        )

        saved_next_3_day_label = int(
            row[
                "next_3_day_onset_label"
            ]
        )

        if (
            derived_next_day_label
            !=
            saved_next_day_label
        ):

            next_day_label_mismatches += 1

        if (
            derived_next_3_day_label
            !=
            saved_next_3_day_label
        ):

            next_3_day_label_mismatches += 1

        # ----------------------------------------------------
        # Save approved sample.
        # ----------------------------------------------------

        sample_counter += 1

        sample_id = (
            f"SAMPLE_"
            f"{sample_counter:06d}"
        )

        sequence_start_date = pd.Timestamp(
            raw_dates[
                sequence_start
            ]
        )

        sequence_end_date = pd.Timestamp(
            raw_dates[
                sequence_end
            ]
        )

        X_list.append(
            X_sequence
        )

        hazard_targets.append(
            hazard_target
        )

        hazard_masks.append(
            hazard_mask
        )

        y_next_day.append(
            derived_next_day_label
        )

        y_3_day.append(
            derived_next_3_day_label
        )

        metadata_rows.append({

            "sample_id":
                sample_id,

            "date":
                date,

            "node":
                node,

            "node_index":
                node_index,

            "raw_time_index":
                raw_time_index,

            "split":
                origin_split,

            "sequence_start_date":
                sequence_start_date,

            "sequence_end_date":
                sequence_end_date,

            "prediction_origin_date":
                date,

            "target_day_1_date":
                target_day_1_date,

            "target_day_2_date":
                target_day_2_date,

            "target_day_3_date":
                target_day_3_date,

            "target_horizon_end_date":
                target_day_3_date,

            "event_day_1_to_3":
                event_day,

            "next_day_onset_label":
                derived_next_day_label,

            "next_3_day_onset_label":
                derived_next_3_day_label,

            "prediction_origin_is_normal":
                int(
                    row[
                        "is_anomaly_state"
                    ]
                    ==
                    0
                ),

            "eligible_for_early_warning":
                int(
                    row[
                        "eligible_for_early_warning"
                    ]
                ),
        })

    # --------------------------------------------------------
    # Convert to arrays.
    # --------------------------------------------------------

    X_self = np.asarray(
        X_list,
        dtype=np.float32
    )

    y_hazard = np.asarray(
        hazard_targets,
        dtype=np.float32
    )

    hazard_at_risk_mask = np.asarray(
        hazard_masks,
        dtype=np.float32
    )

    y_next_day = np.asarray(
        y_next_day,
        dtype=np.int64
    )

    y_3_day = np.asarray(
        y_3_day,
        dtype=np.int64
    )

    metadata = pd.DataFrame(
        metadata_rows
    )

    build_info = {

        "usable_candidate_rows":
            int(
                len(usable)
            ),

        "final_sample_count":
            int(
                len(metadata)
            ),

        "skipped_insufficient_history":
            int(
                skipped_insufficient_history
            ),

        "skipped_missing_raw_date":
            int(
                skipped_missing_raw_date
            ),

        "skipped_insufficient_future":
            int(
                skipped_insufficient_future
            ),

        "skipped_cross_split_target_horizon":
            int(
                skipped_cross_split_target_horizon
            ),

        "next_day_label_mismatches":
            int(
                next_day_label_mismatches
            ),

        "next_3_day_label_mismatches":
            int(
                next_3_day_label_mismatches
            ),
    }

    print(
        "\n========== RAHT SAMPLE TENSORS =========="
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
        "Hazard mask shape:",
        hazard_at_risk_mask.shape
    )

    print(
        "Next-day positives:",
        int(
            y_next_day.sum()
        )
    )

    print(
        "Three-day positives:",
        int(
            y_3_day.sum()
        )
    )

    print(
        "Skipped cross-split target horizons:",
        skipped_cross_split_target_horizon
    )

    print(
        "Next-day label mismatches:",
        next_day_label_mismatches
    )

    print(
        "Three-day label mismatches:",
        next_3_day_label_mismatches
    )

    return (
        X_self,
        y_hazard,
        hazard_at_risk_mask,
        y_next_day,
        y_3_day,
        metadata,
        build_info,
    )

# ============================================================
# 8. Robust node-index mapping
# ============================================================

def load_index_to_temporal_node(
    sg_dir,
    nodes
):
    """
    Resolve NodesIndex.csv into the exact 41 unique
    temporal node names.

    This deliberately handles duplicated raw names such as:

        POP001L12P
        POP001L12P.1
    """

    path = (
        sg_dir
        / "Nodes"
        / "NodesIndex.csv"
    )

    df = pd.read_csv(
        path
    )

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    if (
        "Node" not in df.columns
        or
        "NodeIndex" not in df.columns
    ):

        raise KeyError(
            "NodesIndex.csv must contain "
            "'Node' and 'NodeIndex'."
        )

    valid_nodes = set(
        nodes
    )

    occurrence_counts = {}

    mapping = {}

    mapping_rows = []

    for _, row in df.iterrows():

        base_name = str(
            row[
                "Node"
            ]
        ).strip()

        occurrence = (
            occurrence_counts.get(
                base_name,
                0
            )
        )

        if occurrence == 0:

            candidate_name = (
                base_name
            )

        else:

            candidate_name = (
                f"{base_name}."
                f"{occurrence}"
            )

        occurrence_counts[
            base_name
        ] = (
            occurrence
            +
            1
        )

        try:

            node_index = int(
                float(
                    row[
                        "NodeIndex"
                    ]
                )
            )

        except Exception:

            continue

        if (
            candidate_name
            in
            valid_nodes
        ):

            mapping[
                node_index
            ] = (
                candidate_name
            )

            mapping_rows.append({

                "source_node_index":
                    node_index,

                "raw_node_name":
                    base_name,

                "resolved_temporal_node":
                    candidate_name,
            })

    mapping_df = pd.DataFrame(
        mapping_rows
    )

    mapping_df.to_csv(
        OUT_DIR
        / "node_index_resolution.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return mapping


# ============================================================
# 9. Hypergraph incidence matrices
# ============================================================

def detect_best_numeric_offset(
    edge_df,
    valid_nodes,
    index_mapping
):
    """
    Detect whether numeric edge endpoints are:
        exact NodeIndex,
        one below NodeIndex,
        or one above NodeIndex.

    The best-matching offset is selected mechanically
    from the data, not from predictive performance.
    """

    candidate_offsets = [
        0,
        -1,
        1,
    ]

    scores = {}

    for offset in candidate_offsets:

        match_count = 0

        for endpoint_column in [
            "node1",
            "node2",
        ]:

            for value in edge_df[
                endpoint_column
            ]:

                text = str(
                    value
                ).strip()

                if text in valid_nodes:

                    match_count += 1
                    continue

                try:

                    numeric = int(
                        float(
                            text
                        )
                    )

                    mapped_index = (
                        numeric
                        +
                        offset
                    )

                    if (
                        mapped_index
                        in
                        index_mapping
                    ):

                        match_count += 1

                except Exception:

                    continue

        scores[
            offset
        ] = match_count

    best_offset = max(
        candidate_offsets,
        key=lambda offset: (
            scores[
                offset
            ],
            int(
                offset
                ==
                0
            ),
        )
    )

    return (
        best_offset,
        scores,
    )


def resolve_edge_node(
    value,
    valid_nodes,
    index_mapping,
    numeric_offset
):
    """
    Resolve one edge endpoint.
    """

    text = str(
        value
    ).strip()

    if text in valid_nodes:

        return text

    try:

        numeric = int(
            float(
                text
            )
        )

        mapped_index = (
            numeric
            +
            numeric_offset
        )

        candidate = index_mapping.get(
            mapped_index
        )

        if (
            candidate
            in
            valid_nodes
        ):

            return candidate

    except Exception:

        pass

    return None


def build_relation_incidence(
    sg_dir,
    nodes,
    relation_name,
    config,
    index_mapping
):
    """
    Construct one true hypergraph incidence matrix H.

    Shape:
        [number_of_nodes, number_of_hyperedges]

    No clique expansion is used.
    """

    path = (
        sg_dir
        / "Edges"
        / config[
            "filename"
        ]
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Cannot find relation file:\n"
            f"{path}"
        )

    df = pd.read_csv(
        path
    )

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    required_columns = [
        "node1",
        "node2",
        config[
            "group_column"
        ],
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise KeyError(
            f"{path.name} is missing:\n"
            f"{missing}"
        )

    valid_nodes = set(
        nodes
    )

    (
        numeric_offset,
        offset_scores,
    ) = detect_best_numeric_offset(
        df,
        valid_nodes,
        index_mapping
    )

    node_to_index = {
        node:
            index

        for index, node
        in enumerate(
            nodes
        )
    }

    hyperedge_records = []

    seen_member_sets = set()

    group_column = config[
        "group_column"
    ]

    for (
        group_id,
        group_df
    ) in df.groupby(
        group_column,
        dropna=False
    ):

        members = set()

        for _, row in group_df.iterrows():

            node_1 = resolve_edge_node(
                row[
                    "node1"
                ],
                valid_nodes,
                index_mapping,
                numeric_offset
            )

            node_2 = resolve_edge_node(
                row[
                    "node2"
                ],
                valid_nodes,
                index_mapping,
                numeric_offset
            )

            if node_1 is not None:

                members.add(
                    node_1
                )

            if node_2 is not None:

                members.add(
                    node_2
                )

        if len(
            members
        ) < 2:

            continue

        ordered_members = tuple(
            sorted(
                members,
                key=lambda node:
                    node_to_index[
                        node
                    ]
            )
        )

        member_key = frozenset(
            ordered_members
        )

        if member_key in seen_member_sets:

            continue

        seen_member_sets.add(
            member_key
        )

        hyperedge_records.append({

            "group_id":
                str(
                    group_id
                ),

            "members":
                ordered_members,
        })

    number_of_nodes = len(
        nodes
    )

    number_of_hyperedges = len(
        hyperedge_records
    )

    H = np.zeros(
        (
            number_of_nodes,
            number_of_hyperedges
        ),
        dtype=np.float32
    )

    metadata_rows = []

    covered_nodes = set()

    for (
        hyperedge_index,
        record
    ) in enumerate(
        hyperedge_records
    ):

        members = record[
            "members"
        ]

        for node in members:

            node_index = node_to_index[
                node
            ]

            H[
                node_index,
                hyperedge_index
            ] = 1.0

            covered_nodes.add(
                node
            )

        metadata_rows.append({

            "relation":
                relation_name,

            "relation_role":
                config[
                    "role"
                ],

            "hyperedge_index":
                hyperedge_index,

            "source_group_id":
                record[
                    "group_id"
                ],

            "hyperedge_size":
                len(
                    members
                ),

            "members":
                " | ".join(
                    members
                ),

            "numeric_index_offset_used":
                numeric_offset,
        })

    metadata = pd.DataFrame(
        metadata_rows
    )

    sizes = [
        len(
            record[
                "members"
            ]
        )
        for record in hyperedge_records
    ]

    summary = {

        "relation":
            relation_name,

        "relation_role":
            config[
                "role"
            ],

        "source_file":
            config[
                "filename"
            ],

        "number_of_hyperedges":
            int(
                number_of_hyperedges
            ),

        "covered_nodes":
            int(
                len(
                    covered_nodes
                )
            ),

        "total_nodes":
            int(
                number_of_nodes
            ),

        "coverage_rate":
            float(
                len(
                    covered_nodes
                )
                /
                max(
                    number_of_nodes,
                    1
                )
            ),

        "minimum_hyperedge_size":
            int(
                min(
                    sizes
                )
            )
            if sizes
            else 0,

        "median_hyperedge_size":
            float(
                np.median(
                    sizes
                )
            )
            if sizes
            else 0.0,

        "maximum_hyperedge_size":
            int(
                max(
                    sizes
                )
            )
            if sizes
            else 0,

        "numeric_index_offset_used":
            int(
                numeric_offset
            ),

        "offset_match_scores":
            str(
                offset_scores
            ),
    }

    return (
        H,
        metadata,
        summary,
    )


def build_all_hypergraphs(
    sg_dir,
    nodes
):
    """
    Build the four locked relation-specific hypergraphs.
    """

    index_mapping = (
        load_index_to_temporal_node(
            sg_dir,
            nodes
        )
    )

    matrices = {}

    metadata_tables = {}

    summary_rows = []

    print(
        "\n========== TRUE HYPERGRAPH INCIDENCE =========="
    )

    for (
        relation_name,
        config
    ) in RELATION_CONFIG.items():

        (
            H,
            metadata,
            summary,
        ) = build_relation_incidence(

            sg_dir=sg_dir,

            nodes=nodes,

            relation_name=
                relation_name,

            config=config,

            index_mapping=
                index_mapping,
        )

        matrices[
            relation_name
        ] = H

        metadata_tables[
            relation_name
        ] = metadata

        summary_rows.append(
            summary
        )

        print(
            relation_name,
            "role=",
            config[
                "role"
            ],
            "shape=",
            H.shape,
            "covered_nodes=",
            summary[
                "covered_nodes"
            ],
            "median_size=",
            summary[
                "median_hyperedge_size"
            ],
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    return (
        matrices,
        metadata_tables,
        summary_df,
    )


# ============================================================
# 10. Train / validation / test split indices
# ============================================================

def build_split_indices(
    metadata
):
    """
    Preserve original chronological split labels.
    """

    split_indices = {}

    for split_name in [
        "train",
        "validation",
        "test",
    ]:

        indices = np.where(
            metadata[
                "split"
            ].values
            ==
            split_name
        )[0]

        split_indices[
            split_name
        ] = indices.astype(
            np.int64
        )

    return split_indices


# ============================================================
# 11. Strict audit
# ============================================================

def add_audit_check(
    audit_rows,
    check_name,
    passed,
    details
):
    """
    Add one audit result.
    """

    audit_rows.append({

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


def run_data_audit(
    protocol,
    protocol_hash,
    anomaly,
    nodes,
    onset_count,
    X_self,
    y_hazard,
    hazard_mask,
    y_next_day,
    y_3_day,
    metadata,
    build_info,
    hypergraph_matrices,
    hypergraph_summary,
    split_indices
):
    """
    Run strict leakage and consistency checks.

    Any failed audit check stops the pipeline.
    """

    audit_rows = []

    # --------------------------------------------------------
    # Protocol
    # --------------------------------------------------------

    add_audit_check(

        audit_rows,

        "locked_protocol_hash_exists",

        len(
            protocol_hash
        )
        ==
        64,

        protocol_hash,
    )

    add_audit_check(

        audit_rows,

        "lookback_locked_at_14_days",

        int(
            protocol[
                "temporal_lookback_days"
            ]
        )
        ==
        14,

        protocol[
            "temporal_lookback_days"
        ],
    )

    # --------------------------------------------------------
    # Original anomaly dataset
    # --------------------------------------------------------

    add_audit_check(

        audit_rows,

        "expected_41_product_nodes",

        len(
            nodes
        )
        ==
        41,

        f"observed={len(nodes)}",
    )

    add_audit_check(

        audit_rows,

        "qualified_onsets_available",

        onset_count
        >
        0,

        f"observed={onset_count}",
    )

    add_audit_check(

        audit_rows,

        "no_duplicate_node_date_in_anomaly_data",

        anomaly.duplicated(
            subset=[
                "date",
                "node",
            ]
        ).sum()
        ==
        0,

        "Duplicate node-date count="
        f"{anomaly.duplicated(subset=['date', 'node']).sum()}",
    )

    # --------------------------------------------------------
    # Tensor shape
    # --------------------------------------------------------

    expected_shape = (

        len(
            metadata
        ),

        14,

        8,
    )

    add_audit_check(

        audit_rows,

        "X_self_shape_is_samples_x_14_x_8",

        X_self.shape
        ==
        expected_shape,

        f"observed={X_self.shape}, "
        f"expected={expected_shape}",
    )

    add_audit_check(

        audit_rows,

        "hazard_target_shape_is_samples_x_3",

        y_hazard.shape
        ==
        (
            len(
                metadata
            ),
            3,
        ),

        y_hazard.shape,
    )

    add_audit_check(

        audit_rows,

        "hazard_mask_shape_is_samples_x_3",

        hazard_mask.shape
        ==
        (
            len(
                metadata
            ),
            3,
        ),

        hazard_mask.shape,
    )

    # --------------------------------------------------------
    # Numerical safety
    # --------------------------------------------------------

    add_audit_check(

        audit_rows,

        "X_self_has_no_nan",

        not np.isnan(
            X_self
        ).any(),

        f"NaN count="
        f"{np.isnan(X_self).sum()}",
    )

    add_audit_check(

        audit_rows,

        "X_self_has_no_infinity",

        np.isfinite(
            X_self
        ).all(),

        "All values finite="
        f"{np.isfinite(X_self).all()}",
    )

    # --------------------------------------------------------
    # Leakage checks
    # --------------------------------------------------------

    sequence_end_matches_origin = (

        pd.to_datetime(
            metadata[
                "sequence_end_date"
            ]
        )
        ==
        pd.to_datetime(
            metadata[
                "prediction_origin_date"
            ]
        )
    ).all()

    add_audit_check(

        audit_rows,

        "sequence_ends_exactly_at_prediction_origin",

        sequence_end_matches_origin,

        "No self-sequence observation "
        "comes after the prediction origin.",
    )

    sequence_length_days = (

        pd.to_datetime(
            metadata[
                "sequence_end_date"
            ]
        )
        -
        pd.to_datetime(
            metadata[
                "sequence_start_date"
            ]
        )
    ).dt.days

    add_audit_check(

        audit_rows,

        "all_sequences_span_14_daily_observations",

        (
            sequence_length_days
            ==
            13
        ).all(),

        "A 14-observation daily window "
        "has a 13-day calendar span.",
    )

    add_audit_check(

        audit_rows,

        "all_prediction_origins_are_normal",

        (
            metadata[
                "prediction_origin_is_normal"
            ]
            ==
            1
        ).all(),

        "Every forecasting sample must "
        "start from a currently normal state.",
    )

    add_audit_check(

        audit_rows,

        "all_samples_are_eligible_for_early_warning",

        (
            metadata[
                "eligible_for_early_warning"
            ]
            ==
            1
        ).all(),

        "Eligibility must equal 1 "
        "for every packaged sample.",
    )

    # --------------------------------------------------------
    # Target-horizon split-boundary leakage check
    # --------------------------------------------------------

    date_split_table = (
        anomaly[
            [
                "date",
                "split",
            ]
        ]
        .drop_duplicates(
            subset=["date"]
        )
        .copy()
    )

    audit_date_to_split = (
        date_split_table
        .set_index("date")["split"]
        .to_dict()
    )

    target_date_columns = [
        "target_day_1_date",
        "target_day_2_date",
        "target_day_3_date",
    ]

    all_target_dates_stay_in_origin_split = True

    boundary_violation_count = 0

    for target_date_column in target_date_columns:

        target_dates = pd.to_datetime(
            metadata[
                target_date_column
            ]
        )

        target_splits = target_dates.map(
            audit_date_to_split
        )

        violations = (
            target_splits.values
            !=
            metadata[
                "split"
            ].values
        )

        boundary_violation_count += int(
            violations.sum()
        )

        if violations.any():

            all_target_dates_stay_in_origin_split = False

    add_audit_check(

        audit_rows,

        "all_three_day_target_windows_remain_inside_origin_split",

        all_target_dates_stay_in_origin_split,

        (
            "Cross-split target-date violations="
            f"{boundary_violation_count}; "
            "Train labels must not use validation outcomes, "
            "and validation labels must not use test outcomes."
        ),
    )

    add_audit_check(

        audit_rows,

        "cross_split_target_horizons_were_removed_before_packaging",

        True,

        (
            "Removed candidate samples="
            f"{build_info.get('skipped_cross_split_target_horizon', 0)}"
        ),
    )

    # --------------------------------------------------------
    # Label consistency
    # --------------------------------------------------------

    add_audit_check(

        audit_rows,

        "next_day_labels_match_previous_locked_labels",

        build_info[
            "next_day_label_mismatches"
        ]
        ==
        0,

        build_info[
            "next_day_label_mismatches"
        ],
    )

    add_audit_check(

        audit_rows,

        "three_day_labels_match_previous_locked_labels",

        build_info[
            "next_3_day_label_mismatches"
        ]
        ==
        0,

        build_info[
            "next_3_day_label_mismatches"
        ],
    )

    hazard_implied_3day = (

        y_hazard.sum(
            axis=1
        )
        >
        0
    ).astype(
        np.int64
    )

    add_audit_check(

        audit_rows,

        "hazard_targets_match_three_day_binary_labels",

        np.array_equal(
            hazard_implied_3day,
            y_3_day
        ),

        "Hazard positive within three days "
        "must exactly equal 3-day label.",
    )

    hazard_implied_nextday = (

        y_hazard[
            :,
            0
        ]
        >
        0
    ).astype(
        np.int64
    )

    add_audit_check(

        audit_rows,

        "hazard_day1_matches_next_day_binary_labels",

        np.array_equal(
            hazard_implied_nextday,
            y_next_day
        ),

        "Day+1 hazard event must exactly "
        "equal next-day onset label.",
    )

    add_audit_check(

        audit_rows,

        "no_sample_contains_multiple_hazard_events",

        (
            y_hazard.sum(
                axis=1
            )
            <=
            1
        ).all(),

        "Discrete first-onset target permits "
        "at most one event in the horizon.",
    )

    # --------------------------------------------------------
    # Split chronology
    # --------------------------------------------------------

    split_nonempty = all(

        len(
            split_indices[
                split_name
            ]
        )
        >
        0

        for split_name in [
            "train",
            "validation",
            "test",
        ]
    )

    add_audit_check(

        audit_rows,

        "all_three_splits_are_nonempty",

        split_nonempty,

        {
            key:
                len(
                    value
                )
            for key, value
            in split_indices.items()
        },
    )

    train_dates = pd.to_datetime(

        metadata.loc[
            split_indices[
                "train"
            ],
            "date"
        ]
    )

    validation_dates = pd.to_datetime(

        metadata.loc[
            split_indices[
                "validation"
            ],
            "date"
        ]
    )

    test_dates = pd.to_datetime(

        metadata.loc[
            split_indices[
                "test"
            ],
            "date"
        ]
    )

    strict_chronology = (

        train_dates.max()
        <
        validation_dates.min()

        and

        validation_dates.max()
        <
        test_dates.min()
    )

    add_audit_check(

        audit_rows,

        "strict_chronological_split_order",

        strict_chronology,

        (
            f"train max={train_dates.max()}, "
            f"validation min={validation_dates.min()}, "
            f"validation max={validation_dates.max()}, "
            f"test min={test_dates.min()}"
        ),
    )

    # --------------------------------------------------------
    # Hypergraph checks
    # --------------------------------------------------------

    for relation_name in [
        "product_group",
        "storage_location",
    ]:

        H = hypergraph_matrices[
            relation_name
        ]

        add_audit_check(

            audit_rows,

            f"{relation_name}_primary_hypergraph_nonempty",

            (
                H.shape[0]
                ==
                41
                and
                H.shape[1]
                >
                0
            ),

            f"shape={H.shape}",
        )

        add_audit_check(

            audit_rows,

            f"{relation_name}_incidence_is_binary",

            np.isin(
                H,
                [
                    0.0,
                    1.0,
                ]
            ).all(),

            "Incidence values must be 0 or 1.",
        )

    # --------------------------------------------------------
    # Test policy
    # --------------------------------------------------------

    add_audit_check(

        audit_rows,

        "test_set_not_used_for_model_selection",

        True,

        (
            "This script constructs package files only. "
            "It calculates no predictive test metric "
            "and performs no model or checkpoint selection."
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
    ]

    print(
        "\n========== STRICT DATA AUDIT =========="
    )

    print(
        audit.to_string(
            index=False
        )
    )

    print(
        "\nAudit summary:"
    )

    print(
        "PASS:",
        int(
            (
                audit[
                    "status"
                ]
                ==
                "PASS"
            ).sum()
        )
    )

    print(
        "FAIL:",
        len(
            failed
        )
    )

    return (
        audit,
        failed
    )


# ============================================================
# 12. Save package
# ============================================================

def save_training_package(
    protocol,
    protocol_hash,
    temporal_paths,
    X_self,
    y_hazard,
    hazard_mask,
    y_next_day,
    y_3_day,
    metadata,
    split_indices,
    hypergraph_matrices,
    hypergraph_metadata,
    hypergraph_summary,
    audit
):
    """
    Save the complete locked RAHT training package.
    """

    # --------------------------------------------------------
    # Core tensors
    # --------------------------------------------------------

    np.save(
        OUT_DIR
        / "X_self.npy",
        X_self
    )

    np.save(
        OUT_DIR
        / "y_hazard.npy",
        y_hazard
    )

    np.save(
        OUT_DIR
        / "hazard_at_risk_mask.npy",
        hazard_mask
    )

    np.save(
        OUT_DIR
        / "y_next_day.npy",
        y_next_day
    )

    np.save(
        OUT_DIR
        / "y_3_day.npy",
        y_3_day
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata.to_csv(
        OUT_DIR
        / "sample_metadata.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Splits
    # --------------------------------------------------------

    np.savez(

        OUT_DIR
        / "split_indices.npz",

        train_indices=
            split_indices[
                "train"
            ],

        validation_indices=
            split_indices[
                "validation"
            ],

        test_indices=
            split_indices[
                "test"
            ],
    )

    # --------------------------------------------------------
    # Hypergraphs
    # --------------------------------------------------------

    for (
        relation_name,
        H
    ) in hypergraph_matrices.items():

        filename = (
            f"H_"
            f"{relation_name}"
            f".npy"
        )

        np.save(
            OUT_DIR
            / filename,
            H
        )

    for (
        relation_name,
        metadata_table
    ) in hypergraph_metadata.items():

        metadata_table.to_csv(

            OUT_DIR
            /
            (
                "hyperedges_"
                f"{relation_name}"
                ".csv"
            ),

            index=False,

            encoding="utf-8-sig"
        )

    hypergraph_summary.to_csv(
        OUT_DIR
        / "hypergraph_structure_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Feature definitions
    # --------------------------------------------------------

    feature_definition = {

        "tensor_name":
            "X_self",

        "shape_semantics": [
            "sample",
            "lookback_day",
            "operational_signal",
        ],

        "lookback_days":
            14,

        "prediction_origin_information":
            (
                "14 daily observations ending "
                "at current prediction-origin day t"
            ),

        "future_target_horizon_days":
            3,

        "signal_order":
            TEMPORAL_KEYS,

        "signal_descriptions":
            TEMPORAL_LABELS,

        "target_files": {

            "y_hazard.npy":
                (
                    "Discrete Day+1/Day+2/Day+3 "
                    "first-onset indicator."
                ),

            "hazard_at_risk_mask.npy":
                (
                    "At-risk periods used by "
                    "discrete-time hazard loss."
                ),

            "y_next_day.npy":
                (
                    "Binary onset within exactly Day+1."
                ),

            "y_3_day.npy":
                (
                    "Binary onset anywhere within "
                    "Day+1 to Day+3."
                ),
        },
    }

    save_json(
        feature_definition,
        OUT_DIR
        / "feature_names.json"
    )

    # --------------------------------------------------------
    # Frozen protocol snapshot
    # --------------------------------------------------------

    save_json(
        protocol,
        OUT_DIR
        / "protocol_snapshot.json"
    )

    with open(
        OUT_DIR
        / "protocol_sha256.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            protocol_hash
        )

    # --------------------------------------------------------
    # Audit
    # --------------------------------------------------------

    audit.to_csv(
        OUT_DIR
        / "data_audit_report.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Package manifest
    # --------------------------------------------------------

    source_files = {

        "locked_protocol":
            str(
                PROTOCOL_FILE
            ),

        "anomaly_onset_file":
            str(
                ANOMALY_FILE
            ),
    }

    for key, path in temporal_paths.items():

        source_files[
            f"temporal_{key}"
        ] = str(
            path
        )

    package_files = []

    for path in sorted(
        OUT_DIR.iterdir()
    ):

        if (
            path.is_file()
            and
            path.name
            !=
            "package_manifest.json"
        ):

            package_files.append({

                "filename":
                    path.name,

                "size_bytes":
                    int(
                        path.stat().st_size
                    ),

                "sha256":
                    sha256_file(
                        path
                    ),
            })

    manifest = {

        "project":
            protocol[
                "project_name"
            ],

        "protocol_sha256":
            protocol_hash,

        "source_files":
            source_files,

        "core_shapes": {

            "X_self":
                list(
                    X_self.shape
                ),

            "y_hazard":
                list(
                    y_hazard.shape
                ),

            "hazard_at_risk_mask":
                list(
                    hazard_mask.shape
                ),

            "H_product_group":
                list(
                    hypergraph_matrices[
                        "product_group"
                    ].shape
                ),

            "H_storage_location":
                list(
                    hypergraph_matrices[
                        "storage_location"
                    ].shape
                ),
        },

        "sample_counts": {

            "total":
                int(
                    len(
                        metadata
                    )
                ),

            "train":
                int(
                    len(
                        split_indices[
                            "train"
                        ]
                    )
                ),

            "validation":
                int(
                    len(
                        split_indices[
                            "validation"
                        ]
                    )
                ),

            "test":
                int(
                    len(
                        split_indices[
                            "test"
                        ]
                    )
                ),

            "three_day_positive_total":
                int(
                    y_3_day.sum()
                ),

            "next_day_positive_total":
                int(
                    y_next_day.sum()
                ),
        },

        "test_policy":
            protocol[
                "test_policy"
            ],

        "package_files":
            package_files,
    }

    save_json(
        manifest,
        OUT_DIR
        / "package_manifest.json"
    )


# ============================================================
# 13. Main
# ============================================================

def main():

    print(
        "=============================================="
    )

    print(
        "STAGE 1: BUILD LOCKED RAHT TRAINING PACKAGE"
    )

    print(
        "=============================================="
    )

    # --------------------------------------------------------
    # Locked protocol
    # --------------------------------------------------------

    (
        protocol,
        protocol_hash,
    ) = load_locked_protocol()

    # --------------------------------------------------------
    # SupplyGraph path
    # --------------------------------------------------------

    sg_dir = (
        find_supplygraph_directory()
    )

    print(
        "\nSupplyGraph folder:"
    )

    print(
        sg_dir
    )

    # --------------------------------------------------------
    # Previous disruption onset dataset
    # --------------------------------------------------------

    (
        anomaly,
        nodes,
        onset_count,
    ) = load_anomaly_dataset()

    # --------------------------------------------------------
    # Raw temporal tables
    # --------------------------------------------------------

    (
        temporal_tables,
        temporal_paths,
    ) = load_temporal_tables(
        sg_dir
    )

    temporal_tables = (
        align_temporal_tables(
            temporal_tables,
            nodes
        )
    )

    (
        temporal_matrices,
        raw_dates,
    ) = build_temporal_matrices(
        temporal_tables,
        nodes
    )

    # --------------------------------------------------------
    # Build leakage-safe samples and hazard targets
    # --------------------------------------------------------

    lookback_days = int(
        protocol[
            "temporal_lookback_days"
        ]
    )

    (
        X_self,
        y_hazard,
        hazard_mask,
        y_next_day,
        y_3_day,
        metadata,
        build_info,
    ) = build_raht_samples(

        anomaly=anomaly,

        nodes=nodes,

        matrices=
            temporal_matrices,

        raw_dates=
            raw_dates,

        lookback_days=
            lookback_days,
    )

    # --------------------------------------------------------
    # True relation-specific hypergraphs
    # --------------------------------------------------------

    (
        hypergraph_matrices,
        hypergraph_metadata,
        hypergraph_summary,
    ) = build_all_hypergraphs(
        sg_dir,
        nodes
    )

    # --------------------------------------------------------
    # Chronological split indices
    # --------------------------------------------------------

    split_indices = (
        build_split_indices(
            metadata
        )
    )

    print(
        "\n========== SAMPLE SPLITS =========="
    )

    for split_name in [
        "train",
        "validation",
        "test",
    ]:

        print(
            split_name,
            ":",
            len(
                split_indices[
                    split_name
                ]
            )
        )

    # --------------------------------------------------------
    # Strict audit
    # --------------------------------------------------------

    (
        audit,
        failed_audit,
    ) = run_data_audit(

        protocol=
            protocol,

        protocol_hash=
            protocol_hash,

        anomaly=
            anomaly,

        nodes=
            nodes,

        onset_count=
            onset_count,

        X_self=
            X_self,

        y_hazard=
            y_hazard,

        hazard_mask=
            hazard_mask,

        y_next_day=
            y_next_day,

        y_3_day=
            y_3_day,

        metadata=
            metadata,

        build_info=
            build_info,

        hypergraph_matrices=
            hypergraph_matrices,

        hypergraph_summary=
            hypergraph_summary,

        split_indices=
            split_indices,
    )

    # Save audit immediately even if failure occurs.
    audit.to_csv(
        OUT_DIR
        / "data_audit_report.csv",
        index=False,
        encoding="utf-8-sig"
    )

    if len(
        failed_audit
    ) > 0:

        print(
            "\n=============================================="
        )

        print(
            "PACKAGE BUILD STOPPED"
        )

        print(
            "=============================================="
        )

        print(
            "\nOne or more strict audit checks failed."
        )

        print(
            "\nFailed checks:"
        )

        print(
            failed_audit.to_string(
                index=False
            )
        )

        raise RuntimeError(
            "RAHT training package was not approved "
            "because strict data-audit checks failed."
        )

    # --------------------------------------------------------
    # Save approved package
    # --------------------------------------------------------

    save_training_package(

        protocol=
            protocol,

        protocol_hash=
            protocol_hash,

        temporal_paths=
            temporal_paths,

        X_self=
            X_self,

        y_hazard=
            y_hazard,

        hazard_mask=
            hazard_mask,

        y_next_day=
            y_next_day,

        y_3_day=
            y_3_day,

        metadata=
            metadata,

        split_indices=
            split_indices,

        hypergraph_matrices=
            hypergraph_matrices,

        hypergraph_metadata=
            hypergraph_metadata,

        hypergraph_summary=
            hypergraph_summary,

        audit=
            audit,
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print(
        "\n=============================================="
    )

    print(
        "STAGE 1 APPROVED: TRAINING PACKAGE BUILT"
    )

    print(
        "=============================================="
    )

    print(
        "\nAll strict audit checks passed."
    )

    print(
        "\nResults saved to:"
    )

    print(
        OUT_DIR
    )

    print(
        "\nCore package files:"
    )

    print(
        " - X_self.npy"
    )

    print(
        " - y_hazard.npy"
    )

    print(
        " - hazard_at_risk_mask.npy"
    )

    print(
        " - y_next_day.npy"
    )

    print(
        " - y_3_day.npy"
    )

    print(
        " - sample_metadata.csv"
    )

    print(
        " - split_indices.npz"
    )

    print(
        " - H_product_group.npy"
    )

    print(
        " - H_storage_location.npy"
    )

    print(
        " - H_plant.npy"
    )

    print(
        " - H_product_subgroup.npy"
    )

    print(
        " - hypergraph_structure_summary.csv"
    )

    print(
        " - data_audit_report.csv"
    )

    print(
        " - package_manifest.json"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "No test predictive performance "
        "was calculated or inspected."
    )


if __name__ == "__main__":

    main()