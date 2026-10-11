# Project supervisor: Dr. Zohreh Kaheh.
# Undergraduate project collaborator: Yuyang Zhang.
# Initial student coding was developed with supervisory teaching and improvement guidance.
# -*- coding: utf-8 -*-
"""Technical implementation for the supply-chain code portfolio."""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings("ignore")


# ============================================================
# 0. Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

CANDIDATE_SG_DIRS = [
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

SG_DIR = None

for candidate in CANDIDATE_SG_DIRS:
    if candidate.exists():
        SG_DIR = candidate
        break

if SG_DIR is None:
    raise FileNotFoundError(
        "Cannot find SupplyGraph folder. Tried:\n"
        + "\n".join(str(p) for p in CANDIDATE_SG_DIRS)
    )


OUT_DIR = BASE_DIR / "supplygraph_disruption_events"
FIG_DIR = OUT_DIR / "figures"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Experimental settings
# ----------------------------

RANDOM_STATE = 42

# Longest history used to predict today's normal operation.
MAX_LAG = 14

# Chronological split.
TRAIN_FRACTION = 0.60
VALIDATION_FRACTION = 0.20

# Default official anomaly threshold.
# It is computed using training anomaly scores only.
ANOMALY_QUANTILE = 0.975

# A qualified disruption onset must be preceded by
# at least this many consecutive normal days.
MIN_NORMAL_DAYS_BEFORE_ONSET = 3

# First-pass model removes the highest residual contamination
# before fitting the final normal-operation baseline.
NORMAL_BASELINE_CLEAN_QUANTILE = 0.95

EPS = 1e-8


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
# 1. File loading
# ============================================================

def compact_name(value):
    """
    Simplify a filename for robust matching.
    """
    return (
        str(value)
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace(".", "")
    )


def find_temporal_file(folder_name, keyword):
    """
    Find a temporal CSV file by folder and keyword.
    """
    folder = SG_DIR / "Temporal Data" / folder_name

    if not folder.exists():
        raise FileNotFoundError(f"Cannot find temporal folder: {folder}")

    files = list(folder.glob("*.csv"))

    for path in files:
        if keyword in compact_name(path.name):
            return path

    raise FileNotFoundError(
        f"Cannot find keyword='{keyword}' in folder:\n{folder}"
    )


def read_wide_csv(path):
    """
    Read a SupplyGraph wide temporal CSV.
    """
    df = pd.read_csv(path)

    df.columns = [str(c).strip() for c in df.columns]

    first_col = df.columns[0]

    df = df.rename(columns={first_col: "Date"})

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df = df.dropna(subset=["Date"]).copy()

    return df


def load_temporal_tables():
    """
    Read all eight temporal tables.
    """

    paths = {
        "u_delivery": find_temporal_file("Unit", "delivery"),
        "u_factory": find_temporal_file("Unit", "factory"),
        "u_production": find_temporal_file("Unit", "production"),
        "u_sales": find_temporal_file("Unit", "sales"),

        "w_delivery": find_temporal_file("Weight", "delivery"),
        "w_factory": find_temporal_file("Weight", "factory"),
        "w_production": find_temporal_file("Weight", "production"),
        "w_sales": find_temporal_file("Weight", "sales"),
    }

    tables = {}

    print("\n========== Temporal Files ==========")

    for key, path in paths.items():

        print(f"{key}: {path.name}")

        tables[key] = read_wide_csv(path)

    return tables


def load_nodes(tables):
    """
    Use the temporal-table column order as the canonical node order.

    This is more robust than directly using Nodes.csv because the raw
    SupplyGraph files may contain duplicated product names. Pandas
    automatically renames duplicated CSV headers, for example:

        POP001L12P
        POP001L12P.1

    We preserve these unique temporal-table names so that all 41 product
    columns remain available.
    """

    # Use Unit Delivery as the reference ordering.
    reference_key = "u_delivery"

    if reference_key not in tables:
        raise KeyError(
            f"Reference temporal table '{reference_key}' was not loaded."
        )

    reference_nodes = [
        str(c).strip()
        for c in tables[reference_key].columns
        if c != "Date"
    ]

    # Keep only nodes present in all eight temporal tables.
    common_nodes = set(reference_nodes)

    for key, df in tables.items():

        current_nodes = set(
            str(c).strip()
            for c in df.columns
            if c != "Date"
        )

        common_nodes = common_nodes.intersection(current_nodes)

    # Preserve the original order of the reference table.
    nodes = [
        node
        for node in reference_nodes
        if node in common_nodes
    ]

    # Safety check: node names used by the script must be unique.
    duplicate_nodes = pd.Series(nodes)[
        pd.Series(nodes).duplicated()
    ].tolist()

    if duplicate_nodes:
        raise RuntimeError(
            "Duplicate node names remain after temporal-table alignment: "
            f"{duplicate_nodes}"
        )

    if len(nodes) < 30:
        raise RuntimeError(
            f"Only {len(nodes)} common nodes were found. "
            "Expected approximately 41."
        )

    print("\n========== Nodes ==========")
    print("Number of nodes:", len(nodes))
    print("First five:", nodes[:5])

    suspicious_nodes = [
        node
        for node in nodes
        if ".1" in str(node)
    ]

    if suspicious_nodes:
        print(
            "Pandas-renamed duplicated source columns detected:",
            suspicious_nodes
        )

    return nodes


def align_temporal_tables(tables, nodes):
    """
    Align all eight temporal tables using:

    1. common dates;
    2. the exact same unique 41-node column order.

    The conversion is deliberately performed one column at a time to avoid
    Pandas assignment errors caused by duplicated or automatically renamed
    source headers.
    """

    # --------------------------------------------------------
    # Find dates present in every temporal table.
    # --------------------------------------------------------

    common_dates = None

    for key, df in tables.items():

        current_dates = set(
            pd.to_datetime(
                df["Date"],
                errors="coerce"
            ).dropna()
        )

        if common_dates is None:
            common_dates = current_dates
        else:
            common_dates = common_dates.intersection(
                current_dates
            )

    common_dates = sorted(common_dates)

    if not common_dates:
        raise RuntimeError(
            "No common dates were found across the eight temporal tables."
        )

    # --------------------------------------------------------
    # Align every table.
    # --------------------------------------------------------

    aligned = {}

    for key, df in tables.items():

        temp = df.copy()

        temp["Date"] = pd.to_datetime(
            temp["Date"],
            errors="coerce"
        )

        temp = (
            temp[
                temp["Date"].isin(common_dates)
            ]
            .sort_values("Date")
            .drop_duplicates(
                subset=["Date"],
                keep="first"
            )
            .reset_index(drop=True)
        )

        # Check that there are no duplicate column names remaining.
        duplicated_columns = (
            temp.columns[
                temp.columns.duplicated()
            ]
            .tolist()
        )

        if duplicated_columns:
            raise RuntimeError(
                f"Duplicated columns remain in temporal table '{key}': "
                f"{duplicated_columns}"
            )

        # Check required nodes.
        missing_nodes = [
            node
            for node in nodes
            if node not in temp.columns
        ]

        if missing_nodes:
            raise RuntimeError(
                f"Temporal table '{key}' is missing "
                f"{len(missing_nodes)} expected nodes:\n"
                f"{missing_nodes}"
            )

        # Explicitly select the exact desired columns.
        selected = temp.loc[
            :,
            ["Date"] + nodes
        ].copy()

        # Convert one node column at a time.
        for node in nodes:

            selected[node] = pd.to_numeric(
                selected[node],
                errors="coerce"
            )

            selected[node] = (
                selected[node]
                .replace(
                    [np.inf, -np.inf],
                    np.nan
                )
                .fillna(0.0)
                .astype(float)
            )

        aligned[key] = selected

        print(
            f"Aligned {key}: "
            f"shape={selected.shape}, "
            f"nodes={len(nodes)}"
        )

    print("\n========== Temporal Alignment ==========")
    print("Common dates:", len(common_dates))
    print(
        "Date range:",
        common_dates[0],
        "to",
        common_dates[-1]
    )

    # Final shape check.
    expected_columns = 1 + len(nodes)

    for key, df in aligned.items():

        if df.shape[1] != expected_columns:
            raise RuntimeError(
                f"Alignment error for '{key}': "
                f"expected {expected_columns} columns, "
                f"but obtained {df.shape[1]}."
            )

    return aligned


def wide_to_matrix(df, nodes):

    return (
        df[nodes]
        .astype(float)
        .values
        .astype(np.float32)
    )


# ============================================================
# 3. Build node-day operational dataset
# ============================================================

def build_node_day_dataset(tables, nodes):
    """
    Build one row for each node-day.

    Predictors use only information available BEFORE the current day.

    Example:
    For date t, features use:
        t-1
        t-2
        t-3
        t-7
        t-14
        past 7-day rolling statistics
        past 14-day rolling statistics

    Targets are today's actual operational values.
    """

    matrices = {
        key: wide_to_matrix(df, nodes)
        for key, df in tables.items()
    }

    dates = tables["w_delivery"]["Date"].values

    total_times = len(dates)
    total_nodes = len(nodes)

    rows = []

    print("\n========== Building Node-Day Dataset ==========")

    for t in range(MAX_LAG, total_times):

        current_date = pd.Timestamp(dates[t])

        for node_index, node in enumerate(nodes):

            row = {
                "date": current_date,
                "time_index": t,
                "node": node,
                "node_index": node_index,
            }

            # ------------------------------------
            # Historical predictors
            # ------------------------------------

            for key in TEMPORAL_KEYS:

                series = matrices[key][:, node_index]

                # Individual lags
                for lag in [1, 2, 3, 7, 14]:

                    row[f"feat_{key}_lag{lag}"] = float(
                        series[t - lag]
                    )

                # Rolling windows strictly before date t
                history_7 = series[t - 7:t]
                history_14 = series[t - 14:t]

                row[f"feat_{key}_roll7_mean"] = float(
                    np.mean(history_7)
                )

                row[f"feat_{key}_roll7_std"] = float(
                    np.std(history_7)
                )

                row[f"feat_{key}_roll14_mean"] = float(
                    np.mean(history_14)
                )

                row[f"feat_{key}_roll14_std"] = float(
                    np.std(history_14)
                )

                # Today's actual value.
                row[f"actual_{key}"] = float(
                    series[t]
                )

            # ------------------------------------
            # Temporal calendar information
            # ------------------------------------

            day_of_week = current_date.dayofweek
            month = current_date.month

            row["feat_day_sin"] = np.sin(
                2 * np.pi * day_of_week / 7
            )

            row["feat_day_cos"] = np.cos(
                2 * np.pi * day_of_week / 7
            )

            row["feat_month_sin"] = np.sin(
                2 * np.pi * month / 12
            )

            row["feat_month_cos"] = np.cos(
                2 * np.pi * month / 12
            )

            rows.append(row)

    data = pd.DataFrame(rows)

    print("Dataset shape:", data.shape)
    print("Number of dates:", data["date"].nunique())
    print("Number of nodes:", data["node"].nunique())

    return data


# ============================================================
# 4. Chronological train / validation / test split
# ============================================================

def assign_chronological_splits(data):
    """
    Split strictly by date.

    First 60%:
        training

    Next 20%:
        validation

    Final 20%:
        test
    """

    data = data.copy()

    unique_dates = sorted(data["date"].unique())

    n_dates = len(unique_dates)

    train_end = int(
        n_dates * TRAIN_FRACTION
    )

    validation_end = int(
        n_dates * (
            TRAIN_FRACTION
            + VALIDATION_FRACTION
        )
    )

    train_dates = set(
        unique_dates[:train_end]
    )

    validation_dates = set(
        unique_dates[
            train_end:validation_end
        ]
    )

    test_dates = set(
        unique_dates[validation_end:]
    )

    data["split"] = "test"

    data.loc[
        data["date"].isin(train_dates),
        "split"
    ] = "train"

    data.loc[
        data["date"].isin(validation_dates),
        "split"
    ] = "validation"

    print("\n========== Chronological Split ==========")

    for split_name in [
        "train",
        "validation",
        "test"
    ]:

        subset = data[
            data["split"] == split_name
        ]

        print(
            split_name,
            "rows=",
            len(subset),
            "dates=",
            subset["date"].nunique(),
            "from",
            subset["date"].min(),
            "to",
            subset["date"].max(),
        )

    return data


# ============================================================
# 5. Build forecasting design matrix
# ============================================================

def build_forecasting_matrix(data):
    """
    Build numerical matrix for pooled normal-operation forecasting.

    Historical temporal features
    +
    one-hot node identity.
    """

    feature_cols = [
        c for c in data.columns
        if c.startswith("feat_")
    ]

    X_numeric = (
        data[feature_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(float)
    )

    node_dummies = pd.get_dummies(
        data["node"],
        prefix="node",
        dtype=float
    )

    X = pd.concat(
        [
            X_numeric.reset_index(drop=True),
            node_dummies.reset_index(drop=True),
        ],
        axis=1
    )

    return X, feature_cols


# ============================================================
# 6. Normal-operation forecasting baselines
# ============================================================

def create_regressor():
    """
    Gradient boosting normal-operation forecasting model.
    """

    return HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.04,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        early_stopping=True,
        random_state=RANDOM_STATE,
    )


def fit_normal_operation_models(data, X):
    """
    For each of the eight operational signals:

    Stage A:
        Fit an initial forecasting model using all training rows.

    Stage B:
        Calculate training residuals.

    Stage C:
        Remove the largest 5% training residuals as possible contamination.

    Stage D:
        Refit a cleaner normal-operation model.

    This reduces the chance that strong anomalies become part of the
    learned definition of "normal".
    """

    data = data.copy()

    train_mask = (
        data["split"] == "train"
    ).values

    X_array = X.values.astype(np.float32)

    model_summary = []

    print(
        "\n========== Normal-Operation Forecasting =========="
    )

    for key in TEMPORAL_KEYS:

        target_col = f"actual_{key}"

        y_actual = (
            data[target_col]
            .astype(float)
            .values
        )

        # Log transform reduces scale skew.
        y_log = np.log1p(
            np.clip(
                y_actual,
                a_min=0.0,
                a_max=None
            )
        )

        # ------------------------------------------
        # Initial model
        # ------------------------------------------

        first_model = create_regressor()

        first_model.fit(
            X_array[train_mask],
            y_log[train_mask]
        )

        first_train_prediction_log = (
            first_model.predict(
                X_array[train_mask]
            )
        )

        first_train_prediction = np.expm1(
            first_train_prediction_log
        )

        first_train_prediction = np.clip(
            first_train_prediction,
            0.0,
            None
        )

        first_train_residual = np.abs(
            y_actual[train_mask]
            - first_train_prediction
        )

        # ------------------------------------------
        # Remove strongest residual contamination
        # ------------------------------------------

        contamination_cutoff = np.quantile(
            first_train_residual,
            NORMAL_BASELINE_CLEAN_QUANTILE
        )

        clean_train_local = (
            first_train_residual
            <= contamination_cutoff
        )

        train_indices = np.where(
            train_mask
        )[0]

        clean_train_indices = (
            train_indices[
                clean_train_local
            ]
        )

        # Safety check.
        if len(clean_train_indices) < 500:

            clean_train_indices = train_indices

        # ------------------------------------------
        # Final normal-operation model
        # ------------------------------------------

        final_model = create_regressor()

        final_model.fit(
            X_array[clean_train_indices],
            y_log[clean_train_indices]
        )

        prediction_log = final_model.predict(
            X_array
        )

        prediction = np.expm1(
            prediction_log
        )

        prediction = np.clip(
            prediction,
            0.0,
            None
        )

        residual = np.abs(
            y_actual
            - prediction
        )

        data[f"pred_{key}"] = prediction
        data[f"abs_residual_{key}"] = residual

        model_summary.append({
            "signal": key,
            "training_rows_initial": int(
                train_mask.sum()
            ),
            "training_rows_after_cleaning": int(
                len(clean_train_indices)
            ),
            "contamination_residual_cutoff": float(
                contamination_cutoff
            ),
            "train_mean_absolute_residual": float(
                np.mean(
                    residual[train_mask]
                )
            ),
        })

        print(
            f"{key}: "
            f"kept {len(clean_train_indices)} "
            f"of {train_mask.sum()} "
            f"training rows."
        )

    model_summary_df = pd.DataFrame(
        model_summary
    )

    model_summary_df.to_csv(
        OUT_DIR
        / "normal_operation_model_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return data


# ============================================================
# 7. Robust TRAINING-ONLY standardisation
# ============================================================

def robust_positive_z_by_node(
    data,
    value_col
):
    """
    Convert a non-negative anomaly magnitude into a robust positive z-score.

    Statistics are calculated using TRAINING DATA ONLY,
    separately for each node.

    z = (value - training median) / robust scale

    Negative z-values are clipped to zero because values below
    the normal median are not considered abnormal magnitudes here.
    """

    output = np.zeros(
        len(data),
        dtype=float
    )

    train_all = data.loc[
        data["split"] == "train",
        value_col
    ].astype(float).values

    global_median = np.median(
        train_all
    )

    global_mad = np.median(
        np.abs(
            train_all
            - global_median
        )
    )

    global_scale = max(
        1.4826 * global_mad,
        np.std(train_all) * 0.25,
        1e-6
    )

    for node, group in data.groupby(
        "node",
        sort=False
    ):

        group_indices = group.index.values

        train_values = group.loc[
            group["split"] == "train",
            value_col
        ].astype(float).values

        if len(train_values) < 10:

            median_value = global_median
            scale_value = global_scale

        else:

            median_value = np.median(
                train_values
            )

            mad = np.median(
                np.abs(
                    train_values
                    - median_value
                )
            )

            q25 = np.quantile(
                train_values,
                0.25
            )

            q75 = np.quantile(
                train_values,
                0.75
            )

            iqr_scale = (
                (q75 - q25)
                / 1.349
            )

            scale_value = max(
                1.4826 * mad,
                iqr_scale,
                global_scale * 0.05,
                1e-6
            )

        values = (
            group[value_col]
            .astype(float)
            .values
        )

        z = (
            values
            - median_value
        ) / scale_value

        # Only positive anomaly magnitude.
        z = np.clip(
            z,
            0.0,
            10.0
        )

        output[group_indices] = z

    return output


# ============================================================
# 8. Six disruption components
# ============================================================

def calculate_anomaly_components(data):
    """
    Build six interpretable operational disruption components:

    1. Delivery residual
    2. Production residual
    3. Factory Issue residual
    4. Sales anomaly
    5. Sales-Delivery mismatch
    6. Production-Delivery mismatch
    """

    data = data.copy()

    print(
        "\n========== Six Anomaly Components =========="
    )

    # --------------------------------------------------------
    # A. Forecast residual primitive scores
    # --------------------------------------------------------

    residual_keys = [
        "u_delivery",
        "w_delivery",
        "u_production",
        "w_production",
        "u_factory",
        "w_factory",
        "u_sales",
        "w_sales",
    ]

    for key in residual_keys:

        raw_col = (
            f"abs_residual_{key}"
        )

        z_col = (
            f"z_residual_{key}"
        )

        data[z_col] = (
            robust_positive_z_by_node(
                data,
                raw_col
            )
        )

    # --------------------------------------------------------
    # B. Delivery residual
    # --------------------------------------------------------

    data["delivery_residual"] = (
        data[
            [
                "z_residual_u_delivery",
                "z_residual_w_delivery",
            ]
        ].mean(axis=1)
    )

    # --------------------------------------------------------
    # C. Production residual
    # --------------------------------------------------------

    data["production_residual"] = (
        data[
            [
                "z_residual_u_production",
                "z_residual_w_production",
            ]
        ].mean(axis=1)
    )

    # --------------------------------------------------------
    # D. Factory Issue residual
    # --------------------------------------------------------

    data["factory_issue_residual"] = (
        data[
            [
                "z_residual_u_factory",
                "z_residual_w_factory",
            ]
        ].mean(axis=1)
    )

    # --------------------------------------------------------
    # E. Sales anomaly
    # --------------------------------------------------------

    data["sales_anomaly"] = (
        data[
            [
                "z_residual_u_sales",
                "z_residual_w_sales",
            ]
        ].mean(axis=1)
    )

    # --------------------------------------------------------
    # F. Sales-Delivery mismatch
    # --------------------------------------------------------

    data["raw_unit_sales_delivery_mismatch"] = (
        np.abs(
            data["actual_u_sales"]
            - data["actual_u_delivery"]
        )
        /
        (
            np.abs(
                data["actual_u_sales"]
            )
            +
            np.abs(
                data["actual_u_delivery"]
            )
            +
            EPS
        )
    )

    data["raw_weight_sales_delivery_mismatch"] = (
        np.abs(
            data["actual_w_sales"]
            - data["actual_w_delivery"]
        )
        /
        (
            np.abs(
                data["actual_w_sales"]
            )
            +
            np.abs(
                data["actual_w_delivery"]
            )
            +
            EPS
        )
    )

    data[
        "z_unit_sales_delivery_mismatch"
    ] = robust_positive_z_by_node(
        data,
        "raw_unit_sales_delivery_mismatch"
    )

    data[
        "z_weight_sales_delivery_mismatch"
    ] = robust_positive_z_by_node(
        data,
        "raw_weight_sales_delivery_mismatch"
    )

    data[
        "sales_delivery_mismatch"
    ] = data[
        [
            "z_unit_sales_delivery_mismatch",
            "z_weight_sales_delivery_mismatch",
        ]
    ].mean(axis=1)

    # --------------------------------------------------------
    # G. Production-Delivery mismatch
    # --------------------------------------------------------

    data[
        "raw_unit_production_delivery_mismatch"
    ] = (
        np.abs(
            data["actual_u_production"]
            - data["actual_u_delivery"]
        )
        /
        (
            np.abs(
                data["actual_u_production"]
            )
            +
            np.abs(
                data["actual_u_delivery"]
            )
            +
            EPS
        )
    )

    data[
        "raw_weight_production_delivery_mismatch"
    ] = (
        np.abs(
            data["actual_w_production"]
            - data["actual_w_delivery"]
        )
        /
        (
            np.abs(
                data["actual_w_production"]
            )
            +
            np.abs(
                data["actual_w_delivery"]
            )
            +
            EPS
        )
    )

    data[
        "z_unit_production_delivery_mismatch"
    ] = robust_positive_z_by_node(
        data,
        "raw_unit_production_delivery_mismatch"
    )

    data[
        "z_weight_production_delivery_mismatch"
    ] = robust_positive_z_by_node(
        data,
        "raw_weight_production_delivery_mismatch"
    )

    data[
        "production_delivery_mismatch"
    ] = data[
        [
            "z_unit_production_delivery_mismatch",
            "z_weight_production_delivery_mismatch",
        ]
    ].mean(axis=1)

    component_cols = [
        "delivery_residual",
        "production_residual",
        "factory_issue_residual",
        "sales_anomaly",
        "sales_delivery_mismatch",
        "production_delivery_mismatch",
    ]

    print("Components:")

    for col in component_cols:

        print(
            f" - {col}: "
            f"mean={data[col].mean():.4f}, "
            f"max={data[col].max():.4f}"
        )

    return data, component_cols


# ============================================================
# 9. Robust multivariate anomaly score
# ============================================================

def build_multivariate_anomaly_score(
    data,
    component_cols
):
    """
    Robust top-k multivariate anomaly aggregation.

    Instead of simply averaging all six components,
    use the mean of the strongest three component scores.

    Reason:
    - One isolated noisy component should not completely dominate.
    - A true operational disruption can manifest through different
      combinations of signals.
    """

    data = data.copy()

    component_matrix = (
        data[component_cols]
        .astype(float)
        .values
    )

    sorted_scores = np.sort(
        component_matrix,
        axis=1
    )

    strongest_three = (
        sorted_scores[:, -3:]
    )

    data[
        "multivariate_anomaly_score"
    ] = strongest_three.mean(
        axis=1
    )

    # Main operational driver.
    driver_index = np.argmax(
        component_matrix,
        axis=1
    )

    data["main_anomaly_driver"] = [
        component_cols[i]
        for i in driver_index
    ]

    print(
        "\n========== Multivariate Anomaly Score =========="
    )

    print(
        data[
            "multivariate_anomaly_score"
        ].describe()
    )

    return data


# ============================================================
# 10. Threshold sensitivity
# ============================================================

def count_qualified_onsets(
    data,
    threshold
):
    """
    Count qualified onsets for a candidate anomaly threshold.
    """

    total_onsets = 0

    for node, group in data.groupby(
        "node"
    ):

        group = (
            group
            .sort_values("date")
            .reset_index(drop=True)
        )

        anomaly = (
            group[
                "multivariate_anomaly_score"
            ].values
            >= threshold
        )

        for i in range(
            len(group)
        ):

            if not anomaly[i]:
                continue

            previous_anomaly = (
                anomaly[i - 1]
                if i > 0
                else False
            )

            if previous_anomaly:
                continue

            if i < MIN_NORMAL_DAYS_BEFORE_ONSET:
                continue

            previous_normal = (
                ~anomaly[
                    i
                    - MIN_NORMAL_DAYS_BEFORE_ONSET
                    :i
                ]
            ).all()

            if previous_normal:

                total_onsets += 1

    return total_onsets


def build_threshold_sensitivity(data):
    """
    Diagnostic only.

    It reports what happens at several pre-specified thresholds.
    The official threshold remains ANOMALY_QUANTILE = 0.975
    unless changed explicitly before final model training.
    """

    train_scores = data.loc[
        data["split"] == "train",
        "multivariate_anomaly_score"
    ].values

    rows = []

    for quantile in [
        0.95,
        0.975,
        0.99,
    ]:

        threshold = np.quantile(
            train_scores,
            quantile
        )

        anomaly_rate = (
            data[
                "multivariate_anomaly_score"
            ]
            >= threshold
        ).mean()

        onset_count = count_qualified_onsets(
            data,
            threshold
        )

        rows.append({
            "training_quantile": quantile,
            "threshold": threshold,
            "overall_anomaly_state_rate": anomaly_rate,
            "qualified_onset_count": onset_count,
        })

    sensitivity = pd.DataFrame(rows)

    sensitivity.to_csv(
        OUT_DIR
        / "threshold_sensitivity.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(
        "\n========== Threshold Sensitivity =========="
    )

    print(sensitivity)

    return sensitivity


# ============================================================
# 11. Identify anomaly state
# ============================================================

def assign_anomaly_state(data):
    """
    Official threshold calculated from training scores only.
    """

    data = data.copy()

    train_scores = data.loc[
        data["split"] == "train",
        "multivariate_anomaly_score"
    ].values

    threshold = np.quantile(
        train_scores,
        ANOMALY_QUANTILE
    )

    data["is_anomaly_state"] = (
        data[
            "multivariate_anomaly_score"
        ]
        >= threshold
    ).astype(int)

    print(
        "\n========== Official Anomaly State =========="
    )

    print(
        "Training quantile:",
        ANOMALY_QUANTILE
    )

    print(
        "Anomaly threshold:",
        threshold
    )

    print(
        "Overall anomaly-state rate:",
        data[
            "is_anomaly_state"
        ].mean()
    )

    print(
        "Training anomaly-state rate:",
        data.loc[
            data["split"] == "train",
            "is_anomaly_state"
        ].mean()
    )

    print(
        "Validation anomaly-state rate:",
        data.loc[
            data["split"] == "validation",
            "is_anomaly_state"
        ].mean()
    )

    print(
        "Test anomaly-state rate:",
        data.loc[
            data["split"] == "test",
            "is_anomaly_state"
        ].mean()
    )

    return data, float(threshold)


# ============================================================
# 12. Merge anomaly states into disruption events
# ============================================================

def detect_disruption_events(
    data,
    component_cols
):
    """
    Merge consecutive anomaly states into disruption events.

    Qualified disruption onset:
    - first abnormal day of an event
    - preceded by at least MIN_NORMAL_DAYS_BEFORE_ONSET
      consecutive normal observations.
    """

    data = data.copy()

    data["event_id"] = ""
    data["is_disruption_onset"] = 0

    all_events = []

    event_counter = 0

    for node, group in data.groupby(
        "node",
        sort=False
    ):

        group = (
            group
            .sort_values("date")
        )

        indices = group.index.values

        anomaly = (
            group[
                "is_anomaly_state"
            ]
            .astype(int)
            .values
        )

        dates = (
            group["date"]
            .values
        )

        i = 0

        while i < len(group):

            if anomaly[i] == 0:

                i += 1
                continue

            # Start of event.
            start_i = i

            end_i = i

            while (
                end_i + 1 < len(group)
                and
                anomaly[end_i + 1] == 1
            ):

                end_i += 1

            event_counter += 1

            event_id = (
                f"EVENT_{event_counter:04d}"
            )

            event_indices = (
                indices[
                    start_i:end_i + 1
                ]
            )

            data.loc[
                event_indices,
                "event_id"
            ] = event_id

            # --------------------------------------
            # Qualified onset rule
            # --------------------------------------

            qualified_onset = False

            previous_normal_days = 0

            if (
                start_i
                >= MIN_NORMAL_DAYS_BEFORE_ONSET
            ):

                previous_period = (
                    anomaly[
                        start_i
                        - MIN_NORMAL_DAYS_BEFORE_ONSET
                        :start_i
                    ]
                )

                if (
                    previous_period == 0
                ).all():

                    qualified_onset = True

            # Count consecutive normal observations
            # before event start.
            cursor = start_i - 1

            while (
                cursor >= 0
                and anomaly[cursor] == 0
            ):

                previous_normal_days += 1

                cursor -= 1

            onset_index = indices[
                start_i
            ]

            if qualified_onset:

                data.loc[
                    onset_index,
                    "is_disruption_onset"
                ] = 1

            onset_row = data.loc[
                onset_index
            ]

            component_values = (
                onset_row[
                    component_cols
                ]
                .astype(float)
            )

            main_driver = (
                component_values.idxmax()
            )

            start_date = pd.Timestamp(
                dates[start_i]
            )

            end_date = pd.Timestamp(
                dates[end_i]
            )

            all_events.append({
                "event_id": event_id,
                "node": node,
                "onset_date": start_date,
                "end_date": end_date,
                "duration_observations": int(
                    end_i - start_i + 1
                ),
                "previous_consecutive_normal_days": int(
                    previous_normal_days
                ),
                "qualified_disruption_onset": int(
                    qualified_onset
                ),
                "onset_anomaly_score": float(
                    onset_row[
                        "multivariate_anomaly_score"
                    ]
                ),
                "peak_anomaly_score": float(
                    data.loc[
                        event_indices,
                        "multivariate_anomaly_score"
                    ].max()
                ),
                "main_driver_at_onset": main_driver,
                "split_at_onset": onset_row[
                    "split"
                ],
            })

            i = end_i + 1

    events = pd.DataFrame(
        all_events
    )

    print(
        "\n========== Disruption Events =========="
    )

    print(
        "All merged events:",
        len(events)
    )

    qualified = events[
        events[
            "qualified_disruption_onset"
        ] == 1
    ].copy()

    print(
        "Qualified disruption onsets:",
        len(qualified)
    )

    if len(qualified) > 0:

        print(
            "\nQualified onsets by split:"
        )

        print(
            qualified[
                "split_at_onset"
            ].value_counts()
        )

        print(
            "\nMain drivers:"
        )

        print(
            qualified[
                "main_driver_at_onset"
            ].value_counts()
        )

    return data, events, qualified


# ============================================================
# 13. Future onset labels
# ============================================================

def build_future_onset_labels(data):
    """
    Construct:

    next_day_onset_label:
        Will a qualified disruption onset occur tomorrow?

    next_3_day_onset_label:
        Will a qualified disruption onset occur within the next 3 days?

    Early-warning eligibility:
        Current observation must be normal.

    Final rows without sufficient future observations receive NA.
    """

    data = data.copy()

    data[
        "eligible_for_early_warning"
    ] = (
        data[
            "is_anomaly_state"
        ] == 0
    ).astype(int)

    data["next_day_onset_label"] = pd.NA
    data["next_3_day_onset_label"] = pd.NA

    for node, group in data.groupby(
        "node",
        sort=False
    ):

        group = (
            group
            .sort_values("date")
        )

        indices = group.index.values

        onset = (
            group[
                "is_disruption_onset"
            ]
            .astype(int)
            .values
        )

        n = len(group)

        for i in range(n):

            # Next-day target.
            if i + 1 < n:

                data.loc[
                    indices[i],
                    "next_day_onset_label"
                ] = int(
                    onset[i + 1]
                )

            # Next-three-day target.
            if i + 3 < n:

                future_three = (
                    onset[i + 1:i + 4]
                )

                data.loc[
                    indices[i],
                    "next_3_day_onset_label"
                ] = int(
                    future_three.max()
                )

    data[
        "next_day_onset_label"
    ] = data[
        "next_day_onset_label"
    ].astype("Int64")

    data[
        "next_3_day_onset_label"
    ] = data[
        "next_3_day_onset_label"
    ].astype("Int64")

    return data


# ============================================================
# 14. Summary tables
# ============================================================

def build_label_summary(data, events):
    """
    Summarise event and label distributions.
    """

    rows = []

    for split_name in [
        "train",
        "validation",
        "test",
        "all",
    ]:

        if split_name == "all":

            subset = data.copy()

        else:

            subset = data[
                data["split"]
                == split_name
            ].copy()

        eligible = subset[
            subset[
                "eligible_for_early_warning"
            ] == 1
        ]

        valid_next1 = eligible[
            eligible[
                "next_day_onset_label"
            ].notna()
        ]

        valid_next3 = eligible[
            eligible[
                "next_3_day_onset_label"
            ].notna()
        ]

        rows.append({
            "split": split_name,
            "node_days": len(subset),
            "unique_dates": subset[
                "date"
            ].nunique(),
            "unique_nodes": subset[
                "node"
            ].nunique(),
            "anomaly_state_count": int(
                subset[
                    "is_anomaly_state"
                ].sum()
            ),
            "anomaly_state_rate": float(
                subset[
                    "is_anomaly_state"
                ].mean()
            ),
            "qualified_onset_count": int(
                subset[
                    "is_disruption_onset"
                ].sum()
            ),
            "eligible_early_warning_rows": int(
                len(eligible)
            ),
            "next_day_positive_labels": int(
                valid_next1[
                    "next_day_onset_label"
                ].sum()
            ),
            "next_3_day_positive_labels": int(
                valid_next3[
                    "next_3_day_onset_label"
                ].sum()
            ),
        })

    summary = pd.DataFrame(rows)

    print(
        "\n========== Onset Label Summary =========="
    )

    print(summary)

    return summary


# ============================================================
# 15. Figures
# ============================================================

def figure_1_anomaly_timeline(
    data,
    qualified_events,
    threshold
):
    """
    Plot anomaly timeline for the node with the most disruption onsets.
    """

    if len(qualified_events) > 0:

        chosen_node = (
            qualified_events[
                "node"
            ]
            .value_counts()
            .index[0]
        )

    else:

        chosen_node = (
            data.groupby("node")[
                "multivariate_anomaly_score"
            ]
            .max()
            .idxmax()
        )

    plot_df = (
        data[
            data["node"] == chosen_node
        ]
        .sort_values("date")
        .copy()
    )

    onset_df = plot_df[
        plot_df[
            "is_disruption_onset"
        ] == 1
    ]

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        plot_df["date"],
        plot_df[
            "multivariate_anomaly_score"
        ],
        linewidth=1.5,
        label="Multivariate anomaly score"
    )

    plt.axhline(
        threshold,
        linestyle="--",
        linewidth=1.5,
        label=(
            f"Training-only threshold "
            f"({ANOMALY_QUANTILE:.3f} quantile)"
        )
    )

    if len(onset_df) > 0:

        plt.scatter(
            onset_df["date"],
            onset_df[
                "multivariate_anomaly_score"
            ],
            s=70,
            marker="o",
            label="Qualified disruption onset"
        )

    plt.xlabel("Date")
    plt.ylabel("Robust anomaly score")

    plt.title(
        f"Operational anomaly timeline for {chosen_node}"
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        FIG_DIR
        / "figure_1_anomaly_score_timeline.png",
        dpi=300
    )

    plt.close()


def figure_2_onsets_by_node(
    qualified_events
):
    """
    Number of qualified disruption onsets by node.
    """

    if len(qualified_events) == 0:

        return

    counts = (
        qualified_events[
            "node"
        ]
        .value_counts()
        .sort_values(
            ascending=True
        )
    )

    plt.figure(
        figsize=(9, 8)
    )

    plt.barh(
        counts.index,
        counts.values
    )

    plt.xlabel(
        "Number of qualified disruption onsets"
    )

    plt.ylabel(
        "Product node"
    )

    plt.title(
        "Disruption onsets by SupplyGraph node"
    )

    plt.grid(
        axis="x",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        FIG_DIR
        / "figure_2_disruption_onsets_by_node.png",
        dpi=300
    )

    plt.close()


def figure_3_onsets_over_time(
    qualified_events
):
    """
    Monthly onset counts.
    """

    if len(qualified_events) == 0:

        return

    temp = qualified_events.copy()

    temp["month"] = (
        pd.to_datetime(
            temp["onset_date"]
        )
        .dt.to_period("M")
        .astype(str)
    )

    monthly = (
        temp.groupby(
            "month"
        )
        .size()
    )

    plt.figure(
        figsize=(10, 5)
    )

    plt.bar(
        monthly.index,
        monthly.values
    )

    plt.xlabel("Month")
    plt.ylabel(
        "Number of qualified disruption onsets"
    )

    plt.title(
        "Disruption onsets over time"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        FIG_DIR
        / "figure_3_disruption_onsets_over_time.png",
        dpi=300
    )

    plt.close()


def robust_signal_z(
    node_df,
    column
):
    """
    Robustly standardise an actual operational signal
    using the training period of the selected node.
    """

    train_values = node_df.loc[
        node_df["split"] == "train",
        column
    ].astype(float).values

    median = np.median(
        train_values
    )

    mad = np.median(
        np.abs(
            train_values
            - median
        )
    )

    scale = max(
        1.4826 * mad,
        np.std(
            train_values
        ) * 0.1,
        1e-6
    )

    return (
        node_df[column]
        - median
    ) / scale


def figure_4_operational_signal_example(
    data,
    qualified_events
):
    """
    Plot operational signals around the strongest qualified disruption onset.
    """

    if len(qualified_events) == 0:

        return

    strongest = (
        qualified_events
        .sort_values(
            "onset_anomaly_score",
            ascending=False
        )
        .iloc[0]
    )

    node = strongest["node"]

    onset_date = pd.Timestamp(
        strongest[
            "onset_date"
        ]
    )

    node_df = (
        data[
            data["node"] == node
        ]
        .sort_values("date")
        .copy()
    )

    start_date = (
        onset_date
        - pd.Timedelta(
            days=14
        )
    )

    end_date = (
        onset_date
        + pd.Timedelta(
            days=14
        )
    )

    window = node_df[
        (
            node_df["date"]
            >= start_date
        )
        &
        (
            node_df["date"]
            <= end_date
        )
    ].copy()

    signal_cols = {
        "Sales Order":
            "actual_w_sales",

        "Production":
            "actual_w_production",

        "Factory Issue Flow":
            "actual_w_factory",

        "Delivery":
            "actual_w_delivery",
    }

    plt.figure(
        figsize=(12, 6)
    )

    for label, column in signal_cols.items():

        node_df[
            f"plot_z_{column}"
        ] = robust_signal_z(
            node_df,
            column
        )

        plot_values = (
            node_df.loc[
                window.index,
                f"plot_z_{column}"
            ]
        )

        plt.plot(
            window["date"],
            plot_values,
            linewidth=1.5,
            label=label
        )

    plt.axvline(
        onset_date,
        linestyle="--",
        linewidth=2,
        label="Disruption onset"
    )

    plt.xlabel("Date")

    plt.ylabel(
        "Robustly standardised operational signal"
    )

    plt.title(
        f"Operational signals around a disruption onset: {node}"
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        FIG_DIR
        / "figure_4_operational_signal_example.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 16. Main
# ============================================================

def main():

    print(
        "SupplyGraph folder:"
    )

    print(
        SG_DIR
    )

    # --------------------------------------------------------
    # Step 1:
    # Read eight temporal tables.
    # --------------------------------------------------------

    tables = (
        load_temporal_tables()
    )

    # --------------------------------------------------------
    # Step 2:
    # Load and align 41 product nodes.
    # --------------------------------------------------------

    nodes = load_nodes(
        tables
    )

    tables = (
        align_temporal_tables(
            tables,
            nodes
        )
    )

    # --------------------------------------------------------
    # Step 3:
    # Build node-day operational dataset.
    # --------------------------------------------------------

    data = (
        build_node_day_dataset(
            tables,
            nodes
        )
    )

    data = (
        assign_chronological_splits(
            data
        )
    )

    # Save original node-day operational data.
    data.to_csv(
        OUT_DIR
        / "node_day_operational_signals.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Step 4:
    # Build normal-operation forecasting baseline.
    # --------------------------------------------------------

    X, feature_cols = (
        build_forecasting_matrix(
            data
        )
    )

    data = (
        fit_normal_operation_models(
            data,
            X
        )
    )

    # --------------------------------------------------------
    # Step 5:
    # Six interpretable anomaly components.
    # --------------------------------------------------------

    data, component_cols = (
        calculate_anomaly_components(
            data
        )
    )

    # --------------------------------------------------------
    # Step 6:
    # Robust multivariate anomaly score.
    # --------------------------------------------------------

    data = (
        build_multivariate_anomaly_score(
            data,
            component_cols
        )
    )

    # Diagnostic sensitivity table.
    build_threshold_sensitivity(
        data
    )

    # Official training-only anomaly threshold.
    data, anomaly_threshold = (
        assign_anomaly_state(
            data
        )
    )

    # --------------------------------------------------------
    # Step 7 + Step 8:
    # Merge consecutive abnormal observations into events
    # and identify qualified disruption onsets.
    # --------------------------------------------------------

    (
        data,
        all_events,
        qualified_events,
    ) = detect_disruption_events(
        data,
        component_cols
    )

    # --------------------------------------------------------
    # Step 9:
    # Future early-warning labels.
    # --------------------------------------------------------

    data = (
        build_future_onset_labels(
            data
        )
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    label_summary = (
        build_label_summary(
            data,
            all_events
        )
    )

    # --------------------------------------------------------
    # Save final files.
    # --------------------------------------------------------

    anomaly_output_cols = [
        "date",
        "time_index",
        "node",
        "node_index",
        "split",

        "actual_u_delivery",
        "actual_u_factory",
        "actual_u_production",
        "actual_u_sales",

        "actual_w_delivery",
        "actual_w_factory",
        "actual_w_production",
        "actual_w_sales",

        "pred_u_delivery",
        "pred_u_factory",
        "pred_u_production",
        "pred_u_sales",

        "pred_w_delivery",
        "pred_w_factory",
        "pred_w_production",
        "pred_w_sales",

        "delivery_residual",
        "production_residual",
        "factory_issue_residual",
        "sales_anomaly",
        "sales_delivery_mismatch",
        "production_delivery_mismatch",

        "multivariate_anomaly_score",
        "main_anomaly_driver",

        "is_anomaly_state",
        "event_id",
        "is_disruption_onset",

        "eligible_for_early_warning",

        "next_day_onset_label",
        "next_3_day_onset_label",
    ]

    data[
        anomaly_output_cols
    ].to_csv(
        OUT_DIR
        / "node_day_anomaly_scores.csv",
        index=False,
        encoding="utf-8-sig"
    )

    all_events.to_csv(
        OUT_DIR
        / "disruption_event_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    qualified_events.to_csv(
        OUT_DIR
        / "disruption_onset_events.csv",
        index=False,
        encoding="utf-8-sig"
    )

    label_summary.to_csv(
        OUT_DIR
        / "onset_label_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame([{
        "official_anomaly_quantile":
            ANOMALY_QUANTILE,

        "official_anomaly_threshold":
            anomaly_threshold,

        "minimum_normal_days_before_onset":
            MIN_NORMAL_DAYS_BEFORE_ONSET,

        "total_node_days":
            len(data),

        "total_anomaly_states":
            int(
                data[
                    "is_anomaly_state"
                ].sum()
            ),

        "total_merged_events":
            len(all_events),

        "qualified_disruption_onsets":
            len(qualified_events),

    }]).to_csv(
        OUT_DIR
        / "disruption_definition_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Figures.
    # --------------------------------------------------------

    figure_1_anomaly_timeline(
        data,
        qualified_events,
        anomaly_threshold
    )

    figure_2_onsets_by_node(
        qualified_events
    )

    figure_3_onsets_over_time(
        qualified_events
    )

    figure_4_operational_signal_example(
        data,
        qualified_events
    )

    # --------------------------------------------------------
    # Final console output.
    # --------------------------------------------------------

    print(
        "\n=============================================="
    )

    print(
        "FINISHED BUILDING DISRUPTION EVENTS"
    )

    print(
        "=============================================="
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
        "1. node_day_operational_signals.csv"
    )

    print(
        "2. node_day_anomaly_scores.csv"
    )

    print(
        "3. disruption_event_summary.csv"
    )

    print(
        "4. disruption_onset_events.csv"
    )

    print(
        "5. onset_label_summary.csv"
    )

    print(
        "6. threshold_sensitivity.csv"
    )

    print(
        "7. disruption_definition_summary.csv"
    )

    print(
        "\nImportant figures:"
    )

    print(
        "1. figure_1_anomaly_score_timeline.png"
    )

    print(
        "2. figure_2_disruption_onsets_by_node.png"
    )

    print(
        "3. figure_3_disruption_onsets_over_time.png"
    )

    print(
        "4. figure_4_operational_signal_example.png"
    )


if __name__ == "__main__":
    main()