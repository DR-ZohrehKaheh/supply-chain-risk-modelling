# Project supervisor: Dr. Zohreh Kaheh.
# Undergraduate project collaborator: Yuyang Zhang.
# Initial student coding was developed with supervisory teaching and improvement guidance.
# -*- coding: utf-8 -*-

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
import tempfile

import numpy as np
import pandas as pd
import torch
import os
torch.set_num_threads(int(os.environ.get("PORTFOLIO_TORCH_THREADS", "1")))


# ============================================================
# 0. Paths and frozen design
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

STAGE3B_SCRIPT = BASE_DIR / "03b_train_genuine_hgnn_baselines.py"
STAGE3E_SCRIPT = BASE_DIR / "03e_train_credible_mlp_gnn_baselines.py"

STAGE3E_DIR = BASE_DIR / "credible_baseline_results"
STAGE3E_CHECKPOINT_DIR = STAGE3E_DIR / "checkpoints"
STAGE3E_PREDICTION_DIR = STAGE3E_DIR / "predictions"

HGNN_FREEZE_DIR = BASE_DIR / "hgnn_extension_frozen"
HGNN_FREEZE_MANIFEST = (
    HGNN_FREEZE_DIR / "hgnn_extension_freeze_manifest.json"
)
HGNN_FREEZE_PLAN = HGNN_FREEZE_DIR / "hgnn_extension_evaluation_plan.json"
HGNN_FREEZE_CHECKSUMS = (
    HGNN_FREEZE_DIR / "hgnn_extension_checkpoint_checksums.csv"
)
HGNN_FREEZE_AUDIT = (
    HGNN_FREEZE_DIR / "hgnn_extension_freeze_audit.csv"
)
HGNN_FREEZE_MARKER = HGNN_FREEZE_DIR / "FREEZE_COMPLETE.txt"
HGNN_TEST_SUMMARY = (
    BASE_DIR / "hgnn_extension_test_results" / "hgnn_test_result_summary.json"
)

FREEZE_DIR = BASE_DIR / "credible_baseline_frozen"

MODEL_NAMES = [
    "Temporal-MLP",
    "Clique-GCN-Gated-Residual",
    "Clique-GraphSAGE-Gated-Residual",
]
FULL_HGNN_NAME = "HGNN-Gated-Residual"
PARAMETER_MATCH_TOLERANCE = 0.05

STAGE3E_OUTPUT_NAMES = [
    "stage3e_design_manifest.json",
    "credible_baseline_seed_metrics_validation.csv",
    "credible_baseline_validation_model_comparison.csv",
    "credible_baseline_parameter_report.csv",
    "credible_baseline_validation_moving_block_bootstrap.csv",
    "credible_baseline_training_history.csv",
    "stage3e_audit_report.csv",
    "predictions/validation_credible_baseline_ensemble_predictions.csv",
]

PACKAGE_NAMES = [
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

EXPECTED_STAGE3E_AUDIT_CHECKS = {
    "protocol_matches_stage1_package",
    "development_horizon_excludes_test_origins",
    "all_models_trained_for_all_locked_seeds",
    "temporal_mlp_parameter_matched_within_5pct",
    "pairwise_and_hgnn_operator_differences_recorded",
    "no_test_prediction_file_created",
    "held_out_test_performance_not_calculated",
}


# ============================================================
# 1. Utilities
# ============================================================

def sha256_file(path):
    path = Path(path)
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_int64_array(values):
    array = np.ascontiguousarray(np.asarray(values, dtype=np.int64))
    return hashlib.sha256(array.tobytes()).hexdigest()


def relative_path(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(BASE_DIR.resolve()).as_posix()
    except ValueError:
        return str(path)


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(value, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def load_stage3b():
    if not STAGE3B_SCRIPT.exists():
        raise FileNotFoundError(STAGE3B_SCRIPT)
    spec = importlib.util.spec_from_file_location(
        "stage3b_hgnn_for_stage3f",
        STAGE3B_SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checkpoint_path(model_name, seed):
    return (
        STAGE3E_CHECKPOINT_DIR
        / model_name.replace("-", "_")
        / f"seed_{seed}.pt"
    )


def add_check(rows, name, passed, details):
    rows.append(
        {
            "check_name": name,
            "status": "PASS" if bool(passed) else "FAIL",
            "details": str(details),
        }
    )


def stop_if_failed(rows, context):
    failed = [row for row in rows if row["status"] == "FAIL"]
    if failed:
        print("\n" + "=" * 72)
        print(f"{context}: FAILED")
        print("=" * 72)
        print(pd.DataFrame(rows).to_string(index=False))
        raise RuntimeError(
            f"{context} failed; no credible-baseline freeze was created."
        )


def checksum_record(role, path):
    path = Path(path)
    return {
        "role": role,
        "relative_path": relative_path(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


# ============================================================
# 2. Freeze
# ============================================================

def main():
    print("=" * 72)
    print("STAGE 3f: FREEZE CREDIBLE BASELINES")
    print("VALIDATION-SELECTED CHECKPOINTS ONLY; NO TEST EVALUATION")
    print("=" * 72)

    if FREEZE_DIR.exists():
        raise FileExistsError(
            "Refusing to overwrite the existing freeze directory: "
            f"{FREEZE_DIR}"
        )

    audit_rows = []

    required_paths = [
        STAGE3B_SCRIPT,
        STAGE3E_SCRIPT,
        *[STAGE3E_DIR / name for name in STAGE3E_OUTPUT_NAMES],
        HGNN_FREEZE_MANIFEST,
        HGNN_FREEZE_PLAN,
        HGNN_FREEZE_CHECKSUMS,
        HGNN_FREEZE_AUDIT,
        HGNN_FREEZE_MARKER,
    ]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    add_check(
        audit_rows,
        "all_required_source_and_freeze_files_exist",
        not missing_paths,
        missing_paths,
    )
    stop_if_failed(audit_rows, "Stage 3f prerequisite audit")

    s3b = load_stage3b()
    protocol, protocol_hash = s3b.load_locked_protocol()
    package = s3b.load_package()
    seeds = [int(seed) for seed in protocol["random_seeds"]]

    design_manifest = load_json(
        STAGE3E_DIR / "stage3e_design_manifest.json"
    )
    stage3e_audit = pd.read_csv(STAGE3E_DIR / "stage3e_audit_report.csv")
    seed_metrics = pd.read_csv(
        STAGE3E_DIR / "credible_baseline_seed_metrics_validation.csv"
    )
    parameter_report = pd.read_csv(
        STAGE3E_DIR / "credible_baseline_parameter_report.csv"
    )
    validation_comparison = pd.read_csv(
        STAGE3E_DIR / "credible_baseline_validation_model_comparison.csv"
    )
    validation_bootstrap = pd.read_csv(
        STAGE3E_DIR
        / "credible_baseline_validation_moving_block_bootstrap.csv"
    )
    validation_predictions = pd.read_csv(
        STAGE3E_PREDICTION_DIR
        / "validation_credible_baseline_ensemble_predictions.csv"
    )

    add_check(
        audit_rows,
        "stage3e_audit_contains_expected_checks",
        set(stage3e_audit["check_name"].astype(str))
        == EXPECTED_STAGE3E_AUDIT_CHECKS,
        stage3e_audit[["check_name", "status"]].to_dict("records"),
    )
    add_check(
        audit_rows,
        "stage3e_audit_all_pass",
        len(stage3e_audit) == len(EXPECTED_STAGE3E_AUDIT_CHECKS)
        and stage3e_audit["status"].astype(str).eq("PASS").all(),
        stage3e_audit[["check_name", "status"]].to_dict("records"),
    )
    add_check(
        audit_rows,
        "stage3e_design_was_created_without_test_evaluation",
        design_manifest.get("created_without_held_out_test_evaluation") is True,
        design_manifest.get("held_out_test_policy"),
    )
    add_check(
        audit_rows,
        "stage3e_models_match_frozen_design",
        design_manifest.get("models") == MODEL_NAMES,
        design_manifest.get("models"),
    )
    add_check(
        audit_rows,
        "protocol_hash_consistent_across_design_and_package",
        design_manifest.get("protocol_sha256") == protocol_hash
        == package["package_protocol_hash"],
        {
            "stage3e_design": design_manifest.get("protocol_sha256"),
            "current_protocol": protocol_hash,
            "package": package["package_protocol_hash"],
        },
    )
    add_check(
        audit_rows,
        "stage3b_script_hash_matches_stage3e_design",
        design_manifest.get("stage3b_script_sha256")
        == sha256_file(STAGE3B_SCRIPT),
        {
            "stage3e_design": design_manifest.get(
                "stage3b_script_sha256"
            ),
            "current": sha256_file(STAGE3B_SCRIPT),
        },
    )

    expected_pairs = {
        (int(seed), model) for seed in seeds for model in MODEL_NAMES
    }
    observed_seed_metric_pairs = set(
        zip(
            seed_metrics["seed"].astype(int),
            seed_metrics["model"].astype(str),
        )
    )
    observed_parameter_pairs = set(
        zip(
            parameter_report["seed"].astype(int),
            parameter_report["model"].astype(str),
        )
    )
    add_check(
        audit_rows,
        "seed_metric_rows_cover_all_locked_models_and_seeds",
        len(seed_metrics) == len(expected_pairs)
        and observed_seed_metric_pairs == expected_pairs,
        {
            "rows": len(seed_metrics),
            "expected_rows": len(expected_pairs),
        },
    )
    add_check(
        audit_rows,
        "parameter_rows_cover_all_locked_models_and_seeds",
        len(parameter_report) == len(expected_pairs)
        and observed_parameter_pairs == expected_pairs,
        {
            "rows": len(parameter_report),
            "expected_rows": len(expected_pairs),
        },
    )
    max_parameter_difference = float(
        parameter_report["relative_difference_vs_full_hgnn"].max()
    )
    add_check(
        audit_rows,
        "all_credible_baselines_parameter_matched_within_5pct",
        np.isfinite(max_parameter_difference)
        and max_parameter_difference <= PARAMETER_MATCH_TOLERANCE,
        {
            "maximum_relative_difference": max_parameter_difference,
            "tolerance": PARAMETER_MATCH_TOLERANCE,
        },
    )

    expected_comparison_models = {
        FULL_HGNN_NAME + " validation ensemble",
        *[name + " validation ensemble" for name in MODEL_NAMES],
    }
    add_check(
        audit_rows,
        "validation_comparison_has_exact_frozen_model_set",
        set(validation_comparison["model"].astype(str))
        == expected_comparison_models,
        validation_comparison["model"].astype(str).tolist(),
    )
    add_check(
        audit_rows,
        "validation_bootstrap_has_all_three_hgnn_comparisons",
        len(validation_bootstrap) == len(MODEL_NAMES)
        and validation_bootstrap["split"].astype(str).eq("validation").all()
        and validation_bootstrap["candidate"].astype(str)
        .eq(FULL_HGNN_NAME)
        .all()
        and set(validation_bootstrap["baseline"].astype(str))
        == set(MODEL_NAMES),
        validation_bootstrap[
            ["candidate", "baseline", "metric"]
        ].to_dict("records"),
    )

    validation_indices = package["validation_indices"].astype(np.int64)
    test_indices = package["test_indices"].astype(np.int64)
    train_indices = package["train_indices"].astype(np.int64)
    aligned_validation_predictions = (
        "package_sample_index" in validation_predictions.columns
        and len(validation_predictions) == len(validation_indices)
        and np.array_equal(
            validation_predictions["package_sample_index"]
            .astype(np.int64)
            .to_numpy(),
            validation_indices,
        )
    )
    add_check(
        audit_rows,
        "validation_predictions_align_with_locked_validation_indices",
        aligned_validation_predictions,
        {
            "prediction_rows": len(validation_predictions),
            "expected_rows": len(validation_indices),
        },
    )
    if "y_3_day" in validation_predictions.columns:
        saved_validation_labels = validation_predictions[
            "y_3_day"
        ].astype(int).to_numpy()
        expected_validation_labels = package["y_3_day"][
            validation_indices
        ].astype(int)
        labels_match = np.array_equal(
            saved_validation_labels,
            expected_validation_labels,
        )
    else:
        labels_match = False
    add_check(
        audit_rows,
        "validation_prediction_labels_match_locked_package",
        labels_match,
        "y_3_day alignment",
    )

    development_indices = np.concatenate([train_indices, validation_indices])
    maximum_development_time = int(
        package["metadata"]
        .iloc[development_indices]["raw_time_index"]
        .max()
    )
    minimum_test_time = int(
        package["metadata"].iloc[test_indices]["raw_time_index"].min()
    )
    add_check(
        audit_rows,
        "development_horizon_stops_before_test_origins",
        maximum_development_time < minimum_test_time,
        {
            "max_development_raw_time_index": maximum_development_time,
            "min_test_raw_time_index": minimum_test_time,
        },
    )

    test_named_stage3e_files = [
        relative_path(path)
        for path in STAGE3E_DIR.rglob("*")
        if path.is_file() and "test" in path.name.lower()
    ]
    add_check(
        audit_rows,
        "stage3e_directory_contains_no_test_named_output",
        not test_named_stage3e_files,
        test_named_stage3e_files,
    )

    expected_checkpoint_paths = {
        checkpoint_path(model, seed).resolve()
        for seed in seeds
        for model in MODEL_NAMES
    }
    actual_checkpoint_paths = {
        path.resolve() for path in STAGE3E_CHECKPOINT_DIR.rglob("*.pt")
    }
    add_check(
        audit_rows,
        "exactly_thirty_expected_stage3e_checkpoints_present",
        actual_checkpoint_paths == expected_checkpoint_paths
        and len(actual_checkpoint_paths) == len(expected_pairs),
        {
            "actual": len(actual_checkpoint_paths),
            "expected": len(expected_checkpoint_paths),
            "missing": sorted(
                relative_path(path)
                for path in expected_checkpoint_paths
                - actual_checkpoint_paths
            ),
            "unexpected": sorted(
                relative_path(path)
                for path in actual_checkpoint_paths
                - expected_checkpoint_paths
            ),
        },
    )
    stop_if_failed(audit_rows, "Stage 3f development-output audit")

    checkpoint_rows = []
    checkpoint_metadata_ok = True
    checkpoint_validation_metrics_ok = True
    checkpoint_parameter_counts_ok = True
    device = torch.device("cpu")

    for seed in seeds:
        for model in MODEL_NAMES:
            path = checkpoint_path(model, seed)
            checkpoint = s3b.safe_torch_load(path, device)

            metadata_ok = (
                str(checkpoint.get("stage")) == "Stage 3e"
                and str(checkpoint.get("model")) == model
                and int(checkpoint.get("seed", -1)) == int(seed)
                and "state_dict" in checkpoint
            )
            checkpoint_metadata_ok = checkpoint_metadata_ok and metadata_ok

            metric_row = seed_metrics[
                seed_metrics["seed"].astype(int).eq(int(seed))
                & seed_metrics["model"].astype(str).eq(model)
            ].iloc[0]
            parameter_row = parameter_report[
                parameter_report["seed"].astype(int).eq(int(seed))
                & parameter_report["model"].astype(str).eq(model)
            ].iloc[0]

            checkpoint_ap = float(
                checkpoint["validation_metrics"]["average_precision"]
            )
            csv_ap = float(metric_row["validation_average_precision"])
            metric_ok = bool(
                np.isclose(checkpoint_ap, csv_ap, rtol=1e-8, atol=1e-10)
            )
            checkpoint_validation_metrics_ok = (
                checkpoint_validation_metrics_ok and metric_ok
            )

            checkpoint_parameters = int(
                checkpoint["trainable_parameter_count"]
            )
            csv_parameters = int(parameter_row["trainable_parameters"])
            parameter_ok = checkpoint_parameters == csv_parameters
            checkpoint_parameter_counts_ok = (
                checkpoint_parameter_counts_ok and parameter_ok
            )

            checkpoint_rows.append(
                {
                    "model": model,
                    "seed": int(seed),
                    "checkpoint_relative_path": relative_path(path),
                    "sha256": sha256_file(path),
                    "size_bytes": int(path.stat().st_size),
                    "best_epoch": int(checkpoint["best_epoch"]),
                    "validation_average_precision": checkpoint_ap,
                    "trainable_parameters": checkpoint_parameters,
                }
            )

    add_check(
        audit_rows,
        "checkpoint_internal_model_seed_and_stage_metadata_match",
        checkpoint_metadata_ok,
        {"checked": len(checkpoint_rows)},
    )
    add_check(
        audit_rows,
        "checkpoint_validation_auprc_matches_seed_metric_csv",
        checkpoint_validation_metrics_ok,
        {"checked": len(checkpoint_rows)},
    )
    add_check(
        audit_rows,
        "checkpoint_parameter_counts_match_parameter_report",
        checkpoint_parameter_counts_ok,
        {"checked": len(checkpoint_rows)},
    )

    hgnn_manifest = load_json(HGNN_FREEZE_MANIFEST)
    hgnn_audit = pd.read_csv(HGNN_FREEZE_AUDIT)
    add_check(
        audit_rows,
        "existing_genuine_hgnn_freeze_audit_all_pass",
        len(hgnn_audit) > 0
        and hgnn_audit["status"].astype(str).eq("PASS").all(),
        hgnn_audit[["check_name", "status"]].to_dict("records"),
    )
    add_check(
        audit_rows,
        "existing_genuine_hgnn_freeze_matches_current_protocol_and_script",
        hgnn_manifest.get("protocol_sha256") == protocol_hash
        and hgnn_manifest.get("stage3b_script_sha256")
        == sha256_file(STAGE3B_SCRIPT)
        and [int(seed) for seed in hgnn_manifest.get("locked_seeds", [])]
        == seeds,
        {
            "hgnn_protocol": hgnn_manifest.get("protocol_sha256"),
            "current_protocol": protocol_hash,
            "hgnn_stage3b_script": hgnn_manifest.get(
                "stage3b_script_sha256"
            ),
            "current_stage3b_script": sha256_file(STAGE3B_SCRIPT),
        },
    )

    hgnn_checksum_table = pd.read_csv(HGNN_FREEZE_CHECKSUMS)
    hgnn_checkpoint_mismatches = []
    for _, row in hgnn_checksum_table.iterrows():
        path = BASE_DIR / str(row["checkpoint_relative_path"])
        if not path.exists():
            hgnn_checkpoint_mismatches.append(
                {"path": str(path), "reason": "missing"}
            )
        elif sha256_file(path) != str(row["sha256"]):
            hgnn_checkpoint_mismatches.append(
                {"path": str(path), "reason": "sha256 mismatch"}
            )
    add_check(
        audit_rows,
        "existing_genuine_hgnn_checkpoint_hashes_still_match",
        not hgnn_checkpoint_mismatches,
        hgnn_checkpoint_mismatches,
    )

    source_and_input_rows = [
        checksum_record("source_stage3b", STAGE3B_SCRIPT),
        checksum_record("source_stage3e", STAGE3E_SCRIPT),
        checksum_record("locked_protocol", s3b.PROTOCOL_FILE),
        checksum_record(
            "train_only_normalization",
            s3b.SELF_DIR / "train_only_normalization.json",
        ),
        checksum_record("existing_hgnn_freeze_manifest", HGNN_FREEZE_MANIFEST),
        checksum_record("existing_hgnn_evaluation_plan", HGNN_FREEZE_PLAN),
        checksum_record(
            "existing_hgnn_checkpoint_checksums",
            HGNN_FREEZE_CHECKSUMS,
        ),
        checksum_record("existing_hgnn_freeze_audit", HGNN_FREEZE_AUDIT),
        checksum_record("existing_hgnn_freeze_marker", HGNN_FREEZE_MARKER),
    ]
    for name in PACKAGE_NAMES:
        source_and_input_rows.append(
            checksum_record("locked_training_package", s3b.PACKAGE_DIR / name)
        )

    node_table = (
        package["metadata"][["node_index", "node"]]
        .drop_duplicates()
        .sort_values("node_index")
    )
    nodes = node_table["node"].astype(str).tolist()
    _, _, raw_paths = s3b.load_raw_temporal_tensor(nodes)
    for logical_name, path in sorted(raw_paths.items()):
        source_and_input_rows.append(
            checksum_record(f"raw_temporal_{logical_name}", path)
        )

    development_output_rows = [
        checksum_record("stage3e_development_output", STAGE3E_DIR / name)
        for name in STAGE3E_OUTPUT_NAMES
    ]

    add_check(
        audit_rows,
        "no_held_out_test_prediction_or_metric_calculated_by_stage3f",
        True,
        (
            "Stage 3f reads locked indices and development artifacts only; "
            "it does not construct test features or score test labels."
        ),
    )
    stop_if_failed(audit_rows, "Stage 3f final audit")

    full_hgnn_test_pre_existing = HGNN_TEST_SUMMARY.exists()
    if full_hgnn_test_pre_existing:
        interpretation_constraint = (
            "The full-HGNN and Self-GRU test results existed before this "
            "freeze. The new baseline checkpoints were selected using "
            "validation data only and had not produced test predictions. "
            "These are therefore exploratory extension comparisons, not an "
            "untouched confirmatory test."
        )
    else:
        interpretation_constraint = (
            "All candidate checkpoints were frozen before any test prediction "
            "recorded by this workflow. This procedural fact does not by "
            "itself establish that the period was never examined outside the "
            "workflow."
        )

    evaluation_plan = {
        "stage": "Stage 3g planned credible-baseline extension evaluation",
        "frozen_before_new_baseline_test_prediction": True,
        "training_or_tuning_allowed": False,
        "evaluation_split": "pre-existing held-out test split",
        "interpretation_constraint": interpretation_constraint,
        "ensemble": {
            "seeds": seeds,
            "aggregation": "mean of per-seed risk scores",
        },
        "candidate": FULL_HGNN_NAME,
        "reviewer_requested_baselines": MODEL_NAMES,
        "auprc_comparisons": [
            {
                "candidate": FULL_HGNN_NAME,
                "baseline": baseline,
                "metric": "average_precision",
                "reported_effect": "candidate minus baseline",
                "interpretation": "descriptive paired contrast",
            }
            for baseline in MODEL_NAMES
        ],
        "comparison_role": (
            "secondary reviewer-motivated extension comparisons; no new "
            "single primary endpoint is declared"
        ),
        "statistical_reporting": (
            "Report paired moving-block means, percentile intervals, and "
            "ordinary-bootstrap non-positive fractions descriptively. "
            "Do not label the fractions as p-values or apply FDR claims."
        ),
        "multiple_testing": (
            "none; all comparisons and ordinary-bootstrap summaries are descriptive"
        ),
        "secondary_descriptive_metrics": [
            "roc_auc",
            "precision_at_top10",
            "recall_at_top10",
            "lift_at_top10",
        ],
        "top_alert_rate": 0.10,
        "bootstrap": {
            "type": "paired moving-block bootstrap over dates",
            "block_length_dates": 7,
            "iterations": 5000,
            "random_state": 20260816,
            "confidence_interval": "2.5% to 97.5%",
        },
        "reporting_requirements": [
            "Report all three baseline comparisons regardless of direction.",
            "Report effect estimates and confidence intervals even when the "
            "HGNN is not superior.",
            "Keep validation results separate from held-out-test results.",
            "Describe the test split as pre-existing, not never-before-seen.",
        ],
        "no_post_freeze_changes": [
            "architecture",
            "checkpoint choice",
            "random seeds",
            "relation definitions",
            "parameter-matching rule",
            "evaluation metrics",
            "bootstrap block length",
            "bootstrap iterations",
            "descriptive statistical reporting",
        ],
    }

    created_utc = datetime.now(timezone.utc).isoformat()
    split_snapshot = {
        "train_count": int(len(train_indices)),
        "validation_count": int(len(validation_indices)),
        "test_count": int(len(test_indices)),
        "train_index_sha256": sha256_int64_array(train_indices),
        "validation_index_sha256": sha256_int64_array(validation_indices),
        "test_index_sha256": sha256_int64_array(test_indices),
    }

    checkpoint_df = pd.DataFrame(checkpoint_rows).sort_values(
        ["model", "seed"]
    )
    development_output_df = pd.DataFrame(development_output_rows)
    source_and_input_df = pd.DataFrame(source_and_input_rows)
    freeze_audit_df = pd.DataFrame(audit_rows)

    temporary_parent = Path(
        tempfile.mkdtemp(prefix=".stage3f_freeze_", dir=BASE_DIR)
    )
    try:
        plan_path = (
            temporary_parent / "credible_baseline_evaluation_plan.json"
        )
        checksum_path = (
            temporary_parent
            / "credible_baseline_checkpoint_checksums.csv"
        )
        development_output_path = (
            temporary_parent
            / "credible_baseline_development_output_checksums.csv"
        )
        input_checksum_path = (
            temporary_parent / "credible_baseline_input_checksums.csv"
        )
        audit_path = temporary_parent / "credible_baseline_freeze_audit.csv"
        manifest_path = (
            temporary_parent / "credible_baseline_freeze_manifest.json"
        )
        marker_path = temporary_parent / "FREEZE_COMPLETE.txt"

        save_json(evaluation_plan, plan_path)
        checkpoint_df.to_csv(
            checksum_path,
            index=False,
            encoding="utf-8-sig",
        )
        development_output_df.to_csv(
            development_output_path,
            index=False,
            encoding="utf-8-sig",
        )
        source_and_input_df.to_csv(
            input_checksum_path,
            index=False,
            encoding="utf-8-sig",
        )
        freeze_audit_df.to_csv(
            audit_path,
            index=False,
            encoding="utf-8-sig",
        )

        freeze_manifest = {
            "stage": "Stage 3f credible-baseline freeze",
            "created_utc": created_utc,
            "no_test_evaluation_performed": True,
            "protocol_sha256": protocol_hash,
            "stage3b_script_sha256": sha256_file(STAGE3B_SCRIPT),
            "stage3e_script_sha256": sha256_file(STAGE3E_SCRIPT),
            "locked_seeds": seeds,
            "models": MODEL_NAMES,
            "reference_candidate": FULL_HGNN_NAME,
            "checkpoint_count": int(len(checkpoint_df)),
            "split_snapshot": split_snapshot,
            "maximum_parameter_difference_vs_full_hgnn": (
                max_parameter_difference
            ),
            "validation_selection_metric": "average_precision",
            "validation_results_seen_before_freeze": True,
            "new_baseline_test_results_seen_before_freeze": False,
            "full_hgnn_test_results_pre_existing_before_freeze": (
                full_hgnn_test_pre_existing
            ),
            "evaluation_plan_sha256": sha256_file(plan_path),
            "checkpoint_checksum_table_sha256": sha256_file(checksum_path),
            "development_output_checksum_table_sha256": sha256_file(
                development_output_path
            ),
            "input_checksum_table_sha256": sha256_file(
                input_checksum_path
            ),
            "freeze_audit_sha256": sha256_file(audit_path),
            "existing_hgnn_freeze_manifest_sha256": sha256_file(
                HGNN_FREEZE_MANIFEST
            ),
        }
        save_json(freeze_manifest, manifest_path)

        marker_text = (
            "STAGE 3f FREEZE COMPLETE\n"
            f"created_utc={created_utc}\n"
            f"manifest_sha256={sha256_file(manifest_path)}\n"
            "held_out_test_evaluated_by_stage3f=false\n"
            "Do not edit Stage-3e code, checkpoints, inputs, or this freeze.\n"
        )
        with open(marker_path, "w", encoding="utf-8") as handle:
            handle.write(marker_text)

        os.replace(temporary_parent, FREEZE_DIR)
    finally:
        if temporary_parent.exists():
            try:
                temporary_parent.rmdir()
            except OSError:
                pass

    print("\n" + "=" * 72)
    print("STAGE 3f FREEZE AUDIT: ALL PASS")
    print("=" * 72)
    print(freeze_audit_df.to_string(index=False))
    print("\nFrozen checkpoints:", len(checkpoint_df))
    print("Freeze directory:", FREEZE_DIR)
    print(
        "\nIMPORTANT: Stage 3f did not calculate held-out-test predictions "
        "or performance. Preserve this directory unchanged."
    )


if __name__ == "__main__":
    main()
