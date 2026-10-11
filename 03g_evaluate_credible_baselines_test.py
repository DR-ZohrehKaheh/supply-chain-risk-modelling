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
import shutil
import tempfile

import numpy as np
import pandas as pd
import torch
import os
torch.set_num_threads(int(os.environ.get("PORTFOLIO_TORCH_THREADS", "1")))


# ============================================================
# 0. Paths and frozen names
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

STAGE3B_SCRIPT = BASE_DIR / "03b_train_genuine_hgnn_baselines.py"
STAGE3E_SCRIPT = BASE_DIR / "03e_train_credible_mlp_gnn_baselines.py"

STAGE3B_DIR = BASE_DIR / "hgnn_ablation_results"
STAGE3B_CHECKPOINT_DIR = STAGE3B_DIR / "checkpoints"

STAGE3E_DIR = BASE_DIR / "credible_baseline_results"
STAGE3E_CHECKPOINT_DIR = STAGE3E_DIR / "checkpoints"

HGNN_FREEZE_DIR = BASE_DIR / "hgnn_extension_frozen"
HGNN_FREEZE_MANIFEST = (
    HGNN_FREEZE_DIR / "hgnn_extension_freeze_manifest.json"
)
HGNN_FREEZE_CHECKSUMS = (
    HGNN_FREEZE_DIR / "hgnn_extension_checkpoint_checksums.csv"
)

FREEZE_DIR = BASE_DIR / "credible_baseline_frozen"
FREEZE_MANIFEST = FREEZE_DIR / "credible_baseline_freeze_manifest.json"
EVALUATION_PLAN = FREEZE_DIR / "credible_baseline_evaluation_plan.json"
CHECKPOINT_CHECKSUMS = (
    FREEZE_DIR / "credible_baseline_checkpoint_checksums.csv"
)
DEVELOPMENT_OUTPUT_CHECKSUMS = (
    FREEZE_DIR / "credible_baseline_development_output_checksums.csv"
)
INPUT_CHECKSUMS = FREEZE_DIR / "credible_baseline_input_checksums.csv"
FREEZE_AUDIT = FREEZE_DIR / "credible_baseline_freeze_audit.csv"
FREEZE_MARKER = FREEZE_DIR / "FREEZE_COMPLETE.txt"

OUT_DIR = BASE_DIR / "credible_baseline_test_results"

FULL_HGNN_NAME = "HGNN-Gated-Residual"
SELF_NAME = "Stage 2 Self-GRU"
BASELINE_NAMES = [
    "Temporal-MLP",
    "Clique-GCN-Gated-Residual",
    "Clique-GraphSAGE-Gated-Residual",
]
EXPECTED_HGNN_MODELS = [
    "HGNN-Gated-Residual",
    "HGNN-Residual-NoGate",
    "HGNN-Gated-NoResidual",
]

REQUIRED_FREEZE_FILES = [
    FREEZE_MANIFEST,
    EVALUATION_PLAN,
    CHECKPOINT_CHECKSUMS,
    DEVELOPMENT_OUTPUT_CHECKSUMS,
    INPUT_CHECKSUMS,
    FREEZE_AUDIT,
    FREEZE_MARKER,
]


# ============================================================
# 1. Utilities
# ============================================================

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_int64_array(values):
    array = np.ascontiguousarray(np.asarray(values, dtype=np.int64))
    return hashlib.sha256(array.tobytes()).hexdigest()


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(value, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
        raise RuntimeError(f"{context} failed; test evaluation stopped.")


def verify_checksum_rows(table, path_column="relative_path"):
    mismatches = []
    for _, row in table.iterrows():
        path = BASE_DIR / str(row[path_column])
        if not path.exists():
            mismatches.append(
                {"path": str(path), "reason": "missing"}
            )
            continue
        current = sha256_file(path)
        frozen = str(row["sha256"])
        if current != frozen:
            mismatches.append(
                {
                    "path": str(path),
                    "reason": "sha256 mismatch",
                    "current": current,
                    "frozen": frozen,
                }
            )
    return mismatches


def ensemble_mean(arrays):
    return np.mean(np.stack(arrays, axis=0), axis=0)


def stage3b_checkpoint_path(model_name, seed):
    return (
        STAGE3B_CHECKPOINT_DIR
        / model_name.replace("-", "_")
        / f"seed_{seed}.pt"
    )


def stage3e_checkpoint_path(model_name, seed):
    return (
        STAGE3E_CHECKPOINT_DIR
        / model_name.replace("-", "_")
        / f"seed_{seed}.pt"
    )


def model_prefix(model_name):
    return model_name.lower().replace("-", "_").replace(" ", "_")


# ============================================================
# 2. Verify both freezes before reading test labels/features
# ============================================================

def verify_frozen_inputs():
    audit_rows = []

    if OUT_DIR.exists():
        raise FileExistsError(
            "Refusing to overwrite an existing Stage-3g output directory: "
            f"{OUT_DIR}"
        )

    missing = [str(path) for path in REQUIRED_FREEZE_FILES if not path.exists()]
    add_check(
        audit_rows,
        "all_stage3f_freeze_files_exist",
        not missing,
        missing,
    )
    stop_if_failed(audit_rows, "Stage 3g freeze prerequisite audit")

    manifest = load_json(FREEZE_MANIFEST)
    plan = load_json(EVALUATION_PLAN)
    freeze_audit = pd.read_csv(FREEZE_AUDIT)
    checkpoint_table = pd.read_csv(CHECKPOINT_CHECKSUMS)
    development_output_table = pd.read_csv(DEVELOPMENT_OUTPUT_CHECKSUMS)
    input_table = pd.read_csv(INPUT_CHECKSUMS)

    internal_hash_mapping = {
        "evaluation_plan_sha256": EVALUATION_PLAN,
        "checkpoint_checksum_table_sha256": CHECKPOINT_CHECKSUMS,
        "development_output_checksum_table_sha256": (
            DEVELOPMENT_OUTPUT_CHECKSUMS
        ),
        "input_checksum_table_sha256": INPUT_CHECKSUMS,
        "freeze_audit_sha256": FREEZE_AUDIT,
    }
    internal_mismatches = []
    for manifest_key, path in internal_hash_mapping.items():
        current = sha256_file(path)
        frozen = str(manifest.get(manifest_key, ""))
        if current != frozen:
            internal_mismatches.append(
                {
                    "manifest_key": manifest_key,
                    "path": str(path),
                    "current": current,
                    "frozen": frozen,
                }
            )
    add_check(
        audit_rows,
        "stage3f_internal_file_hashes_match_manifest",
        not internal_mismatches,
        internal_mismatches,
    )

    marker_lines = FREEZE_MARKER.read_text(
        encoding="utf-8"
    ).splitlines()
    marker_values = {
        line.split("=", 1)[0]: line.split("=", 1)[1]
        for line in marker_lines
        if "=" in line
    }
    add_check(
        audit_rows,
        "stage3f_marker_matches_manifest_and_no_test_policy",
        marker_values.get("manifest_sha256") == sha256_file(FREEZE_MANIFEST)
        and marker_values.get("held_out_test_evaluated_by_stage3f")
        == "false",
        marker_values,
    )
    add_check(
        audit_rows,
        "stage3f_freeze_audit_all_pass",
        len(freeze_audit) > 0
        and freeze_audit["status"].astype(str).eq("PASS").all(),
        freeze_audit[["check_name", "status"]].to_dict("records"),
    )

    input_mismatches = verify_checksum_rows(input_table)
    add_check(
        audit_rows,
        "all_frozen_source_and_input_hashes_match",
        not input_mismatches,
        input_mismatches,
    )
    add_check(
        audit_rows,
        "stage3f_manifest_source_hashes_match_current_scripts",
        manifest.get("stage3b_script_sha256")
        == sha256_file(STAGE3B_SCRIPT)
        and manifest.get("stage3e_script_sha256")
        == sha256_file(STAGE3E_SCRIPT),
        {
            "stage3b_current": sha256_file(STAGE3B_SCRIPT),
            "stage3b_frozen": manifest.get("stage3b_script_sha256"),
            "stage3e_current": sha256_file(STAGE3E_SCRIPT),
            "stage3e_frozen": manifest.get("stage3e_script_sha256"),
        },
    )
    development_mismatches = verify_checksum_rows(development_output_table)
    add_check(
        audit_rows,
        "all_frozen_stage3e_development_output_hashes_match",
        not development_mismatches,
        development_mismatches,
    )

    seeds = [int(seed) for seed in manifest.get("locked_seeds", [])]
    expected_checkpoint_pairs = {
        (model, int(seed)) for model in BASELINE_NAMES for seed in seeds
    }
    observed_checkpoint_pairs = set(
        zip(
            checkpoint_table["model"].astype(str),
            checkpoint_table["seed"].astype(int),
        )
    )
    add_check(
        audit_rows,
        "stage3e_checkpoint_table_has_exact_models_and_seeds",
        len(checkpoint_table) == 30
        and int(manifest.get("checkpoint_count", -1)) == 30
        and manifest.get("models") == BASELINE_NAMES
        and observed_checkpoint_pairs == expected_checkpoint_pairs,
        {
            "rows": len(checkpoint_table),
            "models": manifest.get("models"),
            "seeds": seeds,
        },
    )
    checkpoint_mismatches = verify_checksum_rows(
        checkpoint_table,
        path_column="checkpoint_relative_path",
    )
    add_check(
        audit_rows,
        "all_thirty_stage3e_checkpoint_hashes_match",
        not checkpoint_mismatches,
        checkpoint_mismatches,
    )

    required_hgnn_freeze = [HGNN_FREEZE_MANIFEST, HGNN_FREEZE_CHECKSUMS]
    missing_hgnn_freeze = [
        str(path) for path in required_hgnn_freeze if not path.exists()
    ]
    add_check(
        audit_rows,
        "required_existing_hgnn_freeze_files_exist",
        not missing_hgnn_freeze,
        missing_hgnn_freeze,
    )
    stop_if_failed(audit_rows, "Stage 3g static freeze audit")

    hgnn_manifest = load_json(HGNN_FREEZE_MANIFEST)
    hgnn_checkpoint_table = pd.read_csv(HGNN_FREEZE_CHECKSUMS)
    hgnn_checkpoint_mismatches = verify_checksum_rows(
        hgnn_checkpoint_table,
        path_column="checkpoint_relative_path",
    )
    add_check(
        audit_rows,
        "all_existing_hgnn_checkpoint_hashes_match",
        not hgnn_checkpoint_mismatches,
        hgnn_checkpoint_mismatches,
    )
    freeze_manifest_hash_matches = (
        sha256_file(HGNN_FREEZE_MANIFEST)
        == manifest.get("existing_hgnn_freeze_manifest_sha256")
    )
    add_check(
        audit_rows,
        "stage3f_and_hgnn_freezes_share_protocol_seeds_and_candidate",
        hgnn_manifest.get("protocol_sha256")
        == manifest.get("protocol_sha256")
        and [int(seed) for seed in hgnn_manifest.get("locked_seeds", [])]
        == seeds
        and hgnn_manifest.get("models") == EXPECTED_HGNN_MODELS
        and hgnn_manifest.get("stage3b_script_sha256")
        == sha256_file(STAGE3B_SCRIPT)
        and freeze_manifest_hash_matches
        and manifest.get("reference_candidate") == FULL_HGNN_NAME,
        {
            "stage3f_protocol": manifest.get("protocol_sha256"),
            "hgnn_protocol": hgnn_manifest.get("protocol_sha256"),
            "stage3f_candidate": manifest.get("reference_candidate"),
            "hgnn_models": hgnn_manifest.get("models"),
            "hgnn_stage3b_script": hgnn_manifest.get(
                "stage3b_script_sha256"
            ),
            "hgnn_freeze_manifest_hash_matches_stage3f": (
                freeze_manifest_hash_matches
            ),
        },
    )

    comparisons = plan.get("auprc_comparisons", [])
    plan_baselines = [str(item.get("baseline")) for item in comparisons]
    add_check(
        audit_rows,
        "evaluation_plan_is_locked_for_three_expected_comparisons",
        plan.get("frozen_before_new_baseline_test_prediction") is True
        and plan.get("training_or_tuning_allowed") is False
        and plan.get("candidate") == FULL_HGNN_NAME
        and plan.get("reviewer_requested_baselines") == BASELINE_NAMES
        and plan_baselines == BASELINE_NAMES
        and [int(seed) for seed in plan["ensemble"]["seeds"]] == seeds,
        {
            "candidate": plan.get("candidate"),
            "baselines": plan_baselines,
            "seeds": plan.get("ensemble", {}).get("seeds"),
        },
    )
    add_check(
        audit_rows,
        "freeze_records_new_baseline_test_results_unseen",
        manifest.get("no_test_evaluation_performed") is True
        and manifest.get("new_baseline_test_results_seen_before_freeze")
        is False
        and isinstance(
            manifest.get(
                "full_hgnn_test_results_pre_existing_before_freeze"
            ),
            bool,
        ),
        {
            "new_baseline_test_results_seen": manifest.get(
                "new_baseline_test_results_seen_before_freeze"
            ),
            "full_hgnn_test_results_pre_existing": manifest.get(
                "full_hgnn_test_results_pre_existing_before_freeze"
            ),
        },
    )

    stop_if_failed(audit_rows, "Stage 3g complete pre-test freeze audit")
    return (
        manifest,
        plan,
        checkpoint_table,
        hgnn_manifest,
        audit_rows,
    )


# ============================================================
# 3. Frozen model construction from checkpoint metadata
# ============================================================

def frozen_self_shell(checkpoint, s3b, device):
    config = checkpoint.get("self_checkpoint_config", {})
    hidden_dim = int(checkpoint["hidden_dim"])
    dropout = float(config.get("dropout", 0.0))

    model = s3b.SelfGRUHazardModel(
        input_dim=s3b.INPUT_DIM,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def build_frozen_hgnn(checkpoint, s3b, device):
    self_model = frozen_self_shell(checkpoint, s3b, device)
    model = s3b.build_model(
        FULL_HGNN_NAME,
        self_model,
        int(checkpoint["hidden_dim"]),
        s3b.RELATION_DROPOUT,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model


def build_frozen_credible_baseline(model_name, checkpoint, s3b, s3e, device):
    hidden_dim = int(checkpoint["hidden_dim"])

    if model_name == "Temporal-MLP":
        model = s3e.TemporalMLPHazardModel(
            lookback_days=s3b.LOOKBACK_DAYS,
            input_dim=s3b.INPUT_DIM,
            width=int(checkpoint["mlp_width"]),
            dropout=s3b.RELATION_DROPOUT,
            hazard_days=s3b.HAZARD_DAYS,
        ).to(device)
    else:
        self_model = frozen_self_shell(checkpoint, s3b, device)
        if model_name == "Clique-GCN-Gated-Residual":
            relation_input_dim = s3b.INPUT_DIM
        elif model_name == "Clique-GraphSAGE-Gated-Residual":
            relation_input_dim = s3b.INPUT_DIM
        else:
            raise ValueError(f"Unexpected credible baseline: {model_name}")

        model = s3e.PairwiseGatedResidualModel(
            frozen_self_model=self_model,
            relation_input_dim=relation_input_dim,
            hidden_dim=hidden_dim,
            dropout=s3b.RELATION_DROPOUT,
            input_dim=s3b.INPUT_DIM,
            hazard_days=s3b.HAZARD_DAYS,
            initial_gate_bias=s3b.INITIAL_GATE_BIAS,
        ).to(device)

    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model


# ============================================================
# 4. Frozen test evaluation
# ============================================================

def main():
    print("=" * 72)
    print("STAGE 3g: FROZEN CREDIBLE-BASELINE TEST EVALUATION")
    print("NO TRAINING, TUNING, OR CHECKPOINT SELECTION")
    print("=" * 72)

    (
        freeze_manifest,
        plan,
        frozen_checkpoint_table,
        hgnn_manifest,
        audit_rows,
    ) = verify_frozen_inputs()

    print("\nPre-test freeze verification: ALL PASS")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cpu":
        torch.set_num_threads(int(os.environ.get("PORTFOLIO_TORCH_THREADS", "1")))

    s3b = load_module("stage3b_hgnn_for_stage3g", STAGE3B_SCRIPT)
    s3e = load_module("stage3e_baselines_for_stage3g", STAGE3E_SCRIPT)

    protocol, protocol_hash = s3b.load_locked_protocol()
    package = s3b.load_package()
    normalizer = s3b.load_stage2_normalization()
    seeds = [int(seed) for seed in protocol["random_seeds"]]

    train_indices = package["train_indices"].astype(np.int64)
    validation_indices = package["validation_indices"].astype(np.int64)
    test_indices = package["test_indices"].astype(np.int64)
    split_snapshot = freeze_manifest["split_snapshot"]

    add_check(
        audit_rows,
        "current_protocol_and_package_match_stage3f_freeze",
        protocol_hash == freeze_manifest["protocol_sha256"]
        == package["package_protocol_hash"],
        {
            "current_protocol": protocol_hash,
            "frozen_protocol": freeze_manifest["protocol_sha256"],
            "package_protocol": package["package_protocol_hash"],
        },
    )
    add_check(
        audit_rows,
        "current_locked_seeds_match_stage3f_freeze",
        seeds == [int(seed) for seed in freeze_manifest["locked_seeds"]],
        seeds,
    )
    split_hashes_match = (
        len(train_indices) == int(split_snapshot["train_count"])
        and len(validation_indices) == int(split_snapshot["validation_count"])
        and len(test_indices) == int(split_snapshot["test_count"])
        and sha256_int64_array(train_indices)
        == split_snapshot["train_index_sha256"]
        and sha256_int64_array(validation_indices)
        == split_snapshot["validation_index_sha256"]
        and sha256_int64_array(test_indices)
        == split_snapshot["test_index_sha256"]
    )
    add_check(
        audit_rows,
        "current_split_counts_and_index_hashes_match_freeze",
        split_hashes_match,
        {
            "train": len(train_indices),
            "validation": len(validation_indices),
            "test": len(test_indices),
        },
    )
    stop_if_failed(audit_rows, "Stage 3g locked-package audit")

    test_metadata = (
        package["metadata"].iloc[test_indices].copy().reset_index(drop=True)
    )
    y_test = package["y_3_day"][test_indices].astype(int)

    print("\nTest samples:", len(test_indices))
    print("Test positives:", int(y_test.sum()))
    print("Test prevalence:", float(y_test.mean()))
    print(
        "Test dates:",
        pd.to_datetime(test_metadata["date"]).min(),
        "to",
        pd.to_datetime(test_metadata["date"]).max(),
    )

    node_table = (
        package["metadata"][["node_index", "node"]]
        .drop_duplicates()
        .sort_values("node_index")
    )
    nodes = node_table["node"].astype(str).tolist()
    raw_tensor, _, _ = s3b.load_raw_temporal_tensor(nodes)
    raw_normalized = s3b.transform_all_node_tensor(raw_tensor, normalizer)
    X_self_all = s3b.transform_sample_tensor(package["X_self"], normalizer)
    X_self_test = X_self_all[test_indices]

    H_group = s3b.validate_incidence_matrix(
        package["H_product_group"],
        "Product Group",
    )
    H_storage = s3b.validate_incidence_matrix(
        package["H_storage_location"],
        "Storage Location",
    )

    # Genuine incidence-HGNN relation sequences.
    hgnn_group_tensor = s3b.genuine_node_hyperedge_node_propagation(
        raw_normalized,
        H_group,
    )
    hgnn_storage_tensor = s3b.genuine_node_hyperedge_node_propagation(
        raw_normalized,
        H_storage,
    )
    X_hgnn_group_test = s3b.build_target_relation_sequences(
        package["metadata"],
        hgnn_group_tensor,
        test_indices,
    )
    X_hgnn_storage_test = s3b.build_target_relation_sequences(
        package["metadata"],
        hgnn_storage_tensor,
        test_indices,
    )

    # Pairwise clique-GCN relation sequences.
    gcn_group_tensor = s3e.propagate_node_tensor(
        raw_normalized,
        s3e.symmetric_gcn_operator(H_group),
    )
    gcn_storage_tensor = s3e.propagate_node_tensor(
        raw_normalized,
        s3e.symmetric_gcn_operator(H_storage),
    )
    X_gcn_group_test = s3e.build_target_sequences(
        package["metadata"],
        gcn_group_tensor,
        test_indices,
        s3b.LOOKBACK_DAYS,
    )
    X_gcn_storage_test = s3e.build_target_sequences(
        package["metadata"],
        gcn_storage_tensor,
        test_indices,
        s3b.LOOKBACK_DAYS,
    )

    # Pairwise mean-GraphSAGE sequences: neighbour context only.  The frozen
    # Self-GRU remains the sole path for the focal product's own history.
    sage_group_neighbours = s3e.propagate_node_tensor(
        raw_normalized,
        s3e.row_mean_neighbor_operator(H_group),
    )
    sage_storage_neighbours = s3e.propagate_node_tensor(
        raw_normalized,
        s3e.row_mean_neighbor_operator(H_storage),
    )
    sage_group_tensor = sage_group_neighbours
    sage_storage_tensor = sage_storage_neighbours
    X_sage_group_test = s3e.build_target_sequences(
        package["metadata"],
        sage_group_tensor,
        test_indices,
        s3b.LOOKBACK_DAYS,
    )
    X_sage_storage_test = s3e.build_target_sequences(
        package["metadata"],
        sage_storage_tensor,
        test_indices,
        s3b.LOOKBACK_DAYS,
    )

    expected_self_shape = (
        len(test_indices),
        s3b.LOOKBACK_DAYS,
        s3b.INPUT_DIM,
    )
    expected_sage_shape = (
        len(test_indices),
        s3b.LOOKBACK_DAYS,
        s3b.INPUT_DIM,
    )
    shape_check = (
        X_self_test.shape == expected_self_shape
        and X_hgnn_group_test.shape == expected_self_shape
        and X_hgnn_storage_test.shape == expected_self_shape
        and X_gcn_group_test.shape == expected_self_shape
        and X_gcn_storage_test.shape == expected_self_shape
        and X_sage_group_test.shape == expected_sage_shape
        and X_sage_storage_test.shape == expected_sage_shape
    )
    add_check(
        audit_rows,
        "all_test_feature_shapes_match_frozen_architectures",
        shape_check,
        {
            "self": X_self_test.shape,
            "hgnn_group": X_hgnn_group_test.shape,
            "gcn_group": X_gcn_group_test.shape,
            "sage_group": X_sage_group_test.shape,
        },
    )
    stop_if_failed(audit_rows, "Stage 3g test-feature audit")

    baseline_inputs = {
        "Temporal-MLP": (
            X_self_test,
            X_self_test,
            X_self_test,
        ),
        "Clique-GCN-Gated-Residual": (
            X_self_test,
            X_gcn_group_test,
            X_gcn_storage_test,
        ),
        "Clique-GraphSAGE-Gated-Residual": (
            X_self_test,
            X_sage_group_test,
            X_sage_storage_test,
        ),
    }

    risk_by_model = {FULL_HGNN_NAME: []}
    hazards_by_model = {FULL_HGNN_NAME: []}
    gate_group_by_model = {FULL_HGNN_NAME: []}
    gate_storage_by_model = {FULL_HGNN_NAME: []}
    for name in BASELINE_NAMES:
        risk_by_model[name] = []
        hazards_by_model[name] = []
        gate_group_by_model[name] = []
        gate_storage_by_model[name] = []

    self_risk_by_seed = []
    seed_metric_rows = []
    base_consistency_rows = []
    checkpoint_metadata_ok = True
    parameter_counts_ok = True

    for seed in seeds:
        print("\n" + "=" * 72)
        print("SEED:", seed)
        print("=" * 72)

        hgnn_path = stage3b_checkpoint_path(FULL_HGNN_NAME, seed)
        hgnn_checkpoint = s3b.safe_torch_load(hgnn_path, device)
        hgnn_metadata_ok = (
            str(hgnn_checkpoint.get("stage")) == "Stage 3b"
            and str(hgnn_checkpoint.get("model")) == FULL_HGNN_NAME
            and int(hgnn_checkpoint.get("seed", -1)) == int(seed)
        )
        checkpoint_metadata_ok = checkpoint_metadata_ok and hgnn_metadata_ok

        hgnn_model = build_frozen_hgnn(hgnn_checkpoint, s3b, device)
        hgnn_parameter_count = s3b.count_trainable_parameters(hgnn_model)
        parameter_counts_ok = parameter_counts_ok and (
            hgnn_parameter_count
            == int(hgnn_checkpoint["trainable_parameter_count"])
        )
        hgnn_prediction = s3b.predict_model(
            hgnn_model,
            X_self_test,
            X_hgnn_group_test,
            X_hgnn_storage_test,
            device,
        )
        risk_by_model[FULL_HGNN_NAME].append(hgnn_prediction["risk"])
        hazards_by_model[FULL_HGNN_NAME].append(
            hgnn_prediction["hazards"]
        )
        gate_group_by_model[FULL_HGNN_NAME].append(
            hgnn_prediction["gate_group"]
        )
        gate_storage_by_model[FULL_HGNN_NAME].append(
            hgnn_prediction["gate_storage"]
        )
        self_risk_by_seed.append(hgnn_prediction["base_risk"])

        self_metrics = s3b.evaluate_risk_predictions(
            y_test,
            hgnn_prediction["base_risk"],
        )
        hgnn_metrics = s3b.evaluate_risk_predictions(
            y_test,
            hgnn_prediction["risk"],
        )
        seed_metric_rows.append(
            {"seed": seed, "model": SELF_NAME, **self_metrics}
        )
        seed_metric_rows.append(
            {"seed": seed, "model": FULL_HGNN_NAME, **hgnn_metrics}
        )
        print(
            f"{FULL_HGNN_NAME}: AP={hgnn_metrics['average_precision']:.6f}, "
            f"ROC-AUC={hgnn_metrics['roc_auc']:.6f}"
        )

        seed_base_differences = {}

        for model_name in BASELINE_NAMES:
            path = stage3e_checkpoint_path(model_name, seed)
            checkpoint = s3b.safe_torch_load(path, device)
            metadata_ok = (
                str(checkpoint.get("stage")) == "Stage 3e"
                and str(checkpoint.get("model")) == model_name
                and int(checkpoint.get("seed", -1)) == int(seed)
            )
            checkpoint_metadata_ok = checkpoint_metadata_ok and metadata_ok

            model = build_frozen_credible_baseline(
                model_name,
                checkpoint,
                s3b,
                s3e,
                device,
            )
            parameter_count = s3b.count_trainable_parameters(model)
            parameter_counts_ok = parameter_counts_ok and (
                parameter_count
                == int(checkpoint["trainable_parameter_count"])
            )

            X_self, X_group, X_storage = baseline_inputs[model_name]
            prediction = s3b.predict_model(
                model,
                X_self,
                X_group,
                X_storage,
                device,
            )
            risk_by_model[model_name].append(prediction["risk"])
            hazards_by_model[model_name].append(prediction["hazards"])
            gate_group_by_model[model_name].append(
                prediction["gate_group"]
            )
            gate_storage_by_model[model_name].append(
                prediction["gate_storage"]
            )

            metrics = s3b.evaluate_risk_predictions(
                y_test,
                prediction["risk"],
            )
            seed_metric_rows.append(
                {"seed": seed, "model": model_name, **metrics}
            )
            print(
                f"{model_name}: AP={metrics['average_precision']:.6f}, "
                f"ROC-AUC={metrics['roc_auc']:.6f}"
            )

            if model_name != "Temporal-MLP":
                seed_base_differences[model_name] = float(
                    np.max(
                        np.abs(
                            prediction["base_risk"]
                            - hgnn_prediction["base_risk"]
                        )
                    )
                )

        base_consistency_rows.append(
            {
                "seed": int(seed),
                "max_abs_diff_hgnn_vs_gcn_self_base": (
                    seed_base_differences[
                        "Clique-GCN-Gated-Residual"
                    ]
                ),
                "max_abs_diff_hgnn_vs_graphsage_self_base": (
                    seed_base_differences[
                        "Clique-GraphSAGE-Gated-Residual"
                    ]
                ),
            }
        )

        del hgnn_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    add_check(
        audit_rows,
        "all_checkpoint_internal_stage_model_seed_metadata_match",
        checkpoint_metadata_ok,
        {"hgnn": len(seeds), "credible_baselines": len(seeds) * 3},
    )
    add_check(
        audit_rows,
        "all_reconstructed_trainable_parameter_counts_match_checkpoints",
        parameter_counts_ok,
        {"checked_models": len(seeds) * 4},
    )
    expected_seed_metric_rows = len(seeds) * (2 + len(BASELINE_NAMES))
    add_check(
        audit_rows,
        "all_five_models_evaluated_for_all_ten_seeds",
        len(seed_metric_rows) == expected_seed_metric_rows,
        {
            "actual_rows": len(seed_metric_rows),
            "expected_rows": expected_seed_metric_rows,
        },
    )

    base_consistency_df = pd.DataFrame(base_consistency_rows)
    maximum_base_difference = float(
        base_consistency_df.drop(columns=["seed"]).to_numpy().max()
    )
    add_check(
        audit_rows,
        "frozen_self_gru_base_matches_across_hgnn_gcn_and_graphsage",
        maximum_base_difference <= 1e-7,
        {"maximum_absolute_difference": maximum_base_difference},
    )
    stop_if_failed(audit_rows, "Stage 3g model-evaluation audit")

    ensemble_risk = {
        name: ensemble_mean(values)
        for name, values in risk_by_model.items()
    }
    ensemble_hazards = {
        name: ensemble_mean(values)
        for name, values in hazards_by_model.items()
    }
    ensemble_gate_group = {
        name: ensemble_mean(values)
        for name, values in gate_group_by_model.items()
    }
    ensemble_gate_storage = {
        name: ensemble_mean(values)
        for name, values in gate_storage_by_model.items()
    }
    self_ensemble_risk = ensemble_mean(self_risk_by_seed)

    comparison_rows = [
        {
            "model": SELF_NAME + " ensemble",
            **s3b.evaluate_risk_predictions(y_test, self_ensemble_risk),
        },
        {
            "model": FULL_HGNN_NAME + " ensemble",
            **s3b.evaluate_risk_predictions(
                y_test,
                ensemble_risk[FULL_HGNN_NAME],
            ),
        },
    ]
    for name in BASELINE_NAMES:
        comparison_rows.append(
            {
                "model": name + " ensemble",
                **s3b.evaluate_risk_predictions(
                    y_test,
                    ensemble_risk[name],
                ),
            }
        )
    comparison_df = pd.DataFrame(comparison_rows)

    # Optional consistency check against the already reported Stage-3d HGNN
    # ensemble. The current frozen checkpoints remain authoritative.
    prior_hgnn_table_path = (
        BASE_DIR
        / "hgnn_extension_test_results"
        / "hgnn_test_model_comparison.csv"
    )
    prior_hgnn_consistency = "not available"
    prior_hgnn_ok = True
    if prior_hgnn_table_path.exists():
        prior_table = pd.read_csv(prior_hgnn_table_path)
        prior_row = prior_table[
            prior_table["model"].astype(str)
            .eq(FULL_HGNN_NAME + " ensemble")
        ]
        current_row = comparison_df[
            comparison_df["model"].astype(str)
            .eq(FULL_HGNN_NAME + " ensemble")
        ].iloc[0]
        if len(prior_row) == 1:
            metric_columns = [
                "average_precision",
                "roc_auc",
                "precision_at_top10",
                "recall_at_top10",
                "lift_at_top10",
            ]
            differences = {
                column: abs(
                    float(prior_row.iloc[0][column])
                    - float(current_row[column])
                )
                for column in metric_columns
            }
            # CPU/GPU kernels can differ at tiny floating-point scales even
            # with identical frozen weights.  This tolerance is far below any
            # reported precision while avoiding a device-only false failure.
            prior_hgnn_ok = max(differences.values()) <= 1e-7
            prior_hgnn_consistency = differences
        else:
            prior_hgnn_ok = False
            prior_hgnn_consistency = "expected exactly one full-HGNN row"
    add_check(
        audit_rows,
        "current_full_hgnn_matches_prior_stage3d_table_when_available",
        prior_hgnn_ok,
        prior_hgnn_consistency,
    )

    bootstrap_cfg = plan["bootstrap"]
    block_length = int(bootstrap_cfg["block_length_dates"])
    iterations = int(bootstrap_cfg["iterations"])
    random_state = int(bootstrap_cfg["random_state"])

    bootstrap_rows = []
    full_ap = float(
        comparison_df.loc[
            comparison_df["model"].eq(FULL_HGNN_NAME + " ensemble"),
            "average_precision",
        ].iloc[0]
    )
    for offset, comparison in enumerate(plan["auprc_comparisons"]):
        baseline = str(comparison["baseline"])
        baseline_ap = float(
            comparison_df.loc[
                comparison_df["model"].eq(baseline + " ensemble"),
                "average_precision",
            ].iloc[0]
        )
        result = s3b.moving_block_bootstrap_improvement(
            test_metadata,
            y_test,
            ensemble_risk[baseline],
            ensemble_risk[FULL_HGNN_NAME],
            s3b.safe_average_precision,
            block_length=block_length,
            n_bootstrap=iterations,
            random_state=random_state + offset,
        )
        bootstrap_rows.append(
            {
                "role": "secondary_reviewer_requested_baseline",
                "candidate": FULL_HGNN_NAME,
                "baseline": baseline,
                "metric": "average_precision",
                "observed_candidate_metric": full_ap,
                "observed_baseline_metric": baseline_ap,
                "observed_improvement": full_ap - baseline_ap,
                "block_length_dates": block_length,
                "bootstrap_iterations_requested": iterations,
                **result,
            }
        )

    bootstrap_df = pd.DataFrame(bootstrap_rows)
    bootstrap_df["descriptive_positive_interval_flag"] = (
        bootstrap_df["mean_improvement"].gt(0)
        & bootstrap_df["ci_lower_95"].gt(0)
    )

    add_check(
        audit_rows,
        "all_three_pre_specified_bootstraps_completed",
        len(bootstrap_df) == 3
        and bootstrap_df["bootstrap_valid_iterations"].eq(iterations).all(),
        bootstrap_df[
            ["baseline", "bootstrap_valid_iterations"]
        ].to_dict("records"),
    )
    add_check(
        audit_rows,
        "no_training_tuning_or_checkpoint_selection_performed",
        True,
        (
            "Stage 3g only loaded hashed validation-selected checkpoints and "
            "executed the frozen evaluation plan."
        ),
    )
    stop_if_failed(audit_rows, "Stage 3g statistical-evaluation audit")

    prediction_df = test_metadata.copy()
    for column in prediction_df.columns:
        if np.issubdtype(prediction_df[column].dtype, np.datetime64):
            prediction_df[column] = pd.to_datetime(
                prediction_df[column]
            ).astype(str)
    prediction_df.insert(0, "package_sample_index", test_indices)
    prediction_df["y_3_day"] = y_test
    prediction_df["self_gru_ensemble_risk"] = self_ensemble_risk

    ordered_models = [FULL_HGNN_NAME, *BASELINE_NAMES]
    for name in ordered_models:
        prefix = model_prefix(name)
        prediction_df[f"{prefix}_risk"] = ensemble_risk[name]
        for day in range(s3b.HAZARD_DAYS):
            prediction_df[f"{prefix}_hazard_day{day + 1}"] = (
                ensemble_hazards[name][:, day]
            )
        prediction_df[f"{prefix}_gate_group"] = (
            ensemble_gate_group[name]
        )
        prediction_df[f"{prefix}_gate_storage"] = (
            ensemble_gate_storage[name]
        )

    seed_metrics_df = pd.DataFrame(seed_metric_rows)
    final_audit_df = pd.DataFrame(audit_rows)

    created_utc = datetime.now(timezone.utc).isoformat()
    temporary_output = Path(
        tempfile.mkdtemp(prefix=".stage3g_results_", dir=BASE_DIR)
    )
    try:
        prediction_dir = temporary_output / "predictions"
        prediction_dir.mkdir(parents=True, exist_ok=True)

        comparison_path = (
            temporary_output / "credible_baseline_test_model_comparison.csv"
        )
        seed_metrics_path = (
            temporary_output / "credible_baseline_test_seed_metrics.csv"
        )
        bootstrap_path = (
            temporary_output
            / "credible_baseline_test_moving_block_bootstrap.csv"
        )
        base_consistency_path = (
            temporary_output / "credible_baseline_test_base_consistency.csv"
        )
        prediction_path = (
            prediction_dir
            / "test_credible_baseline_ensemble_predictions.csv"
        )
        audit_path = temporary_output / "stage3g_test_audit_report.csv"
        manifest_path = temporary_output / "stage3g_execution_manifest.json"
        marker_path = temporary_output / "EVALUATION_COMPLETE.txt"

        comparison_df.to_csv(
            comparison_path,
            index=False,
            encoding="utf-8-sig",
        )
        seed_metrics_df.to_csv(
            seed_metrics_path,
            index=False,
            encoding="utf-8-sig",
        )
        bootstrap_df.to_csv(
            bootstrap_path,
            index=False,
            encoding="utf-8-sig",
        )
        base_consistency_df.to_csv(
            base_consistency_path,
            index=False,
            encoding="utf-8-sig",
        )
        prediction_df.to_csv(
            prediction_path,
            index=False,
            encoding="utf-8-sig",
        )
        final_audit_df.to_csv(
            audit_path,
            index=False,
            encoding="utf-8-sig",
        )

        # Paths are still temporary; store stable output-relative names.
        output_hashes = {
            path.relative_to(temporary_output).as_posix(): sha256_file(path)
            for path in [
                comparison_path,
                seed_metrics_path,
                bootstrap_path,
                base_consistency_path,
                prediction_path,
                audit_path,
            ]
        }

        execution_manifest = {
            "stage": "Stage 3g frozen credible-baseline test evaluation",
            "created_utc": created_utc,
            "training_or_tuning_performed": False,
            "evaluation_split": "pre-existing held-out test split",
            "interpretation_constraint": plan["interpretation_constraint"],
            "stage3g_script_sha256": sha256_file(Path(__file__)),
            "stage3f_freeze_manifest_sha256": sha256_file(FREEZE_MANIFEST),
            "existing_hgnn_freeze_manifest_sha256": sha256_file(
                HGNN_FREEZE_MANIFEST
            ),
            "protocol_sha256": protocol_hash,
            "locked_seeds": seeds,
            "candidate": FULL_HGNN_NAME,
            "baselines": BASELINE_NAMES,
            "test_count": int(len(test_indices)),
            "test_positive_count": int(y_test.sum()),
            "bootstrap": bootstrap_cfg,
            "multiple_testing": plan["multiple_testing"],
            "output_sha256": output_hashes,
        }
        save_json(execution_manifest, manifest_path)

        marker_text = (
            "STAGE 3g EVALUATION COMPLETE\n"
            f"created_utc={created_utc}\n"
            f"manifest_sha256={sha256_file(manifest_path)}\n"
            "training_or_tuning_performed=false\n"
            "evaluation_split=pre-existing held-out test split\n"
        )
        with open(marker_path, "w", encoding="utf-8") as handle:
            handle.write(marker_text)

        os.replace(temporary_output, OUT_DIR)
    finally:
        if temporary_output.exists():
            shutil.rmtree(temporary_output)

    print("\n" + "=" * 72)
    print("HELD-OUT TEST MODEL COMPARISON")
    print("=" * 72)
    print(comparison_df.to_string(index=False))

    print("\n" + "=" * 72)
    print("PRE-SPECIFIED PAIRED MOVING-BLOCK BOOTSTRAP")
    print("=" * 72)
    print(bootstrap_df.to_string(index=False))

    print("\n" + "=" * 72)
    print("STAGE 3g AUDIT: ALL PASS")
    print("=" * 72)
    print(final_audit_df[["check_name", "status"]].to_string(index=False))
    print("\nResults saved to:", OUT_DIR)
    print(
        "\nInterpretation constraint: these are frozen extension comparisons "
        "on a pre-existing held-out split; report every comparison regardless "
        "of direction."
    )


if __name__ == "__main__":
    main()
