# Project supervisor: Dr. Zohreh Kaheh.
# Undergraduate project collaborator: Yuyang Zhang.
# Initial student coding was developed with supervisory teaching and improvement guidance.
# -*- coding: utf-8 -*-

from pathlib import Path
from datetime import datetime
import hashlib
import importlib.util
import json

import pandas as pd
import torch
import os
torch.set_num_threads(int(os.environ.get("PORTFOLIO_TORCH_THREADS", "1")))

BASE_DIR = Path(__file__).resolve().parent

STAGE3B_SCRIPT = BASE_DIR / "03b_train_genuine_hgnn_baselines.py"
STAGE3B_DIR = BASE_DIR / "hgnn_ablation_results"
CHECKPOINT_DIR = STAGE3B_DIR / "checkpoints"

FREEZE_DIR = BASE_DIR / "hgnn_extension_frozen"
FREEZE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAMES = [
    "HGNN-Gated-Residual",
    "HGNN-Residual-NoGate",
    "HGNN-Gated-NoResidual",
]

PRIMARY_MODEL = "HGNN-Gated-Residual"
PRIMARY_BASELINE = "Stage 2 Self-GRU"
PARAMETER_MATCH_TOLERANCE = 0.05

# Frozen Stage-3d inference settings
BOOTSTRAP_BLOCK_LENGTH_DATES = 7
BOOTSTRAP_ITERATIONS = 5000
BOOTSTRAP_RANDOM_STATE = 20260810
PRIMARY_METRIC = "average_precision"
TOP_ALERT_RATE = 0.10


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def safe_torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_stage3b():
    if not STAGE3B_SCRIPT.exists():
        raise FileNotFoundError(STAGE3B_SCRIPT)
    spec = importlib.util.spec_from_file_location("stage3b_hgnn", STAGE3B_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checkpoint_path(model_name, seed):
    return (
        CHECKPOINT_DIR
        / model_name.replace("-", "_")
        / f"seed_{seed}.pt"
    )


def audit_check(rows, name, ok, details):
    rows.append({
        "check_name": name,
        "status": "PASS" if ok else "FAIL",
        "details": str(details),
    })
    if not ok:
        raise RuntimeError(f"Stage 3c audit failed: {name}\n{details}")


def main():
    print("=" * 68)
    print("STAGE 3c: FREEZE GENUINE-HGNN EXTENSION")
    print("=" * 68)

    s3b = load_stage3b()
    protocol, protocol_hash = s3b.load_locked_protocol()
    package = s3b.load_package()
    seeds = [int(x) for x in protocol["random_seeds"]]

    audit = []

    required = [
        STAGE3B_DIR / "stage3b_design_manifest.json",
        STAGE3B_DIR / "stage3b_audit_report.csv",
        STAGE3B_DIR / "hgnn_validation_model_comparison.csv",
        STAGE3B_DIR / "hgnn_seed_metrics_validation.csv",
        STAGE3B_DIR / "hgnn_parameter_match_summary.csv",
        STAGE3B_DIR / "hgnn_validation_moving_block_bootstrap.csv",
    ]
    missing = [str(p) for p in required if not p.exists()]
    audit_check(audit, "all_required_stage3b_outputs_exist", not missing, missing)

    audit_check(
        audit,
        "protocol_matches_stage1_package",
        protocol_hash == package["package_protocol_hash"],
        {
            "protocol": protocol_hash,
            "package": package["package_protocol_hash"],
        },
    )

    stage3b_audit = pd.read_csv(STAGE3B_DIR / "stage3b_audit_report.csv")
    audit_check(
        audit,
        "stage3b_audit_all_pass",
        len(stage3b_audit) > 0
        and stage3b_audit["status"].astype(str).eq("PASS").all(),
        stage3b_audit[["check_name", "status"]].to_dict("records"),
    )

    with open(
        STAGE3B_DIR / "stage3b_design_manifest.json",
        "r",
        encoding="utf-8",
    ) as f:
        design = json.load(f)

    audit_check(
        audit,
        "stage3b_created_without_test_evaluation",
        bool(design.get("created_without_locked_test_evaluation", False)),
        design.get("locked_test_policy"),
    )

    audit_check(
        audit,
        "stage3b_models_match_freeze_plan",
        list(design.get("models", [])) == MODEL_NAMES,
        design.get("models"),
    )

    audit_check(
        audit,
        "focal_excluded_incidence_message_passing_frozen",
        design.get("message_passing", {}).get("type")
        == "focal-excluded incidence node-hyperedge-node"
        and design.get("message_passing", {}).get("self_feature_exclusion")
        == "exact zero diagonal",
        design.get("message_passing"),
    )

    param_df = pd.read_csv(STAGE3B_DIR / "hgnn_parameter_match_summary.csv")
    max_rel_diff = float(param_df["relative_difference_vs_full"].max())
    match_flags = (
        param_df["within_5pct_parameter_match"]
        .astype(str)
        .str.lower()
        .isin(["true", "1"])
    )
    audit_check(
        audit,
        "all_parameter_matches_within_5pct",
        len(param_df) == len(seeds) * len(MODEL_NAMES)
        and match_flags.all()
        and max_rel_diff <= PARAMETER_MATCH_TOLERANCE,
        {
            "rows": len(param_df),
            "max_relative_difference_vs_full": max_rel_diff,
        },
    )

    # Freeze all 30 Stage-3b checkpoints by SHA256.
    checksum_rows = []
    for model_name in MODEL_NAMES:
        for seed in seeds:
            path = checkpoint_path(model_name, seed)
            if not path.exists():
                raise FileNotFoundError(path)

            ckpt = safe_torch_load(path)
            if ckpt.get("model") != model_name:
                raise RuntimeError(f"Model mismatch in {path}")
            if int(ckpt.get("seed")) != seed:
                raise RuntimeError(f"Seed mismatch in {path}")

            checksum_rows.append({
                "model": model_name,
                "seed": seed,
                "checkpoint_relative_path": str(path.relative_to(BASE_DIR)),
                "sha256": sha256_file(path),
                "hidden_dim": int(ckpt["hidden_dim"]),
                "best_epoch": int(ckpt["best_epoch"]),
                "trainable_parameter_count": int(
                    ckpt["trainable_parameter_count"]
                ),
                "validation_average_precision": float(
                    ckpt["validation_metrics"]["average_precision"]
                ),
            })

    checksum_df = pd.DataFrame(checksum_rows)
    audit_check(
        audit,
        "all_30_checkpoints_frozen",
        len(checksum_df) == 30,
        {"rows": len(checksum_df), "expected": 30},
    )
    checksum_df.to_csv(
        FREEZE_DIR / "hgnn_extension_checkpoint_checksums.csv",
        index=False,
        encoding="utf-8-sig",
    )

    test_indices = package["test_indices"].astype("int64")
    test_meta = (
        package["metadata"].iloc[test_indices].copy().reset_index(drop=True)
    )

    split_snapshot = {
        "test_sample_count": int(len(test_indices)),
        "test_index_sha256": hashlib.sha256(test_indices.tobytes()).hexdigest(),
        "test_date_min": str(pd.to_datetime(test_meta["date"]).min()),
        "test_date_max": str(pd.to_datetime(test_meta["date"]).max()),
        "test_unique_dates": int(pd.to_datetime(test_meta["date"]).nunique()),
    }

    validation_snapshot = pd.read_csv(
        STAGE3B_DIR / "hgnn_validation_model_comparison.csv"
    ).to_dict("records")

    manifest = {
        "stage": "Stage 3c",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": (
            "Freeze the reviewer-requested genuine-HGNN extension "
            "before Stage-3d test evaluation."
        ),
        "protocol_sha256": protocol_hash,
        "stage3b_script_sha256": sha256_file(STAGE3B_SCRIPT),
        "stage3b_design_manifest_sha256": sha256_file(
            STAGE3B_DIR / "stage3b_design_manifest.json"
        ),
        "stage3b_audit_sha256": sha256_file(
            STAGE3B_DIR / "stage3b_audit_report.csv"
        ),
        "models": MODEL_NAMES,
        "primary_model": PRIMARY_MODEL,
        "primary_baseline": PRIMARY_BASELINE,
        "locked_seeds": seeds,
        "parameter_match_tolerance": PARAMETER_MATCH_TOLERANCE,
        "message_passing": (
            "focal-excluded P_ij=1[i!=j]/m_i sum_e H_ie H_je/(d_e-1); Z=PX"
        ),
        "validation_snapshot": validation_snapshot,
        "split_snapshot": split_snapshot,
        "scientific_caveat": (
            "The test split was used in earlier RAHT-RH work. "
            "Stage 3b did not use it while developing the new HGNN extension. "
            "Stage 3d is therefore a frozen extension evaluation on the "
            "pre-existing held-out split, not a virgin-test claim."
        ),
    }

    plan = {
        "stage": "Stage 3d planned evaluation",
        "training_or_tuning_allowed": False,
        "evaluation_split": "pre-existing held-out test split",
        "ensemble": {
            "seeds": seeds,
            "aggregation": "mean of per-seed risk scores",
        },
        "primary_comparison": {
            "candidate": PRIMARY_MODEL,
            "baseline": PRIMARY_BASELINE,
            "metric": PRIMARY_METRIC,
            "interpretation": "descriptive candidate-minus-baseline contrast",
        },
        "primary_descriptive_stability_summary": {
            "report_mean_improvement": True,
            "report_percentile_interval": True,
            "report_nonpositive_resample_fraction": True,
            "interpretation": (
                "The non-positive fraction comes from the ordinary "
                "bootstrap distribution and is not a null-based p-value."
            ),
        },
        "secondary_ablation_comparisons": [
            {
                "candidate": PRIMARY_MODEL,
                "baseline": "HGNN-Residual-NoGate",
                "metric": PRIMARY_METRIC,
            },
            {
                "candidate": PRIMARY_MODEL,
                "baseline": "HGNN-Gated-NoResidual",
                "metric": PRIMARY_METRIC,
            },
        ],
        "secondary_ablation_reporting": (
            "Report all paired effect estimates and percentile intervals "
            "descriptively; do not apply null-hypothesis or FDR language."
        ),
        "secondary_descriptive_metrics": [
            "roc_auc",
            "precision_at_top10",
            "recall_at_top10",
            "lift_at_top10",
        ],
        "top_alert_rate": TOP_ALERT_RATE,
        "bootstrap": {
            "type": "paired moving-block bootstrap over dates",
            "block_length_dates": BOOTSTRAP_BLOCK_LENGTH_DATES,
            "iterations": BOOTSTRAP_ITERATIONS,
            "random_state": BOOTSTRAP_RANDOM_STATE,
            "confidence_interval": "2.5% to 97.5%",
        },
        "no_post_test_changes": [
            "architecture",
            "checkpoint choice",
            "relation definitions",
            "random seeds",
            "primary metric",
            "bootstrap block length",
            "bootstrap iterations",
            "descriptive reporting rule",
        ],
    }

    save_json(manifest, FREEZE_DIR / "hgnn_extension_freeze_manifest.json")
    save_json(plan, FREEZE_DIR / "hgnn_extension_evaluation_plan.json")

    audit_check(
        audit,
        "no_test_performance_calculated_in_stage3c",
        True,
        (
            "Stage 3c records only test split identity/count/date range; "
            "it does not calculate predictions or test metrics."
        ),
    )

    audit_df = pd.DataFrame(audit)
    audit_df.to_csv(
        FREEZE_DIR / "hgnn_extension_freeze_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    marker = (
        "STAGE 3c FREEZE COMPLETE\n"
        f"protocol_sha256={protocol_hash}\n"
        f"stage3b_script_sha256={sha256_file(STAGE3B_SCRIPT)}\n"
        f"models={MODEL_NAMES}\n"
        f"seeds={seeds}\n"
        f"test_samples={len(test_indices)}\n"
        "NO TEST PERFORMANCE WAS CALCULATED IN STAGE 3c.\n"
    )
    (FREEZE_DIR / "FREEZE_COMPLETE.txt").write_text(
        marker,
        encoding="utf-8",
    )

    print("\n" + "=" * 68)
    print("STAGE 3c AUDIT")
    print("=" * 68)
    print(audit_df.to_string(index=False))

    print("\nFreeze directory:")
    print(FREEZE_DIR)
    print("\nCreated:")
    print(" - hgnn_extension_freeze_manifest.json")
    print(" - hgnn_extension_evaluation_plan.json")
    print(" - hgnn_extension_checkpoint_checksums.csv")
    print(" - hgnn_extension_freeze_audit.csv")
    print(" - FREEZE_COMPLETE.txt")
    print("\nRun 03d next. Do not retrain Stage-3b checkpoints.")


if __name__ == "__main__":
    main()
