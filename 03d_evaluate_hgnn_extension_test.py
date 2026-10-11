# Project supervisor: Dr. Zohreh Kaheh.
# Undergraduate project collaborator: Yuyang Zhang.
# Initial student coding was developed with supervisory teaching and improvement guidance.
# -*- coding: utf-8 -*-

from pathlib import Path
import hashlib
import importlib.util
import json
import os

import numpy as np
import pandas as pd
import torch
import os
torch.set_num_threads(int(os.environ.get("PORTFOLIO_TORCH_THREADS", "1")))

BASE_DIR = Path(__file__).resolve().parent

STAGE3B_SCRIPT = BASE_DIR / "03b_train_genuine_hgnn_baselines.py"
STAGE3B_DIR = BASE_DIR / "hgnn_ablation_results"
CHECKPOINT_DIR = STAGE3B_DIR / "checkpoints"

FREEZE_DIR = BASE_DIR / "hgnn_extension_frozen"
FREEZE_MANIFEST = FREEZE_DIR / "hgnn_extension_freeze_manifest.json"
EVALUATION_PLAN = FREEZE_DIR / "hgnn_extension_evaluation_plan.json"
CHECKSUM_FILE = FREEZE_DIR / "hgnn_extension_checkpoint_checksums.csv"
FREEZE_AUDIT = FREEZE_DIR / "hgnn_extension_freeze_audit.csv"
FREEZE_MARKER = FREEZE_DIR / "FREEZE_COMPLETE.txt"

OUT_DIR = BASE_DIR / "hgnn_extension_test_results"
PRED_DIR = OUT_DIR / "predictions"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAMES = [
    "HGNN-Gated-Residual",
    "HGNN-Residual-NoGate",
    "HGNN-Gated-NoResidual",
]
PRIMARY_MODEL = "HGNN-Gated-Residual"
SELF_BASELINE = "Stage 2 Self-GRU"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_stage3b():
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
        raise RuntimeError(f"Stage 3d audit failed: {name}\n{details}")


def ensemble_mean(arrays):
    return np.mean(np.stack(arrays, axis=0), axis=0)


def verify_freeze(s3b):
    audit = []
    required = [
        FREEZE_MANIFEST,
        EVALUATION_PLAN,
        CHECKSUM_FILE,
        FREEZE_AUDIT,
        FREEZE_MARKER,
    ]
    missing = [str(p) for p in required if not p.exists()]
    audit_check(audit, "stage3c_freeze_files_exist", not missing, missing)

    manifest = load_json(FREEZE_MANIFEST)
    plan = load_json(EVALUATION_PLAN)

    freeze_audit_df = pd.read_csv(FREEZE_AUDIT)
    audit_check(
        audit,
        "stage3c_freeze_audit_all_pass",
        len(freeze_audit_df) > 0
        and freeze_audit_df["status"].astype(str).eq("PASS").all(),
        freeze_audit_df[["check_name", "status"]].to_dict("records"),
    )

    audit_check(
        audit,
        "stage3b_script_hash_matches_freeze",
        sha256_file(STAGE3B_SCRIPT) == manifest["stage3b_script_sha256"],
        {
            "current": sha256_file(STAGE3B_SCRIPT),
            "frozen": manifest["stage3b_script_sha256"],
        },
    )

    protocol, protocol_hash = s3b.load_locked_protocol()
    seeds = [int(x) for x in protocol["random_seeds"]]

    audit_check(
        audit,
        "protocol_hash_matches_freeze",
        protocol_hash == manifest["protocol_sha256"],
        {"current": protocol_hash, "frozen": manifest["protocol_sha256"]},
    )
    audit_check(
        audit,
        "seeds_match_freeze",
        seeds == [int(x) for x in manifest["locked_seeds"]],
        seeds,
    )
    audit_check(
        audit,
        "models_match_freeze",
        manifest["models"] == MODEL_NAMES,
        manifest["models"],
    )

    checksum_df = pd.read_csv(CHECKSUM_FILE)
    mismatches = []
    for _, row in checksum_df.iterrows():
        p = BASE_DIR / str(row["checkpoint_relative_path"])
        if not p.exists():
            mismatches.append({"path": str(p), "reason": "missing"})
            continue
        current = sha256_file(p)
        frozen = str(row["sha256"])
        if current != frozen:
            mismatches.append({
                "path": str(p),
                "reason": "sha256 mismatch",
                "current": current,
                "frozen": frozen,
            })

    audit_check(
        audit,
        "all_checkpoint_hashes_match_freeze",
        not mismatches,
        mismatches,
    )

    return manifest, plan, protocol, protocol_hash, seeds, audit


def main():
    print("=" * 72)
    print("STAGE 3d: FROZEN GENUINE-HGNN EXTENSION TEST EVALUATION")
    print("=" * 72)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nDevice:", device)

    if device.type == "cpu":
        torch.set_num_threads(int(os.environ.get("PORTFOLIO_TORCH_THREADS", "1")))

    s3b = load_stage3b()
    manifest, plan, protocol, protocol_hash, seeds, audit = verify_freeze(s3b)

    bootstrap_cfg = plan["bootstrap"]
    block_length = int(bootstrap_cfg["block_length_dates"])
    n_bootstrap = int(bootstrap_cfg["iterations"])
    bootstrap_random_state = int(bootstrap_cfg["random_state"])

    package = s3b.load_package()
    normalizer = s3b.load_stage2_normalization()

    audit_check(
        audit,
        "package_protocol_matches_freeze",
        package["package_protocol_hash"] == manifest["protocol_sha256"],
        {
            "package": package["package_protocol_hash"],
            "freeze": manifest["protocol_sha256"],
        },
    )

    test_indices = package["test_indices"].astype(np.int64)
    current_test_hash = hashlib.sha256(test_indices.tobytes()).hexdigest()

    audit_check(
        audit,
        "test_indices_match_freeze",
        current_test_hash == manifest["split_snapshot"]["test_index_sha256"],
        {
            "current": current_test_hash,
            "frozen": manifest["split_snapshot"]["test_index_sha256"],
        },
    )

    test_meta = (
        package["metadata"].iloc[test_indices].copy().reset_index(drop=True)
    )
    y_test = package["y_3_day"][test_indices].astype(int)

    print("\n========== TEST SPLIT ==========")
    print("Samples:", len(test_indices))
    print("Positives:", int(y_test.sum()))
    print("Prevalence:", float(y_test.mean()))
    print(
        "Dates:",
        pd.to_datetime(test_meta["date"]).min(),
        "to",
        pd.to_datetime(test_meta["date"]).max(),
    )

    node_table = (
        package["metadata"][["node_index", "node"]]
        .drop_duplicates()
        .sort_values("node_index")
    )
    nodes = node_table["node"].astype(str).tolist()
    if len(nodes) != 41:
        raise RuntimeError(f"Expected 41 nodes, got {len(nodes)}.")

    raw_tensor, raw_dates, raw_paths = s3b.load_raw_temporal_tensor(nodes)
    max_test_raw_time_index = int(test_meta["raw_time_index"].max())

    audit_check(
        audit,
        "raw_tensor_covers_test_origins",
        max_test_raw_time_index < raw_tensor.shape[0],
        {
            "max_test_raw_time_index": max_test_raw_time_index,
            "raw_tensor_time_length": raw_tensor.shape[0],
        },
    )

    # Same TRAIN-ONLY normalizer as Stage 2/3b.
    raw_norm = s3b.transform_all_node_tensor(raw_tensor, normalizer)
    X_self_norm = s3b.transform_sample_tensor(package["X_self"], normalizer)

    H_group = s3b.validate_incidence_matrix(
        package["H_product_group"], "Product Group"
    )
    H_storage = s3b.validate_incidence_matrix(
        package["H_storage_location"], "Storage Location"
    )

    print("\n========== GENUINE HGNN MESSAGE PASSING ==========")
    group_msg = s3b.genuine_node_hyperedge_node_propagation(raw_norm, H_group)
    storage_msg = s3b.genuine_node_hyperedge_node_propagation(
        raw_norm, H_storage
    )

    X_group_test = s3b.build_target_relation_sequences(
        package["metadata"], group_msg, test_indices
    )
    X_storage_test = s3b.build_target_relation_sequences(
        package["metadata"], storage_msg, test_indices
    )
    X_self_test = X_self_norm[test_indices]

    expected_shape = (len(test_indices), 14, 8)
    audit_check(
        audit,
        "test_input_shapes_correct",
        X_self_test.shape == X_group_test.shape == X_storage_test.shape
        == expected_shape,
        {
            "self": X_self_test.shape,
            "group": X_group_test.shape,
            "storage": X_storage_test.shape,
        },
    )

    risk_by_model = {m: [] for m in MODEL_NAMES}
    hazards_by_model = {m: [] for m in MODEL_NAMES}
    gate_group_by_model = {m: [] for m in MODEL_NAMES}
    gate_storage_by_model = {m: [] for m in MODEL_NAMES}

    self_risk_by_seed = []
    self_hazards_by_seed = []
    seed_metric_rows = []
    base_consistency_rows = []

    for seed in seeds:
        print("\n" + "=" * 72)
        print("SEED:", seed)
        print("=" * 72)

        seed_preds = {}

        for model_name in MODEL_NAMES:
            ckpt_path = checkpoint_path(model_name, seed)
            ckpt = s3b.safe_torch_load(ckpt_path, device)

            frozen_self, _ = s3b.load_frozen_self_model(seed, device)

            model = s3b.build_model(
                model_name,
                frozen_self,
                int(ckpt["hidden_dim"]),
                s3b.RELATION_DROPOUT,
            ).to(device)

            model.load_state_dict(ckpt["state_dict"])
            model.eval()

            pred = s3b.predict_model(
                model,
                X_self_test,
                X_group_test,
                X_storage_test,
                device,
            )
            seed_preds[model_name] = pred

            risk_by_model[model_name].append(pred["risk"])
            hazards_by_model[model_name].append(pred["hazards"])
            gate_group_by_model[model_name].append(pred["gate_group"])
            gate_storage_by_model[model_name].append(pred["gate_storage"])

            metrics = s3b.evaluate_risk_predictions(y_test, pred["risk"])
            seed_metric_rows.append({
                "seed": seed,
                "model": model_name,
                **metrics,
            })

            print(
                f"{model_name}: "
                f"AP={metrics['average_precision']:.6f}, "
                f"ROC-AUC={metrics['roc_auc']:.6f}"
            )

        base_full = seed_preds["HGNN-Gated-Residual"]["base_risk"]
        base_nogate = seed_preds["HGNN-Residual-NoGate"]["base_risk"]
        base_nores = seed_preds["HGNN-Gated-NoResidual"]["base_risk"]

        d1 = float(np.max(np.abs(base_full - base_nogate)))
        d2 = float(np.max(np.abs(base_full - base_nores)))
        base_consistency_rows.append({
            "seed": seed,
            "max_abs_diff_full_vs_nogate": d1,
            "max_abs_diff_full_vs_noresidual": d2,
        })
        if d1 > 1e-7 or d2 > 1e-7:
            raise RuntimeError(f"Frozen Self-GRU base mismatch at seed {seed}")

        self_risk_by_seed.append(base_full)

        # Save baseline hazards directly from the same frozen Self-GRU.
        frozen_self, _ = s3b.load_frozen_self_model(seed, device)
        hazards_batches = []
        with torch.no_grad():
            for start in range(0, len(X_self_test), s3b.BATCH_SIZE):
                end = min(start + s3b.BATCH_SIZE, len(X_self_test))
                xb = torch.from_numpy(X_self_test[start:end]).float().to(device)
                hazards_batches.append(
                    torch.sigmoid(frozen_self(xb)).cpu().numpy()
                )
        self_hazards_by_seed.append(np.concatenate(hazards_batches, axis=0))

    audit_check(
        audit,
        "all_three_models_evaluated_for_all_seeds",
        len(seed_metric_rows) == len(seeds) * len(MODEL_NAMES),
        {"rows": len(seed_metric_rows), "expected": len(seeds) * 3},
    )

    base_consistency_df = pd.DataFrame(base_consistency_rows)
    audit_check(
        audit,
        "frozen_self_base_identical_across_variants",
        base_consistency_df[
            ["max_abs_diff_full_vs_nogate",
             "max_abs_diff_full_vs_noresidual"]
        ].to_numpy().max() <= 1e-7,
        base_consistency_df.to_dict("records"),
    )

    self_ensemble_risk = ensemble_mean(self_risk_by_seed)
    self_ensemble_hazards = ensemble_mean(self_hazards_by_seed)

    ensemble_risk = {}
    ensemble_hazards = {}
    ensemble_gate_group = {}
    ensemble_gate_storage = {}

    comparison_rows = [{
        "model": "Stage 2 Self-GRU ensemble",
        **s3b.evaluate_risk_predictions(y_test, self_ensemble_risk),
    }]

    for model_name in MODEL_NAMES:
        ensemble_risk[model_name] = ensemble_mean(risk_by_model[model_name])
        ensemble_hazards[model_name] = ensemble_mean(hazards_by_model[model_name])
        ensemble_gate_group[model_name] = ensemble_mean(
            gate_group_by_model[model_name]
        )
        ensemble_gate_storage[model_name] = ensemble_mean(
            gate_storage_by_model[model_name]
        )

        comparison_rows.append({
            "model": model_name + " ensemble",
            **s3b.evaluate_risk_predictions(
                y_test, ensemble_risk[model_name]
            ),
        })

    comparison_df = pd.DataFrame(comparison_rows)

    print("\n" + "=" * 72)
    print("HELD-OUT TEST MODEL COMPARISON")
    print("=" * 72)
    print(comparison_df.to_string(index=False))

    pred_df = test_meta.copy()
    for col in pred_df.columns:
        if np.issubdtype(pred_df[col].dtype, np.datetime64):
            pred_df[col] = pd.to_datetime(pred_df[col]).astype(str)

    pred_df.insert(0, "package_sample_index", test_indices)
    pred_df["y_3_day"] = y_test
    pred_df["self_gru_ensemble_risk"] = self_ensemble_risk
    for day in range(3):
        pred_df[f"self_gru_hazard_day{day+1}"] = self_ensemble_hazards[:, day]

    for model_name in MODEL_NAMES:
        prefix = model_name.lower().replace("-", "_")
        pred_df[f"{prefix}_risk"] = ensemble_risk[model_name]
        for day in range(3):
            pred_df[f"{prefix}_hazard_day{day+1}"] = (
                ensemble_hazards[model_name][:, day]
            )
        pred_df[f"{prefix}_gate_group"] = ensemble_gate_group[model_name]
        pred_df[f"{prefix}_gate_storage"] = ensemble_gate_storage[model_name]

    pred_path = PRED_DIR / "test_hgnn_extension_ensemble_predictions.csv"
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    bootstrap_rows = []

    def run_bootstrap(
        role,
        candidate_name,
        baseline_name,
        candidate_score,
        baseline_score,
        metric_name,
        metric_fn,
        random_state_offset,
    ):
        result = s3b.moving_block_bootstrap_improvement(
            test_meta,
            y_test,
            baseline_score,
            candidate_score,
            metric_fn,
            block_length=block_length,
            n_bootstrap=n_bootstrap,
            random_state=bootstrap_random_state + random_state_offset,
        )
        bootstrap_rows.append({
            "role": role,
            "candidate": candidate_name,
            "baseline": baseline_name,
            "metric": metric_name,
            "block_length_dates": block_length,
            "bootstrap_iterations_requested": n_bootstrap,
            **result,
        })

    # Frozen primary AUPRC comparison.
    run_bootstrap(
        "primary",
        PRIMARY_MODEL,
        SELF_BASELINE,
        ensemble_risk[PRIMARY_MODEL],
        self_ensemble_risk,
        "average_precision",
        s3b.safe_average_precision,
        0,
    )

    # Frozen parameter-matched component ablations.
    run_bootstrap(
        "secondary_ablation",
        PRIMARY_MODEL,
        "HGNN-Residual-NoGate",
        ensemble_risk[PRIMARY_MODEL],
        ensemble_risk["HGNN-Residual-NoGate"],
        "average_precision",
        s3b.safe_average_precision,
        101,
    )

    run_bootstrap(
        "secondary_ablation",
        PRIMARY_MODEL,
        "HGNN-Gated-NoResidual",
        ensemble_risk[PRIMARY_MODEL],
        ensemble_risk["HGNN-Gated-NoResidual"],
        "average_precision",
        s3b.safe_average_precision,
        202,
    )

    # Secondary ROC-AUC context; not part of the primary AUPRC summary.
    run_bootstrap(
        "secondary_metric",
        PRIMARY_MODEL,
        SELF_BASELINE,
        ensemble_risk[PRIMARY_MODEL],
        self_ensemble_risk,
        "roc_auc",
        s3b.safe_roc_auc,
        303,
    )

    bootstrap_df = pd.DataFrame(bootstrap_rows)

    primary_row = bootstrap_df[
        bootstrap_df["role"].eq("primary")
    ].iloc[0]

    primary_descriptively_stable = bool(
        float(primary_row["mean_improvement"]) > 0
        and float(primary_row["ci_lower_95"]) > 0
    )

    bootstrap_df["descriptive_positive_interval_flag"] = False
    bootstrap_df.loc[
        bootstrap_df["role"].eq("primary"),
        "descriptive_positive_interval_flag"
    ] = primary_descriptively_stable

    print("\n" + "=" * 72)
    print("PAIRED MOVING-BLOCK BOOTSTRAP")
    print("=" * 72)
    print(bootstrap_df.to_string(index=False))

    comparison_df.to_csv(
        OUT_DIR / "hgnn_test_model_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(seed_metric_rows).to_csv(
        OUT_DIR / "hgnn_test_seed_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bootstrap_df.to_csv(
        OUT_DIR / "hgnn_test_moving_block_bootstrap.csv",
        index=False,
        encoding="utf-8-sig",
    )
    base_consistency_df.to_csv(
        OUT_DIR / "hgnn_test_frozen_base_consistency.csv",
        index=False,
        encoding="utf-8-sig",
    )

    audit_check(
        audit,
        "evaluation_used_frozen_models_only",
        True,
        (
            "Stage 3d performs no training, checkpoint selection, "
            "hyperparameter tuning, or architecture change."
        ),
    )
    audit_check(
        audit,
        "primary_comparison_matches_freeze_plan",
        plan["primary_comparison"]["candidate"] == PRIMARY_MODEL
        and plan["primary_comparison"]["baseline"] == SELF_BASELINE
        and plan["primary_comparison"]["metric"] == "average_precision",
        plan["primary_comparison"],
    )

    audit_df = pd.DataFrame(audit)
    audit_df.to_csv(
        OUT_DIR / "hgnn_test_audit_report.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "stage": "Stage 3d",
        "evaluation_description": (
            "Frozen genuine-HGNN extension evaluation on the "
            "pre-existing held-out test split."
        ),
        "scientific_caveat": manifest["scientific_caveat"],
        "test_samples": int(len(test_indices)),
        "test_positives": int(y_test.sum()),
        "test_prevalence": float(y_test.mean()),
        "primary_descriptive_positive_interval_flag": (
            primary_descriptively_stable
        ),
        "model_comparison": comparison_df.to_dict("records"),
        "bootstrap": bootstrap_df.to_dict("records"),
    }
    save_json(summary, OUT_DIR / "hgnn_test_result_summary.json")

    text = (
        "STAGE 3d FROZEN HGNN EXTENSION EVALUATION\n"
        + "=" * 58
        + "\n\n"
        + f"Test samples: {len(test_indices)}\n"
        + f"Test positives: {int(y_test.sum())}\n"
        + f"Prevalence: {float(y_test.mean()):.6f}\n\n"
        + "MODEL COMPARISON\n"
        + comparison_df.to_string(index=False)
        + "\n\nBOOTSTRAP\n"
        + bootstrap_df.to_string(index=False)
        + "\n\nPrimary descriptive positive-interval flag: "
        + f"{primary_descriptively_stable}\n\n"
        + "SCIENTIFIC CAVEAT\n"
        + manifest["scientific_caveat"]
        + "\n"
    )
    (OUT_DIR / "hgnn_test_result_summary.txt").write_text(
        text, encoding="utf-8"
    )

    print("\n" + "=" * 72)
    print("STAGE 3d AUDIT")
    print("=" * 72)
    print(audit_df.to_string(index=False))

    print("\nResults saved to:")
    print(OUT_DIR)
    print("\nImportant files:")
    print(" - hgnn_test_model_comparison.csv")
    print(" - hgnn_test_seed_metrics.csv")
    print(" - hgnn_test_moving_block_bootstrap.csv")
    print(" - hgnn_test_frozen_base_consistency.csv")
    print(" - predictions/test_hgnn_extension_ensemble_predictions.csv")
    print(" - hgnn_test_audit_report.csv")
    print(" - hgnn_test_result_summary.json")
    print(" - hgnn_test_result_summary.txt")
    print(
        "\nPRIMARY DESCRIPTIVE POSITIVE-INTERVAL FLAG:",
        primary_descriptively_stable,
    )
    print(
        "\nDo not modify Stage-3b architecture/checkpoints after seeing "
        "these results."
    )


if __name__ == "__main__":
    main()
