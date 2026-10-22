"""Build deterministic local-to-server manifests from completed experiment artifacts."""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from scripts.task_core_common_v1 import canonical_hash, read_csv, read_json, sha256_file, write_csv, write_json

HANDOFF = ROOT / "results" / "task_core_local_server_handoff"
K3 = ROOT / "results" / "k3_spatial_order_task_core_v1"
K4 = ROOT / "results" / "k4_task_core_perturbation_v1"
MLP_SEEDS = tuple(range(21001, 21006)); MODELS = ("B0", "B1", "B2", "B3", "B4", "B5")
TARGETS = (.20, .40, .60, .80, .95)
R1 = {.20: range(51001, 51011), .40: range(52001, 52011), .60: range(53001, 53011), .80: range(54001, 54011), .95: range(55001, 55011)}


def snapshot(root, names):
    target = root / "local"; target.mkdir(parents=True, exist_ok=True)
    for name in names:
        source = root / name
        if source.is_file(): shutil.copy2(source, target / name)


def validation_files():
    paths = [
        "scripts/task_core_common_v1.py", "scripts/run_k3_spatial_order_task_core_v1.py",
        "scripts/run_k4_task_core_perturbation_v1.py", "scripts/server/run_server_k3_remaining.py",
        "scripts/server/run_server_k4_remaining.py", "scripts/server/run_server_remaining.py",
        "scripts/merge_local_server_results.py", "run_strict_timestamp_chronological_core_v1.py",
        "run_strict_timestamp_t2_mlp_local_v1.py", "run_strict_timestamp_t2_mlp_r1_dose_local_v1.py",
        "run_k3_residual_lower_order_conditional_redundancy.py", "kclosure/core.py",
        "results/k3_all_dataset_learned_smoke/K3_SMOKE_DATASET_MANIFEST.json",
        "results/k3_task_core_boundary_v1/K3_TASK_CORE_FEATURE_MANIFEST.json",
        "results/strict_timestamp_chronological_core_v1/01_timestamp_audit/audit_cache.json",
        "results/task_core_local_server_handoff/EXPERIMENT_CONFIG_HASHES.json",
    ]
    return [{"path": value, "sha256": sha256_file(ROOT / value)} for value in paths]


def main():
    HANDOFF.mkdir(parents=True, exist_ok=True)
    snapshot(K3, ["K3_8DATASET_PANEL_MANIFEST.csv", "K3_SATURATION_FEATURE_MANIFEST_AUDIT.md", "K3_L2_SPATIAL_SHELL_DEFINITION.md", "K3_L2_SPATIAL_SHELL_AUDIT.csv", "K3_TASK_CORE_SATURATION_MODEL_MANIFEST.json", "K3_TASK_CORE_SATURATION_BY_SEED.csv", "K3_TASK_CORE_SATURATION_AGGREGATED.csv", "K3_TASK_CORE_SATURATION_PREDICTIONS.parquet", "K3_TASK_CORE_SATURATION_BOOTSTRAP.csv", "K3_TASK_CORE_MARGINAL_GAINS.csv", "K3_H3_DEGENERACY_AUDIT.md", "K3_TASK_CORE_SATURATION_FINAL_REPORT.md"])
    snapshot(K4, ["K4_FEASIBILITY_AUDIT.csv", "K4_DATASET_PANEL_MANIFEST.csv", "K4_H_PERIPHERAL_PROVENANCE_AUDIT.md", "K4_R1_PROTOCOL_AUDIT.md", "K4_H_ONLY_FULL_BY_SEED.csv", "K4_H_ONLY_FULL_AGGREGATED.csv", "K4_R1_STRUCTURAL_AUDIT.csv", "K4_R1_RESULTS_BY_SEED.csv", "K4_R1_RESULTS_AGGREGATED.csv", "K4_R1_PREDICTIONS.parquet", "K4_R1_BOOTSTRAP.csv", "K4_TASK_CORE_PERTURBATION_FINAL_REPORT.md"])
    k3_panel = read_csv(K3 / "K3_8DATASET_PANEL_MANIFEST.csv"); k4_panel = read_csv(K4 / "K4_DATASET_PANEL_MANIFEST.csv")
    completed, skipped, required = [], [], []
    k3_seed = read_csv(K3 / "K3_TASK_CORE_SATURATION_BY_SEED.csv")
    for row in k3_seed:
        completed.append({"case_id": f"K3|{row['dataset']}|{row['model']}|{row['seed']}", "experiment": "K3", "dataset": row["dataset"], "model": row["model"], "seed": row["seed"], "dose": "", "r1_seed": "", "status": "COMPLETED", "output": "results/k3_spatial_order_task_core_v1/local/K3_TASK_CORE_SATURATION_BY_SEED.csv"})
    for row in k3_panel:
        if row.get("server_required") == "YES":
            skipped.append({"case_id": f"K3|{row['dataset']}", "experiment": "K3", "dataset": row["dataset"], "status": row["local_status"], "reason": row.get("status_detail", "")})
            for model in MODELS:
                for seed in MLP_SEEDS:
                    required.append({"case_id": f"K3|{row['dataset']}|{model}|{seed}", "experiment": "K3", "dataset": row["dataset"], "model": model, "seed": seed, "dose": None, "r1_seed": None, "reason_not_run_locally": row["local_status"], "estimated_resource_issue": row.get("status_detail", ""), "server_command": "python3 -u scripts/server/run_server_k3_remaining.py --manifest results/task_core_local_server_handoff/SERVER_REQUIRED_CASES.json --source-root .", "expected_output": f"results/k3_spatial_order_task_core_v1/server/predictions/{row['dataset']}/{model}/seed_{seed}.npz"})
    skipped.append({"case_id": "K3_MATCHED_CONTROL|ALL_8", "experiment": "K3_MATCHED_CONTROL", "dataset": "ALL_8_FROZEN_PANEL", "status": "SERVER_REQUIRED", "reason": "optional priority-5 control deferred to preserve local resources after all mandatory local cases"})
    for dataset in (row["dataset"] for row in k3_panel):
        for structural_seed in (31001, 31002, 31003):
            for seed in MLP_SEEDS:
                required.append({"case_id": f"K3_MATCHED_CONTROL|{dataset}|{structural_seed}|{seed}", "experiment": "K3_MATCHED_CONTROL", "dataset": dataset, "model": "C_LOCAL_VS_MATCHED_FAR", "seed": seed, "dose": None, "r1_seed": structural_seed, "reason_not_run_locally": "SERVER_REQUIRED", "estimated_resource_issue": "optional priority-5 control deferred to preserve mandatory local experiment budget", "server_command": "python3 -u scripts/server/run_server_k3_remaining.py --manifest results/task_core_local_server_handoff/SERVER_REQUIRED_CASES.json --source-root .", "expected_output": "results/k3_spatial_order_task_core_v1/server/K3_LOCAL_L2_MATCHED_CONTROL.csv"})
    for row in read_csv(K4 / "K4_H_ONLY_FULL_BY_SEED.csv"):
        completed.append({"case_id": f"K4|{row['dataset']}|{row['model']}|{row['seed']}", "experiment": "K4", "dataset": row["dataset"], "model": row["model"], "seed": row["seed"], "dose": "", "r1_seed": "", "status": "COMPLETED", "output": "results/k4_task_core_perturbation_v1/local/K4_H_ONLY_FULL_BY_SEED.csv"})
    for row in read_csv(K4 / "K4_R1_RESULTS_BY_SEED.csv"):
        completed.append({"case_id": f"K4|{row['dataset']}|R1|{row['target_changed_fraction']}|{row['r1_seed']}|{row['mlp_seed']}", "experiment": "K4", "dataset": row["dataset"], "model": "K4_FULL_R1", "seed": row["mlp_seed"], "dose": row["target_changed_fraction"], "r1_seed": row["r1_seed"], "status": "COMPLETED", "output": "results/k4_task_core_perturbation_v1/local/K4_R1_RESULTS_BY_SEED.csv"})
    for row in k4_panel:
        if row.get("server_required") == "YES":
            skipped.append({"case_id": f"K4|{row['dataset']}", "experiment": "K4", "dataset": row["dataset"], "status": row["local_status"], "reason": row.get("reason", "")})
            if row.get("local_status") != "PARTIAL_H_FULL_COMPLETED":
                for model in ("K4_H_ONLY", "K4_FULL_REAL"):
                    for seed in MLP_SEEDS:
                        required.append({"case_id": f"K4|{row['dataset']}|{model}|{seed}", "experiment": "K4", "dataset": row["dataset"], "model": model, "seed": seed, "dose": None, "r1_seed": None, "reason_not_run_locally": row["local_status"], "estimated_resource_issue": row.get("reason", ""), "server_command": "python3 -u scripts/server/run_server_k4_remaining.py --manifest results/task_core_local_server_handoff/SERVER_REQUIRED_CASES.json --source-root .", "expected_output": f"results/k4_task_core_perturbation_v1/server/models/{row['dataset']}/{model}/seed_{seed}/model.pt"})
            for target in TARGETS:
                for r1_seed in R1[target]:
                    for seed in MLP_SEEDS:
                        required.append({"case_id": f"K4|{row['dataset']}|R1|{target:.2f}|{r1_seed}|{seed}", "experiment": "K4", "dataset": row["dataset"], "model": "K4_FULL_R1", "seed": seed, "dose": target, "r1_seed": r1_seed, "reason_not_run_locally": row["local_status"], "estimated_resource_issue": row.get("reason", ""), "server_command": "python3 -u scripts/server/run_server_k4_remaining.py --manifest results/task_core_local_server_handoff/SERVER_REQUIRED_CASES.json --source-root .", "expected_output": "results/k4_task_core_perturbation_v1/server/K4_R1_RESULTS_BY_SEED.csv"})
    write_csv(HANDOFF / "LOCAL_COMPLETED_CASES.csv", completed); write_csv(HANDOFF / "LOCAL_SKIPPED_CASES.csv", skipped)
    configs = {"git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(), "strict_T2": True, "full_data": True, "full_candidate_universe": True, "sampling": "NONE", "primary_endpoint": "LogLoss", "margin": .02, "mlp_seeds": list(MLP_SEEDS), "bootstrap_replicates": 2000, "k3_panel_size": 8, "k3_models": list(MODELS), "k4_doses": list(TARGETS), "k4_r1_seeds": {str(key): list(value) for key, value in R1.items()}}
    hashes = {"config": configs, "config_hash": canonical_hash(configs), "dataset_manifest_hash": sha256_file(ROOT / "results/k3_all_dataset_learned_smoke/K3_SMOKE_DATASET_MANIFEST.json"), "feature_manifest_hash": sha256_file(ROOT / "results/k3_task_core_boundary_v1/K3_TASK_CORE_FEATURE_MANIFEST.json"), "strict_T2_split_manifest_hash": sha256_file(AUDIT_CACHE := ROOT / "results/strict_timestamp_chronological_core_v1/01_timestamp_audit/audit_cache.json"), "model_config_hash": canonical_hash({"architecture": [64, 32], "dropout": .2, "optimizer": "AdamW", "lr": 1e-3, "weight_decay": 1e-4, "max_epochs": 100, "patience": 10, "minimum_epochs": 20}), "seed_config_hash": canonical_hash({"mlp": MLP_SEEDS, "r1": {str(k): list(v) for k, v in R1.items()}}), "bootstrap_config_hash": canonical_hash({"replicates": 2000, "margin": .02}), "R1_config_hash": canonical_hash({"targets": TARGETS, "tolerance": .02, "degree_preserving": True})}
    write_json(HANDOFF / "EXPERIMENT_CONFIG_HASHES.json", hashes)
    manifest = {"protocol": "TASK_CORE_LOCAL_SERVER_HANDOFF_V1", "config_hash": hashes["config_hash"], "validation_files": validation_files(), "cases": required}
    write_json(HANDOFF / "SERVER_REQUIRED_CASES.json", manifest)
    lines = ["# Server Execution Plan", "", f"Total granular cases: {len(required)}", "", "| case_id | experiment | dataset | model | seed | dose | R1 seed | local reason | command | expected output |", "|---|---|---|---|---:|---:|---:|---|---|---|"]
    for row in required: lines.append(f"| {row['case_id']} | {row['experiment']} | {row['dataset']} | {row['model']} | {row['seed']} | {row.get('dose') or ''} | {row.get('r1_seed') or ''} | {row['reason_not_run_locally']} | `{row['server_command']}` | `{row['expected_output']}` |")
    (HANDOFF / "SERVER_EXECUTION_PLAN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    k3_complete = sum(row.get("local_status") == "COMPLETED" for row in k3_panel); k3_server = sum(row.get("server_required") == "YES" for row in k3_panel)
    k4_eligible = sum(row.get("eligibility") == "ELIGIBLE" for row in k4_panel); k4_complete = sum(row.get("local_status") == "COMPLETED" for row in k4_panel); k4_server = sum(row.get("server_required") == "YES" for row in k4_panel)
    summary = f"""# Local Run Summary

- K3 local completed: {k3_complete}/8
- K3 server required: {k3_server}/8
- K3 matched-control server required: 8/8 (optional priority 5)
- K4 eligible datasets: {k4_eligible}
- K4 local completed: {k4_complete}
- K4 server required: {k4_server}
- Granular local completed cases: {len(completed)}
- Granular server-required cases: {len(required)}
- No scientific sampling or protocol reduction was used.
"""
    (HANDOFF / "LOCAL_RUN_SUMMARY.md").write_text(summary, encoding="utf-8")
    case_state = {row["case_id"]: {"status": "COMPLETED", "experiment": row["experiment"], "dataset": row["dataset"], "model": row["model"], "seed": row["seed"], "dose": row.get("dose", ""), "r1_seed": row.get("r1_seed", "")} for row in completed}
    case_state.update({row["case_id"]: {"status": "SERVER_REQUIRED", "experiment": row["experiment"], "dataset": row["dataset"], "model": row["model"], "seed": row["seed"], "dose": row.get("dose"), "r1_seed": row.get("r1_seed"), "reason": row["reason_not_run_locally"]} for row in required})
    write_json(HANDOFF / "RUN_STATE.json", {"status": "LOCAL_COMPLETE_SERVER_PENDING" if required else "COMPLETE", "skip_completed": "YES", "resume_granularity": {"K3": "dataset x model x MLP_seed", "K4_R1": "dataset x dose x R1_seed x MLP_seed"}, "k3": {"local_complete": k3_complete, "server_required": k3_server}, "k4": {"eligible": k4_eligible, "local_complete": k4_complete, "server_required": k4_server}, "completed_case_count": len(completed), "server_case_count": len(required), "config_hash": hashes["config_hash"], "cases": case_state})


if __name__ == "__main__": main()
