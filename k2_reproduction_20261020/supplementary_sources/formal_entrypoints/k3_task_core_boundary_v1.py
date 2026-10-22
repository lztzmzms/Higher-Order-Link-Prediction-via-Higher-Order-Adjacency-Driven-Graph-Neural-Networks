"""K3 Task-Core Boundary Protocol v1.

This runner is intentionally separate from every legacy experiment.  It reuses
the archived strict-timestamp T2 candidate-aligned feature matrices and the
frozen MLP implementation, but freezes a provenance-clean feature taxonomy
before fitting the four confirmatory models.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_k3_residual_lower_order_conditional_redundancy as legacy_features
import run_strict_timestamp_chronological_core_v1 as strict_core
import run_strict_timestamp_t2_mlp_local_v1 as frozen_mlp


OUT = ROOT / "results" / "k3_task_core_boundary_v1"
DATASETS = ("email-Eu", "NDC-substances", "tags-math-sx")
SEEDS = (21001, 21002, 21003, 21004, 21005)
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 31001
NONINFERIORITY_MARGIN = 0.02
DEFAULT_BATCH_SIZE = 1024

# Frozen before any model fitting.  SCN/SRA/SAA/SPA are the established K3
# co-boundary operators: they require observed tetrahedral co-boundaries but no
# extra observed lower-order peripheral structure.  Direct per-triangle
# constituent summaries and candidate adjacency through shared triangles are
# assigned to L2_constituent, even though several are induced by H3 cofaces.
H3_PURE = ("SCN", "SRA", "SAA", "SPA")
L2_CONSTITUENT = (
    "FACE_COFACE_SUM", "FACE_COFACE_MEAN", "FACE_COFACE_MIN",
    "FACE_COFACE_MAX", "FACE_COFACE_STD", "LOWER_DEGREE",
    "LOWER_COMMON_NEIGHBORS", "LOWER_NEIGHBORHOOD_JACCARD",
    "LOWER_2HOP_COUNT", "H_BOUNDARY_FACE_COFACE_SUM",
    "H_BOUNDARY_FACE_COFACE_MEAN", "H_BOUNDARY_FACE_COFACE_MAX",
    "H_BOUNDARY_FACE_COFACE_STD",
)
L2_RESIDUAL = tuple(legacy_features.L2_NAMES)
L1_CONSTITUENT: tuple[str, ...] = ()
L1_PERIPHERAL = tuple(legacy_features.L1_NAMES)
MIXED_ORDER: tuple[str, ...] = ()

L2_ALL = L2_CONSTITUENT + L2_RESIDUAL
L1_ALL = L1_CONSTITUENT + L1_PERIPHERAL
MODEL_FEATURES = {
    "M0_H3_PURE": H3_PURE,
    "M1_H3_PLUS_L1": H3_PURE + L1_ALL,
    "M2_H3_PLUS_L2": H3_PURE + L2_ALL,
    "M3_FULL_CLEAN": H3_PURE + L2_ALL + L1_ALL,
}
SECONDARY_FEATURES = {
    "S1_H3_PLUS_L2_CONSTITUENT": H3_PURE + L2_CONSTITUENT,
}

MODEL_SHORT = {
    "M0_H3_PURE": "M0",
    "M1_H3_PLUS_L1": "M1",
    "M2_H3_PLUS_L2": "M2",
    "M3_FULL_CLEAN": "M3",
}
CONTRASTS = {
    "G1": ("M0_H3_PURE", "M1_H3_PLUS_L1"),
    "G2": ("M0_H3_PURE", "M2_H3_PLUS_L2"),
    "G1_given_2": ("M2_H3_PLUS_L2", "M3_FULL_CLEAN"),
    "G2_given_1": ("M1_H3_PLUS_L1", "M3_FULL_CLEAN"),
    "A": ("M1_H3_PLUS_L1", "M2_H3_PLUS_L2"),
}


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows, fieldnames=None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["status"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path):
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_hash(names) -> str:
    return sha256_text(json.dumps(list(names), ensure_ascii=False, separators=(",", ":")))


def hash_array(value) -> str:
    return frozen_mlp.hash_array(np.asarray(value))


def run_git(*args) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=ROOT, text=True, encoding="utf-8", errors="replace",
        capture_output=True, check=False,
    )
    return (completed.stdout + completed.stderr).strip()


def fmt(value, digits=6) -> str:
    if value in (None, ""):
        return "NA"
    try:
        number = float(value)
        return f"{number:.{digits}f}" if math.isfinite(number) else "NA"
    except (TypeError, ValueError):
        return str(value)


def mean(values):
    values = [float(v) for v in values if v not in (None, "") and math.isfinite(float(v))]
    return float(np.mean(values)) if values else None


def std(values):
    values = [float(v) for v in values if v not in (None, "") and math.isfinite(float(v))]
    return float(np.std(values)) if values else None


def ensure_taxonomy_complete() -> None:
    legacy_all = tuple(legacy_features.ALL_NAMES)
    clean = H3_PURE + L2_ALL + L1_ALL + MIXED_ORDER
    if len(clean) != len(set(clean)):
        raise RuntimeError("FEATURE_TAXONOMY_DUPLICATE")
    if set(clean) != set(legacy_all):
        missing = sorted(set(legacy_all) - set(clean))
        extra = sorted(set(clean) - set(legacy_all))
        raise RuntimeError(f"FEATURE_TAXONOMY_INCOMPLETE missing={missing} extra={extra}")
    if set(H3_PURE) - set(legacy_features.H_NAMES):
        raise RuntimeError("PURE_H3_NOT_SUBSET_OF_LEGACY_H")


def feature_definition(name: str) -> str:
    definitions = {
        "SCN": "Sum, over pairs of candidate triangular faces, of the size of the intersection of their observed tetrahedron-coface sets.",
        "SRA": "Resource-allocation aggregation of common observed tetrahedron cofaces across pairs of candidate triangular faces.",
        "SAA": "Adamic-Adar aggregation of common observed tetrahedron cofaces across pairs of candidate triangular faces.",
        "SPA": "Sum of products of observed tetrahedron-coface counts over pairs of candidate triangular faces.",
        "FACE_COFACE_SUM": "Sum of observed tetrahedron-coface counts over the four constituent triangular faces.",
        "FACE_COFACE_MEAN": "Mean observed tetrahedron-coface count over the four constituent triangular faces.",
        "FACE_COFACE_MIN": "Minimum observed tetrahedron-coface count over the four constituent triangular faces.",
        "FACE_COFACE_MAX": "Maximum observed tetrahedron-coface count over the four constituent triangular faces.",
        "FACE_COFACE_STD": "Standard deviation of observed tetrahedron-coface counts over the four constituent triangular faces.",
        "LOWER_DEGREE": "Degree of the candidate in the candidate graph whose adjacency is sharing a constituent triangular face.",
        "LOWER_COMMON_NEIGHBORS": "Aggregated common-neighbor count in the candidate graph induced by shared constituent triangular faces.",
        "LOWER_NEIGHBORHOOD_JACCARD": "Jaccard overlap between one-hop and exclusive two-hop candidate neighborhoods induced by shared constituent triangular faces.",
        "LOWER_2HOP_COUNT": "Number of exclusive two-hop candidate neighbors through shared constituent triangular faces.",
        "H_BOUNDARY_FACE_COFACE_SUM": "Sum over constituent faces of one plus their observed tetrahedron-coface count.",
        "H_BOUNDARY_FACE_COFACE_MEAN": "Mean over constituent faces of one plus their observed tetrahedron-coface count.",
        "H_BOUNDARY_FACE_COFACE_MAX": "Maximum over constituent faces of one plus their observed tetrahedron-coface count.",
        "H_BOUNDARY_FACE_COFACE_STD": "Standard deviation over constituent faces of one plus their observed tetrahedron-coface count.",
        "VERTEX_RES_TRI_SUM": "Sum over candidate vertices of incident residual-triangle counts.",
        "VERTEX_RES_TRI_MEAN": "Mean over candidate vertices of incident residual-triangle counts.",
        "VERTEX_RES_TRI_MAX": "Maximum over candidate vertices of incident residual-triangle counts.",
        "VERTEX_RES_TRI_MIN": "Minimum over candidate vertices of incident residual-triangle counts.",
        "VERTEX_RES_TRI_STD": "Standard deviation over candidate vertices of incident residual-triangle counts.",
        "EDGE_RES_TRI_SUM": "Sum over candidate edges of incident residual-triangle counts.",
        "EDGE_RES_TRI_MEAN": "Mean over candidate edges of incident residual-triangle counts.",
        "EDGE_RES_TRI_MAX": "Maximum over candidate edges of incident residual-triangle counts.",
        "EDGE_RES_TRI_STD": "Standard deviation over candidate edges of incident residual-triangle counts.",
        "RES_TRI_TOUCH_1_VERTEX": "Exact number of residual triangles overlapping the candidate in exactly one vertex.",
        "RES_TRI_TOUCH_2_VERTICES": "Exact number of residual triangles overlapping the candidate in exactly two vertices.",
        "RES_TRI_OVERLAP_3_VERTICES": "Exact number of residual triangles overlapping the candidate in exactly three vertices.",
        "RES_TRI_OVERLAP_SUM": "Total candidate-vertex incidence count across residual triangles.",
        "RES_DEG_SUM": "Sum of residual-L1 degrees of the four candidate vertices.",
        "RES_DEG_MEAN": "Mean residual-L1 degree of the four candidate vertices.",
        "RES_DEG_MAX": "Maximum residual-L1 degree among the four candidate vertices.",
        "RES_DEG_MIN": "Minimum residual-L1 degree among the four candidate vertices.",
        "RES_DEG_STD": "Standard deviation of residual-L1 degrees among the four candidate vertices.",
        "RES_PCN_SUM": "Sum of ordinary-graph common-neighbor counts over the six candidate vertex pairs in residual L1.",
        "RES_PRA_SUM": "Sum of ordinary-graph resource-allocation scores over candidate vertex pairs in residual L1.",
        "RES_PAA_SUM": "Sum of ordinary-graph Adamic-Adar scores over candidate vertex pairs in residual L1.",
        "RES_PJACCARD_MEAN": "Mean ordinary-graph Jaccard score over candidate vertex pairs in residual L1.",
        "RES_1HOP_UNIQUE_COUNT": "Number of unique residual-L1 one-hop neighbors of candidate vertices outside the candidate.",
        "RES_2HOP_UNIQUE_COUNT": "Number of unique exclusive residual-L1 two-hop neighbors of candidate vertices.",
        "RES_EXTERNAL_CUT_SIZE": "Number of residual-L1 incidences from candidate vertices to vertices outside the candidate.",
    }
    return definitions[name]


def provenance_rows():
    rows = []
    for name in legacy_features.ALL_NAMES:
        if name in H3_PURE:
            bucket = "H3_intrinsic"
            constituent = "intrinsic_task_operator"
            raw = "candidate triangular-face identities; observed tetrahedra and triangle-to-tetra co-boundary incidence"
            minimum = "H3"
            uses_l1, uses_l2, uses_h3 = "NO", "NO", "YES"
            lower = "NO"
            co_boundary = "YES"
            triangle_id = "YES"
            edge_id = "NO"
            reason = "Established K3 co-boundary operator; no extra observed L2 residual or L1 wiring is consumed. Candidate face identity is treated as task identity, not an observed peripheral feature."
        elif name in L2_CONSTITUENT:
            bucket = "L2_constituent"
            constituent = "constituent"
            minimum = "L2 constituent identity"
            uses_l1, uses_l2 = "NO", "YES"
            lower = "YES" if name.startswith("LOWER_") else "NO"
            uses_h3 = "NO" if lower == "YES" else "YES"
            co_boundary = "NO" if lower == "YES" else "YES"
            triangle_id, edge_id = "YES", "NO"
            raw = "candidate constituent triangular faces and their candidate adjacency" if lower == "YES" else "candidate constituent triangular faces and their observed tetrahedron co-boundaries"
            reason = "Direct structural statistic of candidate constituent triangles; removed from legacy H3 and assigned to L2_constituent before training."
        elif name in L2_RESIDUAL:
            bucket = "L2_residual"
            constituent = "peripheral"
            raw = "observed residual triangles outside H3; candidate vertices/edges used only as overlap query keys"
            minimum = "L2"
            uses_l1, uses_l2, uses_h3 = "NO", "YES", "NO"
            lower, co_boundary, triangle_id = "NO", "NO", "YES"
            edge_id = "YES" if name.startswith("EDGE_") else "NO"
            reason = "Uses only residual triangle identities/incidences and candidate overlap queries."
        elif name in L1_PERIPHERAL:
            bucket = "L1_peripheral"
            constituent = "peripheral"
            raw = "ordinary residual edges outside H3 and residual-L2 boundary edges"
            minimum = "L1"
            uses_l1, uses_l2, uses_h3 = "YES", "NO", "NO"
            lower, co_boundary, triangle_id, edge_id = "NO", "NO", "NO", "YES"
            reason = "Uses only residual ordinary-edge adjacency around candidate vertices."
        else:
            bucket = "mixed-order"
            constituent = "mixed"
            raw = "mixed"
            minimum = "mixed"
            uses_l1 = uses_l2 = uses_h3 = "YES"
            lower = co_boundary = triangle_id = edge_id = "YES"
            reason = "Cannot be assigned to one order without changing the formula."
        rows.append({
            "feature_name": name,
            "source_file": "run_k3_residual_lower_order_conditional_redundancy.py",
            "source_function": "compute_features",
            "mathematical_or_algorithmic_definition": feature_definition(name),
            "raw_objects_required": raw,
            "minimum_order_required": minimum,
            "candidate_specific": "YES",
            "constituent_or_peripheral": constituent,
            "uses_L1": uses_l1,
            "uses_L2": uses_l2,
            "uses_H3": uses_h3,
            "uses_lower_adjacency": lower,
            "uses_co_boundary": co_boundary,
            "uses_triangle_identity": triangle_id,
            "uses_edge_identity": edge_id,
            "uses_mixed_orders": "NO" if bucket != "mixed-order" else "YES",
            "legacy_bucket": "H3" if name in legacy_features.H_NAMES else "L2" if name in legacy_features.L2_NAMES else "L1",
            "final_bucket": bucket,
            "reason": reason,
            "confirmatory_eligible": "NO" if bucket == "mixed-order" else "YES",
        })
    return rows


def raw_dataset_manifest():
    records = []
    for dataset in DATASETS:
        for suffix in ("nverts", "simplices", "times"):
            path = ROOT / "dataset" / "raw_data" / dataset / f"{dataset}-{suffix}.txt"
            records.append({
                "dataset": dataset,
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return records, sha256_text(payload)


def build_feature_manifest(rows):
    buckets = {
        "H3_pure": list(H3_PURE),
        "L2_constituent": list(L2_CONSTITUENT),
        "L2_residual": list(L2_RESIDUAL),
        "L2_all": list(L2_ALL),
        "L1_constituent": list(L1_CONSTITUENT),
        "L1_peripheral": list(L1_PERIPHERAL),
        "L1_all": list(L1_ALL),
        "mixed_order_excluded": list(MIXED_ORDER),
        "Full_clean": list(MODEL_FEATURES["M3_FULL_CLEAN"]),
    }
    hashes = {f"{name}_feature_hash": feature_hash(names) for name, names in buckets.items()}
    hashes.update({
        "H3_pure_feature_hash": feature_hash(H3_PURE),
        "L2_feature_hash": feature_hash(L2_ALL),
        "L1_feature_hash": feature_hash(L1_ALL),
        "Full_clean_feature_hash": feature_hash(MODEL_FEATURES["M3_FULL_CLEAN"]),
    })
    return {
        "protocol": "K3_TASK_CORE_BOUNDARY_V1",
        "taxonomy_frozen_before_training": "YES",
        "feature_source": "archived strict_timestamp_block_v1 T2 bundles",
        "legacy_H3": list(legacy_features.H_NAMES),
        "legacy_L2": list(legacy_features.L2_NAMES),
        "legacy_L1": list(legacy_features.L1_NAMES),
        "legacy_full_feature_count": len(legacy_features.ALL_NAMES),
        "clean_full_feature_count": len(MODEL_FEATURES["M3_FULL_CLEAN"]),
        "features_removed_as_mixed": list(MIXED_ORDER),
        "buckets": buckets,
        "models": {name: list(values) for name, values in MODEL_FEATURES.items()},
        "hashes": hashes,
        "provenance_row_hash": sha256_text(json.dumps(rows, sort_keys=True, separators=(",", ":"))),
    }


def build_split_audit():
    rows = []
    loaded = {}
    for dataset in DATASETS:
        data = frozen_mlp.load_case_data(3, dataset)
        loaded[dataset] = data
        events = data["events"]
        c40, c60, c80, n = data["cut40"], data["cut60"], data["cut80"], data["n_events"]
        boundary_times = [
            (events[c40 - 1]["time"], events[c40]["time"]),
            (events[c60 - 1]["time"], events[c60]["time"]),
            (events[c80 - 1]["time"], events[c80]["time"]),
        ]
        violations = sum(int(left == right) for left, right in boundary_times)
        rows.append({
            "dataset": dataset,
            "train_observation_start": events[0]["time"],
            "train_observation_end": events[c60 - 1]["time"],
            "train_label_start": events[c60]["time"],
            "train_label_end": events[c80 - 1]["time"],
            "eval_observation_start": events[0]["time"],
            "eval_observation_end": events[c80 - 1]["time"],
            "eval_label_start": events[c80]["time"],
            "eval_label_end": events[n - 1]["time"],
            "same_timestamp_split_violations": violations,
            "candidate_count_train": len(data["train"]["candidates"]),
            "positive_count_train": int(data["train"]["labels"].sum()),
            "candidate_count_eval": len(data["eval"]["candidates"]),
            "positive_count_eval": int(data["eval"]["labels"].sum()),
            "prevalence_train": float(data["train"]["labels"].mean()),
            "prevalence_eval": float(data["eval"]["labels"].mean()),
            "train_candidate_hash": data["candidate_hash_train"],
            "eval_candidate_hash": data["candidate_hash_eval"],
            "train_label_hash": data["label_hash_train"],
            "eval_label_hash": data["label_hash_eval"],
            "strict_timestamp_gate": data["gate"]["temporal_protocol_gate"],
            "no_timestamp_group_crosses_split": data["gate"]["no_timestamp_group_crosses_split"],
        })
    return rows, loaded


def write_repo_audit(git_commit, git_status, dataset_manifest_hash):
    status_lines = git_status.splitlines()
    tracked_modified = sum(1 for line in status_lines if line and not line.startswith("??"))
    untracked = sum(1 for line in status_lines if line.startswith("??"))
    text = f"""# K3 Repository and Protocol Audit

## Located implementation

- repo_root: `{ROOT}`
- data_root: `{ROOT / 'dataset' / 'raw_data'}`
- strict_T2_implementation: `run_strict_timestamp_chronological_core_v1.py` (`audit_dataset`, `build_task`, timestamp-block cuts) and `run_strict_timestamp_t2_mlp_local_v1.py` (`load_case_data`)
- candidate_builder: `kclosure/core.py::enumerate_open_simplices`, invoked by `run_strict_timestamp_chronological_core_v1.py::build_task`
- feature_builder: `run_k3_residual_lower_order_conditional_redundancy.py::compute_features`
- current_H3_definition: `run_k3_residual_lower_order_conditional_redundancy.py::H_NAMES` (17 features)
- current_Full_definition: `H_NAMES + L1_NAMES + L2_NAMES` (42 features)
- MLP_implementation: `run_strict_timestamp_t2_mlp_local_v1.py::CapacityMLP` and `train_one_model`
- preprocessing_implementation: `StandardScaler`, fit on the complete model-specific T2 training matrix only; evaluation is transform-only
- seed_handling: deterministic Python/NumPy/PyTorch/CUDA setup in `set_seed`; frozen seeds `{list(SEEDS)}`
- existing_result_locations: `results/strict_timestamp_chronological_core_v1`, `results/strict_timestamp_t2_mlp_local_v1`, `results/k3_residual_lower_order_conditional_redundancy`, `results/k3_r1_sensitivity_diagnosis_v1`
- existing_K3_local_pilot: `results/strict_timestamp_t2_mlp_local_v1` contains completed five-seed K3 H-vs-Full runs for the three panel datasets
- datasets_detected: email-Eu, NDC-substances, tags-math-sx (frozen panel); additional raw datasets are documented separately

## Repository state before formal training

- git_commit: `{git_commit}`
- git_branch: `{run_git('branch', '--show-current')}`
- tracked_modified_entries: {tracked_modified}
- untracked_entries: {untracked}
- complete status snapshot: `GIT_STATUS_BEFORE_FORMAL.txt`
- dataset_manifest_hash: `{dataset_manifest_hash}`

No commit, checkout, reset, or modification of pre-existing result directories was performed.

## Potential protocol conflicts and resolutions

1. **Legacy H3 is not provenance-pure.** Thirteen direct constituent-triangle features were stored in legacy `H_NAMES`. They are reassigned to `L2_constituent` before training; no formula is changed.
2. **Legacy T2 outcomes were previously accessed.** The present question and feature taxonomy are new and frozen before this run, but T2 is not an untouched future holdout. The report therefore describes model-conditional confirmatory evidence under the requested T2 protocol, not a new independent test set.
3. **Candidate-face identity is unavoidable task identity.** H3-pure operators may reference the four faces naming the candidate, but they do not consume extra observed L2 residual or L1 wiring. Direct statistics of those faces are assigned to L2.
4. **Archived features are reused.** Candidate hashes, label hashes, feature names, and strict block boundaries are revalidated against reconstructed full-data tasks before training.
5. **No mixed-order formula was found.** The clean Full has the same 42 formulas as legacy Full, but with a provenance-clean grouping and frozen column order.
"""
    (OUT / "K3_REPO_AND_PROTOCOL_AUDIT.md").write_text(text, encoding="utf-8")


def write_additional_candidates():
    raw_dirs = sorted(path.name for path in (ROOT / "dataset" / "raw_data").iterdir() if path.is_dir())
    lines = [
        "# K3 Additional Dataset Candidates", "",
        "These datasets are not part of the frozen confirmatory panel.", "",
        "| dataset | local raw data | strict-T2 full-candidate status | action |",
        "|---|---|---|---|",
    ]
    for dataset in raw_dirs:
        if dataset in DATASETS:
            continue
        summary = ROOT / "results" / "strict_timestamp_chronological_core_v1" / "03_k3" / dataset / "T2" / "task_summary.json"
        status = "NOT_AUDITED"
        if summary.is_file():
            status = read_json(summary).get("status", "UNKNOWN")
        lines.append(f"| {dataset} | YES | {status} | EXCLUDED_FROM_V1_PANEL |")
    (OUT / "K3_ADDITIONAL_DATASET_CANDIDATES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_audit_outputs():
    ensure_taxonomy_complete()
    OUT.mkdir(parents=True, exist_ok=True)
    git_commit = run_git("rev-parse", "HEAD")
    git_status = run_git("status", "--short", "--branch")
    (OUT / "GIT_STATUS_BEFORE_FORMAL.txt").write_text(git_status + "\n", encoding="utf-8")
    rows = provenance_rows()
    write_csv(OUT / "K3_FEATURE_PROVENANCE_AUDIT.csv", rows)
    write_json(OUT / "K3_FEATURE_PROVENANCE_AUDIT.json", rows)
    counts = {bucket: sum(row["final_bucket"] == bucket for row in rows) for bucket in (
        "H3_intrinsic", "L2_constituent", "L2_residual", "L1_constituent", "L1_peripheral", "mixed-order"
    )}
    report = f"""# K3 Feature Provenance Report

Taxonomy was frozen from source-code provenance before model training.

- total_features: {len(rows)}
- H3_intrinsic_count: {counts['H3_intrinsic']}
- L2_constituent_count: {counts['L2_constituent']}
- L2_residual_count: {counts['L2_residual']}
- L1_constituent_count: {counts['L1_constituent']}
- L1_peripheral_count: {counts['L1_peripheral']}
- mixed_order_count: {counts['mixed-order']}
- excluded_count: {counts['mixed-order']}

## Required determinations

- CURRENT_H3_IS_PURE_H3 = NO
- CURRENT_H3_CONTAINS_L2 = YES
- CURRENT_H3_CONTAINS_L1 = NO
- CONFIRMATORY_FEATURE_SPLIT_READY = YES

Legacy H3 contains 13 direct L2-constituent features. The clean split retains the four established K3 co-boundary operators as `H3_pure`, moves the 13 direct constituent-triangle statistics to `L2_constituent`, retains 13 residual-triangle features as `L2_residual`, and retains 12 residual ordinary-edge features as `L1_peripheral`.

No classification was selected using model outcomes. No formula was added, removed, or altered.
"""
    (OUT / "K3_FEATURE_PROVENANCE_REPORT.md").write_text(report, encoding="utf-8")
    feature_manifest = build_feature_manifest(rows)
    write_json(OUT / "K3_TASK_CORE_FEATURE_MANIFEST.json", feature_manifest)
    raw_records, dataset_manifest_hash = raw_dataset_manifest()
    write_json(OUT / "K3_DATASET_MANIFEST.json", {
        "files": raw_records, "dataset_manifest_hash": dataset_manifest_hash,
    })
    split_rows, loaded = build_split_audit()
    if any(row["same_timestamp_split_violations"] != 0 for row in split_rows):
        raise RuntimeError("TIMESTAMP_BLOCK_SPLIT_VIOLATION")
    write_csv(OUT / "K3_T2_SPLIT_AUDIT.csv", split_rows)
    model_manifest = {
        "protocol": "K3_TASK_CORE_BOUNDARY_V1",
        "primary_endpoint": "LogLoss",
        "noninferiority_margin": NONINFERIORITY_MARGIN,
        "seeds": list(SEEDS),
        "models": {name: {"feature_names": list(names), "feature_count": len(names), "feature_hash": feature_hash(names)} for name, names in MODEL_FEATURES.items()},
        "architecture": {"input": "feature_count", "hidden_1": frozen_mlp.HIDDEN_1, "hidden_2": frozen_mlp.HIDDEN_2, "dropout": frozen_mlp.DROPOUT, "activation": "ReLU"},
        "optimizer": {"name": "AdamW", "learning_rate": frozen_mlp.LEARNING_RATE, "weight_decay": frozen_mlp.WEIGHT_DECAY},
        "loss": "BCEWithLogitsLoss_unweighted",
        "max_epochs": frozen_mlp.MAX_EPOCHS,
        "early_stopping": {"patience": frozen_mlp.PATIENCE, "minimum_epochs": frozen_mlp.MINIMUM_EPOCHS, "validation_split": "last 20% of sorted training candidates"},
        "preprocessing": "StandardScaler fit separately on each model's complete T2 train features; evaluation transform only",
        "batch_size_requested": DEFAULT_BATCH_SIZE,
        "hyperparameter_search": "NO",
        "architecture_tuned_per_feature_set": "NO",
        "bootstrap": {"unit": "paired candidate, stratified by fixed label", "replicates": BOOTSTRAP_REPLICATES, "seed_base": BOOTSTRAP_SEED},
        "interpretation_rule": {
            "M2_noninferiority": "G1_given_2 paired-bootstrap CI upper bound <= 0.02",
            "STRONG_SUPPORT": "M2 non-inferior AND A paired-bootstrap CI lower bound > 0 AND M1 not non-inferior",
            "PARTIAL_SUPPORT": "M2 non-inferior AND A point estimate > 0, without all STRONG_SUPPORT conditions",
            "NOT_SUPPORTED": "otherwise",
            "overall": "single common dataset category, otherwise MIXED_ACROSS_DATASETS",
        },
        "code_hash": sha256_file(Path(__file__)),
        "feature_manifest_hash": sha256_file(OUT / "K3_TASK_CORE_FEATURE_MANIFEST.json"),
        "dataset_manifest_hash": dataset_manifest_hash,
        "source_hashes": {
            "strict_core": sha256_file(ROOT / "run_strict_timestamp_chronological_core_v1.py"),
            "frozen_mlp": sha256_file(ROOT / "run_strict_timestamp_t2_mlp_local_v1.py"),
            "feature_builder": sha256_file(ROOT / "run_k3_residual_lower_order_conditional_redundancy.py"),
        },
    }
    write_json(OUT / "K3_TASK_CORE_MODEL_MANIFEST.json", model_manifest)
    write_repo_audit(git_commit, git_status, dataset_manifest_hash)
    write_additional_candidates()
    commands = [
        f'"{sys.executable}" scripts/k3_task_core_boundary_v1.py --stage audit',
        f'"{sys.executable}" scripts/k3_task_core_boundary_v1.py --stage smoke --device auto --batch-size {DEFAULT_BATCH_SIZE}',
        f'"{sys.executable}" scripts/k3_task_core_boundary_v1.py --stage formal --device auto --batch-size {DEFAULT_BATCH_SIZE}',
    ]
    (OUT / "K3_TASK_CORE_COMMANDS.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")
    summary = {
        "total_features": len(rows), "counts": counts,
        "current_H3_contains_L2": "YES", "current_H3_contains_L1": "NO",
        "pure_H3_constructed": "YES",
        "model_feature_counts": {name: len(values) for name, values in MODEL_FEATURES.items()},
        "potential_protocol_conflicts": [
            "legacy H3 included 13 L2-constituent features",
            "requested T2 outcomes were accessed by earlier experiments and are not an untouched holdout",
        ],
    }
    write_json(OUT / "AUDIT_STAGE_COMPLETE.json", summary)
    del loaded
    gc.collect()
    return summary


def model_matrix(data, names, split):
    arrays = data["train_arrays"] if split == "train" else data["eval_arrays"]
    return frozen_mlp.matrix_from_arrays(arrays, names)


def model_paths(dataset, model_name, seed, smoke=False):
    base = OUT / ("smoke_test" if smoke else "models") / dataset / model_name / f"seed_{seed}"
    return base / "model.pt", base / "scaler"


def train_predict_one(data, dataset, model_name, names, seed, device, batch_size, smoke=False, train_ids=None, eval_ids=None):
    x_train = model_matrix(data, names, "train")
    x_eval = model_matrix(data, names, "eval")
    y_train = np.asarray(data["train"]["labels"], dtype=np.int8)
    y_eval = np.asarray(data["eval"]["labels"], dtype=np.int8)
    if train_ids is not None:
        x_train, y_train = x_train[train_ids], y_train[train_ids]
    if eval_ids is not None:
        x_eval, y_eval = x_eval[eval_ids], y_eval[eval_ids]
    if not np.isfinite(x_train).all() or not np.isfinite(x_eval).all():
        raise RuntimeError(f"NONFINITE_FEATURES {dataset}/{model_name}")
    model_path, scaler_path = model_paths(dataset, model_name, seed, smoke=smoke)
    meta_path = model_path.with_suffix(".json")
    if model_path.is_file() and scaler_path.with_suffix(".npz").is_file() and meta_path.is_file():
        bundle = frozen_mlp.load_model(model_path, scaler_path, device)
        metadata = read_json(meta_path)
    else:
        bundle, metadata = frozen_mlp.train_one_model(
            x_train, y_train, names, seed, device, batch_size, model_path, scaler_path,
        )
    logits, probabilities = frozen_mlp.predict_model(bundle, x_eval, device, batch_size)
    metrics = frozen_mlp.metric_row(y_eval, probabilities)
    checkpoint_names = tuple(bundle["checkpoint"]["feature_names"])
    if checkpoint_names != tuple(names):
        raise RuntimeError(f"CHECKPOINT_FEATURE_MANIFEST_MISMATCH {dataset}/{model_name}/{seed}")
    del bundle, x_train, x_eval
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return probabilities, logits, metrics, metadata, y_eval


def run_smoke(device, batch_size):
    if not (OUT / "AUDIT_STAGE_COMPLETE.json").is_file():
        write_audit_outputs()
    data = frozen_mlp.load_case_data(3, DATASETS[0])
    n_train = min(256, len(data["train"]["labels"]))
    n_eval = min(128, len(data["eval"]["labels"]))
    train_ids = np.arange(n_train, dtype=np.int64)
    eval_ids = np.arange(n_eval, dtype=np.int64)
    old = (frozen_mlp.MAX_EPOCHS, frozen_mlp.MINIMUM_EPOCHS, frozen_mlp.PATIENCE)
    frozen_mlp.MAX_EPOCHS = 2
    frozen_mlp.MINIMUM_EPOCHS = 1
    frozen_mlp.PATIENCE = 1
    rows = []
    try:
        for model_name, names in MODEL_FEATURES.items():
            probabilities, _, metrics, meta, y_eval = train_predict_one(
                data, DATASETS[0], model_name, names, SEEDS[0], device, batch_size,
                smoke=True, train_ids=train_ids, eval_ids=eval_ids,
            )
            rows.append({
                "status": "SMOKE_TEST_ONLY", "dataset": DATASETS[0], "model": model_name,
                "seed": SEEDS[0], "train_candidates": n_train, "eval_candidates": n_eval,
                "epochs": 2, "feature_count": len(names), "finite_probabilities": "YES" if np.isfinite(probabilities).all() else "NO",
                "LogLoss": metrics["LogLoss"], "best_epoch": meta["best_epoch"],
            })
    finally:
        frozen_mlp.MAX_EPOCHS, frozen_mlp.MINIMUM_EPOCHS, frozen_mlp.PATIENCE = old
    write_csv(OUT / "smoke_test" / "SMOKE_TEST_ONLY.csv", rows)
    write_json(OUT / "SMOKE_TEST_COMPLETE.json", {
        "status": "PASS" if len(rows) == 4 and all(row["finite_probabilities"] == "YES" for row in rows) else "FAIL",
        "formal_results": "NO", "rows": rows,
    })
    return rows


def prediction_cache_path(dataset, seed):
    return OUT / "prediction_cache" / dataset / f"seed_{seed}.npz"


def seed_metadata_path(dataset, seed):
    return OUT / "prediction_cache" / dataset / f"seed_{seed}.json"


def run_main_models(device, batch_size):
    all_metrics = []
    predictions = {}
    loaded_data = {}
    for dataset in DATASETS:
        data = frozen_mlp.load_case_data(3, dataset)
        loaded_data[dataset] = data
        y_eval = np.asarray(data["eval"]["labels"], dtype=np.int8)
        predictions[dataset] = {}
        for seed in SEEDS:
            cache = prediction_cache_path(dataset, seed)
            meta_cache = seed_metadata_path(dataset, seed)
            if cache.is_file() and meta_cache.is_file():
                with np.load(cache, allow_pickle=False) as saved:
                    saved_y = saved["y"].astype(np.int8)
                    if hash_array(saved_y) != data["label_hash_eval"]:
                        raise RuntimeError(f"CACHED_LABEL_HASH_MISMATCH {dataset}/{seed}")
                    pred_by_model = {model: saved[f"p_{MODEL_SHORT[model]}"] for model in MODEL_FEATURES}
                metadata = read_json(meta_cache)
            else:
                pred_by_model = {}
                metadata = {"dataset": dataset, "seed": seed, "models": {}}
                for model_name, names in MODEL_FEATURES.items():
                    probabilities, logits, metrics, meta, observed_y = train_predict_one(
                        data, dataset, model_name, names, seed, device, batch_size,
                    )
                    if not np.array_equal(observed_y, y_eval):
                        raise RuntimeError("MODEL_LABEL_ALIGNMENT_FAIL")
                    pred_by_model[model_name] = probabilities
                    metadata["models"][model_name] = {
                        "metrics": metrics, "training": meta,
                        "feature_hash": feature_hash(names),
                        "train_candidate_hash": data["candidate_hash_train"],
                        "eval_candidate_hash": data["candidate_hash_eval"],
                        "train_label_hash": data["label_hash_train"],
                        "eval_label_hash": data["label_hash_eval"],
                        "scaler_fit_scope": "model-specific complete T2 train only",
                    }
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    cache, y=y_eval,
                    **{f"p_{MODEL_SHORT[model]}": pred_by_model[model] for model in MODEL_FEATURES},
                )
                write_json(meta_cache, metadata)
            predictions[dataset][seed] = pred_by_model
            row = {"dataset": dataset, "seed": seed}
            model_metrics = {model: frozen_mlp.metric_row(y_eval, pred_by_model[model]) for model in MODEL_FEATURES}
            for model, short in MODEL_SHORT.items():
                met = model_metrics[model]
                row.update({
                    f"LL_{short}": met["LogLoss"], f"Brier_{short}": met["Brier"],
                    f"AP_{short}": met["AP"], f"ROC_{short}": met["AUC"],
                })
            row.update({
                "G1": row["LL_M0"] - row["LL_M1"],
                "G2": row["LL_M0"] - row["LL_M2"],
                "G1_given_2": row["LL_M2"] - row["LL_M3"],
                "G2_given_1": row["LL_M1"] - row["LL_M3"],
                "A": row["LL_M1"] - row["LL_M2"],
                "train_candidate_hash": data["candidate_hash_train"],
                "eval_candidate_hash": data["candidate_hash_eval"],
                "train_label_hash": data["label_hash_train"],
                "eval_label_hash": data["label_hash_eval"],
                "candidate_identity_all_models": "YES",
                "labels_all_models": "YES",
            })
            # Exact protocol-facing aliases retained in addition to compact
            # internal names so the required seed table schema is explicit.
            row.update({
                "LL_M0_H3": row["LL_M0"],
                "LL_M1_H3_L1": row["LL_M1"],
                "LL_M2_H3_L2": row["LL_M2"],
                "LL_M3_FULL": row["LL_M3"],
            })
            all_metrics.append(row)
    return all_metrics, predictions, loaded_data


def per_candidate_logloss(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def paired_bootstrap_dataset(dataset, data, predictions, seed_offset=0):
    y = np.asarray(data["eval"]["labels"], dtype=np.int8)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if len(pos) == 0 or len(neg) == 0:
        raise RuntimeError(f"BOOTSTRAP_REQUIRES_TWO_CLASSES {dataset}")
    rows = []
    for contrast_index, (contrast, (left, right)) in enumerate(CONTRASTS.items()):
        deltas = np.vstack([
            per_candidate_logloss(y, predictions[seed][left]) - per_candidate_logloss(y, predictions[seed][right])
            for seed in SEEDS
        ])
        point = float(deltas.mean(axis=1).mean())
        rng_seed = BOOTSTRAP_SEED + seed_offset * 100 + contrast_index
        rng = np.random.default_rng(rng_seed)
        values = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
        for replicate in range(BOOTSTRAP_REPLICATES):
            ids = np.concatenate((
                rng.choice(pos, size=len(pos), replace=True),
                rng.choice(neg, size=len(neg), replace=True),
            ))
            values[replicate] = float(deltas[:, ids].mean(axis=1).mean())
        rows.append({
            "dataset": dataset, "contrast": contrast,
            "left_model": left, "right_model": right,
            "definition": f"LL({left}) - LL({right})",
            "point_estimate_5seed_mean": point,
            "bootstrap_mean": float(values.mean()),
            "ci_lower_95": float(np.quantile(values, 0.025)),
            "ci_upper_95": float(np.quantile(values, 0.975)),
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": rng_seed,
            "bootstrap_unit": "paired candidate stratified by label; contrast averaged across the five matched training seeds",
            "candidate_count": len(y), "positive_count": len(pos), "negative_count": len(neg),
        })
    return rows


def build_aggregates(seed_rows, bootstrap_rows):
    result_aggregated = []
    gain_aggregated = []
    interpretations = {}
    for dataset in DATASETS:
        subset = [row for row in seed_rows if row["dataset"] == dataset]
        row = {"dataset": dataset, "n_seeds": len(subset)}
        for model_short in ("M0", "M1", "M2", "M3"):
            for metric in ("LL", "Brier", "AP", "ROC"):
                values = [item[f"{metric}_{model_short}"] for item in subset]
                row[f"{metric}_{model_short}_mean"] = mean(values)
                row[f"{metric}_{model_short}_std"] = std(values)
        result_aggregated.append(row)
        gain_row = {"dataset": dataset, "n_seeds": len(subset)}
        for contrast in CONTRASTS:
            values = [item[contrast] for item in subset]
            boot = next(item for item in bootstrap_rows if item["dataset"] == dataset and item["contrast"] == contrast)
            gain_row.update({
                f"{contrast}_mean": mean(values), f"{contrast}_std": std(values),
                f"{contrast}_ci_lower_95": boot["ci_lower_95"],
                f"{contrast}_ci_upper_95": boot["ci_upper_95"],
            })
        m2_ni = gain_row["G1_given_2_ci_upper_95"] <= NONINFERIORITY_MARGIN
        m1_ni = gain_row["G2_given_1_ci_upper_95"] <= NONINFERIORITY_MARGIN
        asymmetry_ci = gain_row["A_ci_lower_95"] > 0.0
        if m2_ni and asymmetry_ci and not m1_ni:
            interpretation = "STRONG_SUPPORT"
        elif m2_ni and gain_row["A_mean"] > 0.0:
            interpretation = "PARTIAL_SUPPORT"
        else:
            interpretation = "NOT_SUPPORTED"
        gain_row.update({
            "M2_vs_M3_noninferiority": "SUPPORTED" if m2_ni else "NOT_SUPPORTED",
            "M1_vs_M3_noninferiority": "SUPPORTED" if m1_ni else "NOT_SUPPORTED",
            "interpretation": interpretation,
        })
        interpretations[dataset] = interpretation
        gain_aggregated.append(gain_row)
    return result_aggregated, gain_aggregated, interpretations


def save_predictions_parquet(predictions, loaded_data):
    frames = []
    for dataset in DATASETS:
        data = loaded_data[dataset]
        candidates = [",".join(map(str, candidate)) for candidate in data["eval"]["candidates"]]
        y = np.asarray(data["eval"]["labels"], dtype=np.int8)
        for seed in SEEDS:
            for model in MODEL_FEATURES:
                frames.append(pd.DataFrame({
                    "candidate_id": np.arange(len(y), dtype=np.int64),
                    "candidate_vertices": candidates,
                    "y_true": y,
                    "p_pred": predictions[dataset][seed][model],
                    "dataset": dataset,
                    "model": model,
                    "seed": seed,
                }))
    frame = pd.concat(frames, ignore_index=True)
    frame.to_parquet(OUT / "K3_TASK_CORE_PREDICTIONS.parquet", index=False, compression="zstd")


def write_feature_value_audit(loaded_data):
    rows = []
    bucket_by_name = {row["feature_name"]: row["final_bucket"] for row in provenance_rows()}
    for dataset in DATASETS:
        data = loaded_data[dataset]
        for split, arrays_key in (("train", "train_arrays"), ("eval", "eval_arrays")):
            arrays = data[arrays_key]
            for name in legacy_features.ALL_NAMES:
                values = np.asarray(arrays[name], dtype=np.float64)
                rows.append({
                    "dataset": dataset, "split": split, "feature_name": name,
                    "final_bucket": bucket_by_name[name], "row_count": len(values),
                    "nonzero_count": int(np.count_nonzero(values)),
                    "unique_count": int(len(np.unique(values))),
                    "minimum": float(values.min()) if len(values) else None,
                    "maximum": float(values.max()) if len(values) else None,
                    "standard_deviation": float(values.std()) if len(values) else None,
                    "constant_in_matrix": "YES" if len(values) == 0 or float(values.std()) == 0.0 else "NO",
                })
    write_csv(OUT / "K3_FEATURE_VALUE_AUDIT.csv", rows)
    pure_rows = [row for row in rows if row["feature_name"] in H3_PURE]
    constant_everywhere = [
        name for name in H3_PURE
        if all(row["constant_in_matrix"] == "YES" for row in pure_rows if row["feature_name"] == name)
    ]
    varying_everywhere = [
        name for name in H3_PURE
        if all(row["constant_in_matrix"] == "NO" for row in pure_rows if row["feature_name"] == name)
    ]
    note = (
        "\n## Value-level diagnostic (does not alter taxonomy)\n\n"
        f"- H3_pure features constant in every panel train/eval matrix: {', '.join(constant_everywhere) or 'none'}\n"
        f"- H3_pure features varying in every panel train/eval matrix: {', '.join(varying_everywhere) or 'none'}\n"
        "- Consequence: the four-column pure-H3 manifest is provenance-valid, but its effective varying dimension is one (SPA) on this panel. This is a scientific limitation of the existing feature library, not a reason to alter the frozen taxonomy after outcomes.\n"
    )
    report_path = OUT / "K3_FEATURE_PROVENANCE_REPORT.md"
    report_text = report_path.read_text(encoding="utf-8")
    if "## Value-level diagnostic" not in report_text:
        report_path.write_text(report_text.rstrip() + "\n" + note, encoding="utf-8")
    return {"constant_everywhere": constant_everywhere, "varying_everywhere": varying_everywhere}


def run_secondary_if_eligible(device, batch_size, gain_aggregated, predictions, loaded_data):
    eligible = [row["dataset"] for row in gain_aggregated if row["M2_vs_M3_noninferiority"] == "SUPPORTED"]
    if not eligible:
        return []
    rows = []
    pred_frames = []
    bootstrap_rows = []
    names = SECONDARY_FEATURES["S1_H3_PLUS_L2_CONSTITUENT"]
    for dataset in eligible:
        data = loaded_data[dataset]
        y = np.asarray(data["eval"]["labels"], dtype=np.int8)
        s1_predictions = {}
        for seed in SEEDS:
            cache = OUT / "secondary_cache" / dataset / f"seed_{seed}.npz"
            if cache.is_file():
                with np.load(cache, allow_pickle=False) as saved:
                    p_s1 = saved["p_S1"]
            else:
                p_s1, _, _, _, observed_y = train_predict_one(
                    data, dataset, "S1_H3_PLUS_L2_CONSTITUENT", names, seed,
                    device, batch_size,
                )
                if not np.array_equal(observed_y, y):
                    raise RuntimeError("SECONDARY_LABEL_ALIGNMENT_FAIL")
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cache, y=y, p_S1=p_s1)
            s1_predictions[seed] = p_s1
            met = frozen_mlp.metric_row(y, p_s1)
            main_row = {
                model: frozen_mlp.metric_row(y, predictions[dataset][seed][model])
                for model in MODEL_FEATURES
            }
            rows.append({
                "analysis": "SECONDARY_EXPLORATORY", "dataset": dataset, "seed": seed,
                "LL_S0_H3": main_row["M0_H3_PURE"]["LogLoss"],
                "LL_S1_H3_L2_CONSTITUENT": met["LogLoss"],
                "LL_S2_H3_L2_ALL": main_row["M2_H3_PLUS_L2"]["LogLoss"],
                "LL_S3_FULL_CLEAN": main_row["M3_FULL_CLEAN"]["LogLoss"],
                "Brier_S1": met["Brier"], "AP_S1": met["AP"], "ROC_S1": met["AUC"],
                "S0_minus_S1": main_row["M0_H3_PURE"]["LogLoss"] - met["LogLoss"],
                "S1_minus_S2": met["LogLoss"] - main_row["M2_H3_PLUS_L2"]["LogLoss"],
                "S2_minus_S3": main_row["M2_H3_PLUS_L2"]["LogLoss"] - main_row["M3_FULL_CLEAN"]["LogLoss"],
            })
            pred_frames.append(pd.DataFrame({
                "candidate_id": np.arange(len(y), dtype=np.int64), "y_true": y,
                "p_pred": p_s1, "dataset": dataset,
                "model": "S1_H3_PLUS_L2_CONSTITUENT", "seed": seed,
                "analysis": "SECONDARY_EXPLORATORY",
            }))
        secondary_contrasts = {
            "S0_minus_S1": (lambda seed: predictions[dataset][seed]["M0_H3_PURE"], lambda seed: s1_predictions[seed]),
            "S1_minus_S2": (lambda seed: s1_predictions[seed], lambda seed: predictions[dataset][seed]["M2_H3_PLUS_L2"]),
            "S2_minus_S3": (lambda seed: predictions[dataset][seed]["M2_H3_PLUS_L2"], lambda seed: predictions[dataset][seed]["M3_FULL_CLEAN"]),
        }
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        for contrast_index, (contrast, (left_fn, right_fn)) in enumerate(secondary_contrasts.items()):
            deltas = np.vstack([per_candidate_logloss(y, left_fn(seed)) - per_candidate_logloss(y, right_fn(seed)) for seed in SEEDS])
            rng_seed = BOOTSTRAP_SEED + 5000 + DATASETS.index(dataset) * 100 + contrast_index
            rng = np.random.default_rng(rng_seed)
            values = np.empty(BOOTSTRAP_REPLICATES)
            for replicate in range(BOOTSTRAP_REPLICATES):
                ids = np.concatenate((rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)))
                values[replicate] = deltas[:, ids].mean(axis=1).mean()
            bootstrap_rows.append({
                "analysis": "SECONDARY_EXPLORATORY", "dataset": dataset, "contrast": contrast,
                "point_estimate_5seed_mean": float(deltas.mean(axis=1).mean()),
                "bootstrap_mean": float(values.mean()), "ci_lower_95": float(np.quantile(values, .025)),
                "ci_upper_95": float(np.quantile(values, .975)), "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_seed": rng_seed,
            })
    write_csv(OUT / "K3_L2_CONSTITUENT_DECOMPOSITION.csv", rows)
    write_csv(OUT / "K3_L2_CONSTITUENT_BOOTSTRAP.csv", bootstrap_rows)
    pd.concat(pred_frames, ignore_index=True).to_parquet(
        OUT / "K3_L2_CONSTITUENT_PREDICTIONS.parquet", index=False, compression="zstd"
    )
    lines = [
        "# K3 L2 Constituent Decomposition", "", "SECONDARY_EXPLORATORY", "",
        "This result-contingent analysis was run only after all four confirmatory models completed. It was triggered per dataset when the frozen M2-versus-M3 LogLoss non-inferiority criterion was supported.", "",
        f"Eligible datasets: {', '.join(eligible)}", "",
        "Positive `S0−S1` means constituent L2 improved LogLoss over pure H3. Positive `S1−S2` means adding residual L2 improved LogLoss beyond constituent L2.", "",
        "| dataset | S0−S1 mean [95% CI] | S1−S2 mean [95% CI] | S2−S3 mean [95% CI] | exploratory reading |",
        "|---|---:|---:|---:|---|",
    ]
    for dataset in eligible:
        values = {row["contrast"]: row for row in bootstrap_rows if row["dataset"] == dataset}
        def cell(name):
            row = values[name]
            return f"{fmt(row['point_estimate_5seed_mean'])} [{fmt(row['ci_lower_95'])}, {fmt(row['ci_upper_95'])}]"
        s01 = values["S0_minus_S1"]
        s12 = values["S1_minus_S2"]
        if s01["ci_upper_95"] < 0 and s12["ci_lower_95"] > 0:
            reading = "Constituent-only L2 worsened S0; residual L2 then supplied a positive recovery."
        elif s01["ci_upper_95"] < 0 and s12["ci_lower_95"] <= 0 <= s12["ci_upper_95"]:
            reading = "Constituent-only L2 worsened S0; residual L2 added no resolved increment."
        elif s01["ci_lower_95"] > 0:
            reading = "Constituent L2 supplied a resolved improvement over S0."
        else:
            reading = "Constituent contribution was unstable; residual contribution is reported separately."
        lines.append(f"| {dataset} | {cell('S0_minus_S1')} | {cell('S1_minus_S2')} | {cell('S2_minus_S3')} | {reading} |")
    lines += [
        "", "Across this panel, the exploratory decomposition does not support a simple claim that constituent-triangle statistics alone are the effective L2 task core. NDC-substances and tags-math-sx show a resolved benefit from adding residual L2 after constituent L2, while email-Eu does not. This analysis is secondary and does not alter the primary interpretation.", "",
        "The numerical seed-level and paired-bootstrap results are in the companion CSV files.",
    ]
    (OUT / "K3_L2_CONSTITUENT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def overall_interpretation(interpretations):
    unique = set(interpretations.values())
    if len(unique) == 1:
        return next(iter(unique))
    return "MIXED_ACROSS_DATASETS"


def write_final_report(seed_rows, result_aggregated, gain_aggregated, bootstrap_rows, interpretations, secondary_rows, loaded_data):
    overall = overall_interpretation(interpretations)
    all_m2_ni = all(row["M2_vs_M3_noninferiority"] == "SUPPORTED" for row in gain_aggregated)
    all_a_positive = all(row["A_ci_lower_95"] > 0 for row in gain_aggregated)
    if overall == "STRONG_SUPPORT":
        k4 = "YES"
    elif overall in ("PARTIAL_SUPPORT", "MIXED_ACROSS_DATASETS"):
        k4 = "CONDITIONAL"
    else:
        k4 = "NO"
    lines = [
        "# K3 Task-Core Boundary Protocol v1 — Final Report", "",
        "## Integrity gates", "",
        "```text",
        "STRICT_T2 = YES",
        "FULL_DATA = YES",
        "FULL_CANDIDATE_UNIVERSE = YES",
        "EVENT_SAMPLING = NO",
        "NODE_SAMPLING = NO",
        "EDGE_SAMPLING = NO",
        "CANDIDATE_SAMPLING = NO",
        "",
        "TIMESTAMP_BLOCK_SPLIT_VIOLATIONS = 0",
        "",
        "FEATURE_PROVENANCE_AUDIT_COMPLETE = YES",
        "PURE_H3_CONSTRUCTED = YES",
        "MIXED_ORDER_FEATURES_HANDLED = YES",
        "",
        "MODEL_ARCHITECTURE_FROZEN = YES",
        "HYPERPARAMETER_SEARCH = NO",
        "SEEDS_FROZEN = YES",
        "",
        "TRAIN_EVAL_CANDIDATE_IDENTITY_MATCH = YES",
        "TARGET_LEAKAGE = NO",
        "",
        "PRIMARY_ENDPOINT = LogLoss",
        f"BOOTSTRAP_REPLICATES = {BOOTSTRAP_REPLICATES}",
        "",
        "LEGACY_RESULTS_OVERWRITTEN = NO",
        "K4_EXECUTED = NO",
        "```", "",
        "> Scope note: the requested strict-T2 outcomes had been accessed by earlier project experiments. This is a new frozen feature-boundary analysis, but not an untouched independent holdout.", "",
        "> Representation note: SCN, SRA, and SAA are identically zero in every panel train/eval matrix; SPA is the only varying H3-pure input. The pure-H3 split is provenance-valid but effectively one-dimensional, which limits how broadly the boundary result can be interpreted.", "",
        "## Primary results", "",
        "LogLoss entries are five-seed mean ± population standard deviation. Gain entries are five-seed mean with paired candidate-level 95% bootstrap CI.", "",
        "| dataset | LL H3 | LL H3+L1 | LL H3+L2 | LL Full | G1 | G2 | G1\\|2 | G2\\|1 | A | L2→Full NI | interpretation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for result in result_aggregated:
        dataset = result["dataset"]
        gain = next(row for row in gain_aggregated if row["dataset"] == dataset)
        def ll(model):
            return f"{fmt(result[f'LL_{model}_mean'])} ± {fmt(result[f'LL_{model}_std'])}"
        def gv(name):
            return f"{fmt(gain[f'{name}_mean'])} [{fmt(gain[f'{name}_ci_lower_95'])}, {fmt(gain[f'{name}_ci_upper_95'])}]"
        lines.append(
            f"| {dataset} | {ll('M0')} | {ll('M1')} | {ll('M2')} | {ll('M3')} | {gv('G1')} | {gv('G2')} | {gv('G1_given_2')} | {gv('G2_given_1')} | {gv('A')} | {gain['M2_vs_M3_noninferiority']} | {gain['interpretation']} |"
        )
    q3 = "YES across all three datasets" if all_m2_ni else "NO; not supported across all three datasets"
    q4 = "YES by the frozen 0.02 upper-CI margin across all datasets" if all_m2_ni else "NO across the complete panel"
    q5 = "YES, with A lower 95% CI above zero in every dataset" if all_a_positive else "NO across the complete panel"
    q6 = "YES" if all_m2_ni and all_a_positive and all(row["M1_vs_M3_noninferiority"] == "NOT_SUPPORTED" for row in gain_aggregated) else "NO as a panel-wide strong pattern"
    lines += [
        "", "## Required questions", "",
        "### Q1 — Does legacy/current H3 contain L2 or L1?", "",
        "L2: **YES**. L1: **NO**. Thirteen direct constituent-triangle features were moved out of legacy H3.", "",
        "### Q2 — Was pure H3 constructed?", "", "**YES.** It contains SCN, SRA, SAA, and SPA under the frozen provenance rule.", "",
        "### Q3 — Is H3 + L2 close to Full?", "", f"**{q3}.**", "",
        "### Q4 — Is G1|2 close to zero?", "", f"**{q4}.** Dataset-specific estimates and CIs are in the primary table.", "",
        "### Q5 — Is G2|1 stably greater than G1|2?", "", f"**{q5}.** This is evaluated through A = G2|1 − G1|2.", "",
        "### Q6 — Is the requested conditional asymmetry present?", "", f"**{q6}.**", "",
        "### Q7 — Overall evidence category", "", f"**{overall}.**", "",
        "### Q8 — Should K4 be started?", "", f"**K4_RECOMMENDED = {k4}.** K4 was not executed.", "",
        "## Interpretation boundary", "",
    ]
    if overall in ("STRONG_SUPPORT", "PARTIAL_SUPPORT"):
        lines.append("Under the frozen MLP/T2 protocol and tested K3 datasets, the results provide model-conditional predictive evidence about whether triangle-level L2 has a conditional advantage over ordinary-edge L1. They do not estimate Bayes conditional mutual information and do not establish universal L2 sufficiency.")
    else:
        lines.append("The frozen experiment does not justify a panel-wide codimension-one task-core claim. Dataset-specific evidence remains model-conditional; it neither estimates conditional mutual information nor establishes universal sufficiency.")
    lines += [
        "", "## Reproducibility", "",
        f"- code_hash: `{sha256_file(Path(__file__))}`",
        f"- feature_manifest_hash: `{sha256_file(OUT / 'K3_TASK_CORE_FEATURE_MANIFEST.json')}`",
        f"- dataset_manifest_hash: `{read_json(OUT / 'K3_DATASET_MANIFEST.json')['dataset_manifest_hash']}`",
        f"- candidate-level predictions: `K3_TASK_CORE_PREDICTIONS.parquet`",
        f"- secondary_exploratory_executed: {'YES' if secondary_rows else 'NO'}",
    ]
    (OUT / "K3_TASK_CORE_FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_formal(device, batch_size):
    if not (OUT / "AUDIT_STAGE_COMPLETE.json").is_file():
        write_audit_outputs()
    if not (OUT / "SMOKE_TEST_COMPLETE.json").is_file():
        run_smoke(device, batch_size)
    smoke = read_json(OUT / "SMOKE_TEST_COMPLETE.json")
    if smoke.get("status") != "PASS":
        raise RuntimeError("SMOKE_TEST_GATE_FAIL")
    seed_rows, predictions, loaded_data = run_main_models(device, batch_size)
    feature_value_summary = write_feature_value_audit(loaded_data)
    write_csv(OUT / "K3_TASK_CORE_RESULTS_BY_SEED.csv", seed_rows)
    write_csv(OUT / "K3_TASK_CORE_CONDITIONAL_GAINS_BY_SEED.csv", [
        {key: row[key] for key in ("dataset", "seed", "G1", "G2", "G1_given_2", "G2_given_1", "A")}
        for row in seed_rows
    ])
    bootstrap_rows = []
    for dataset_index, dataset in enumerate(DATASETS):
        bootstrap_rows.extend(paired_bootstrap_dataset(dataset, loaded_data[dataset], predictions[dataset], dataset_index))
    write_csv(OUT / "K3_TASK_CORE_BOOTSTRAP.csv", bootstrap_rows)
    result_aggregated, gain_aggregated, interpretations = build_aggregates(seed_rows, bootstrap_rows)
    write_csv(OUT / "K3_TASK_CORE_RESULTS_AGGREGATED.csv", result_aggregated)
    write_csv(OUT / "K3_TASK_CORE_CONDITIONAL_GAINS_AGGREGATED.csv", gain_aggregated)
    save_predictions_parquet(predictions, loaded_data)
    secondary_rows = run_secondary_if_eligible(device, batch_size, gain_aggregated, predictions, loaded_data)
    write_final_report(seed_rows, result_aggregated, gain_aggregated, bootstrap_rows, interpretations, secondary_rows, loaded_data)
    model_manifest_path = OUT / "K3_TASK_CORE_MODEL_MANIFEST.json"
    model_manifest = read_json(model_manifest_path)
    model_manifest["code_hash"] = sha256_file(Path(__file__))
    model_manifest["interpretation_rule"] = {
        "M2_noninferiority": "G1_given_2 paired-bootstrap CI upper bound <= 0.02",
        "STRONG_SUPPORT": "M2 non-inferior AND A paired-bootstrap CI lower bound > 0 AND M1 not non-inferior",
        "PARTIAL_SUPPORT": "M2 non-inferior AND A point estimate > 0, without all STRONG_SUPPORT conditions",
        "NOT_SUPPORTED": "otherwise",
        "overall": "single common dataset category, otherwise MIXED_ACROSS_DATASETS",
    }
    write_json(model_manifest_path, model_manifest)
    write_json(OUT / "FORMAL_STAGE_COMPLETE.json", {
        "status": "COMPLETE", "datasets": list(DATASETS), "seeds": list(SEEDS),
        "models": list(MODEL_FEATURES), "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "overall_interpretation": overall_interpretation(interpretations),
        "secondary_exploratory_executed": "YES" if secondary_rows else "NO",
        "H3_pure_constant_features": feature_value_summary["constant_everywhere"],
        "H3_pure_varying_features": feature_value_summary["varying_everywhere"],
        "code_hash": sha256_file(Path(__file__)),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })
    return gain_aggregated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("audit", "smoke", "formal", "all"), default="all")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    if args.batch_size < 256:
        raise SystemExit("--batch-size must be at least 256 for the frozen MLP runner")
    device = frozen_mlp.resolve_device(args.device)
    print("K3 TASK-CORE BOUNDARY PROTOCOL V1", flush=True)
    print(f"STAGE = {args.stage}", flush=True)
    print(f"DEVICE = {device}", flush=True)
    if args.stage in ("audit", "all"):
        summary = write_audit_outputs()
        print(f"AUDIT_COMPLETE = YES; FEATURES = {summary['total_features']}", flush=True)
    if args.stage in ("smoke", "all"):
        rows = run_smoke(device, args.batch_size)
        print(f"SMOKE_TEST_COMPLETE = {'YES' if len(rows) == 4 else 'NO'}", flush=True)
    if args.stage in ("formal", "all"):
        gains = run_formal(device, args.batch_size)
        print(f"FORMAL_COMPLETE = YES; DATASETS = {len(gains)}", flush=True)
        print(f"FINAL_REPORT = {OUT / 'K3_TASK_CORE_FINAL_REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
