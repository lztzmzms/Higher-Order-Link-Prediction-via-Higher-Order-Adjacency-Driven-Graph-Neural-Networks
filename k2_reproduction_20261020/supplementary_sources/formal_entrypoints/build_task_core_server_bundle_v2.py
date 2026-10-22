"""Build and validate the self-contained Task-Core v2 server handoff."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import textwrap
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "task_core_server_bundle_v2"
ZIP_PATH = ROOT / "task_core_server_bundle_v2.zip"
RESULTS = ROOT / "results" / "task_core_confirmatory_v2"
K3_SOURCE = ROOT / "results" / "k3_spatial_order_task_core_v1"
K3_OLD = ROOT / "results" / "k3_task_core_boundary_v1"

K3_PANEL = ["email-Eu", "NDC-substances", "tags-math-sx", "tags-ask-ubuntu", "threads-ask-ubuntu"]
K4_PANEL = ["email-Eu", "tags-ask-ubuntu", "tags-math-sx"]
K3_PENDING = ["tags-ask-ubuntu", "threads-ask-ubuntu"]
MODELS = ["B0", "B1", "B2", "B3", "B4", "B5"]
SEEDS = [21001, 21002, 21003, 21004, 21005]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(value).lstrip(), encoding="utf-8", newline="\n")


def copy_file(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def local_module(name: str) -> Path | None:
    file_path = ROOT.joinpath(*name.split(".")).with_suffix(".py")
    if file_path.is_file():
        return file_path
    init_path = ROOT.joinpath(*name.split("."), "__init__.py")
    return init_path if init_path.is_file() else None


def copy_import_closure(initial: list[str]) -> list[str]:
    """Copy the transitive local Python import closure, not the repository."""
    queue = list(initial)
    seen: set[str] = set()
    copied: list[str] = []
    while queue:
        module = queue.pop(0)
        if module in seen:
            continue
        seen.add(module)
        path = local_module(module)
        if path is None:
            continue
        relative = path.relative_to(ROOT)
        copy_file(path, BUNDLE / "src" / "project" / relative)
        copied.append(relative.as_posix())
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        package = module.split(".")[:-1]
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                base_parts = package[: len(package) - max(node.level - 1, 0)] if node.level else []
                base = ".".join(base_parts + ([node.module] if node.module else []))
                if base:
                    names.append(base)
                for alias in node.names:
                    candidate = f"{base}.{alias.name}" if base else alias.name
                    if local_module(candidate):
                        names.append(candidate)
            for imported in names:
                if local_module(imported):
                    queue.append(imported)
                # Ensure package initializers are retained when present.
                pieces = imported.split(".")
                for index in range(1, len(pieces)):
                    prefix = ".".join(pieces[:index])
                    if local_module(prefix):
                        queue.append(prefix)
    return sorted(set(copied))


def data_requirement(dataset: str) -> dict:
    names = [f"{dataset}-nverts.txt", f"{dataset}-simplices.txt", f"{dataset}-times.txt"]
    raw = ROOT / "dataset" / "raw_data" / dataset
    known = {name: sha(raw / name) for name in names if (raw / name).is_file()}
    return {
        "dataset_name": dataset,
        "status": "CANONICAL_DATA_REQUIRED_ON_SERVER" if dataset == "threads-ask-ubuntu" else "FULL_CANONICAL_DATA_REQUIRED_ON_SERVER",
        "required_relative_path": f"dataset/raw_data/{dataset}",
        "expected_file_names": names,
        "expected_sha256_if_available": known,
        "expected_schema": {
            "nverts": "one integer simplex cardinality per event",
            "simplices": "flattened integer vertex identifiers; consecutive blocks follow nverts",
            "times": "one timestamp per event, aligned one-to-one with nverts",
        },
        "raw_data_in_zip": "NO",
    }


RUN_K3 = r'''
from __future__ import annotations
import argparse, sys
from pathlib import Path

BUNDLE = Path(__file__).resolve().parents[1]
PROJECT = BUNDLE / "src" / "project"
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))
from scripts import run_k3_spatial_order_task_core_v1 as runner

def main() -> None:
    runner.ROOT = BUNDLE
    runner.OUT = BUNDLE / "server_results" / "k3_remaining"
    runner.LOCAL = runner.OUT / "local"
    frozen = BUNDLE / "local_results" / "frozen"
    runner.OLD = frozen / "k3_old"
    runner.PANEL_SOURCE = frozen / "k3_panel_source.json"
    runner.AUDIT_CACHE = frozen / "timestamp_audit_cache.json"
    runner.TAG_PROTOCOL = frozen / "tags_ask_protocol.json"
    runner.strict.ROOT = BUNDLE
    runner.mlp.ROOT = BUNDLE
    args = argparse.Namespace(datasets="tags-ask-ubuntu,threads-ask-ubuntu", server_mode=True, negative_control_only=False)
    runner.run(args)

if __name__ == "__main__":
    main()
'''


NOOP_TEMPLATE = r'''
from __future__ import annotations
import json
from pathlib import Path

BUNDLE = Path(__file__).resolve().parents[1]
STAGE = "{stage}"

def main() -> None:
    cases = json.loads((BUNDLE / "manifests" / "SERVER_REQUIRED_CASES.json").read_text(encoding="utf-8"))["cases"]
    pending = [row for row in cases if row["experiment"] == STAGE]
    if pending:
        raise RuntimeError(f"UNEXPECTED_UNIMPLEMENTED_SERVER_CASES:{{STAGE}}:{{len(pending)}}")
    print(f"{{STAGE}}: no server cases; local result is complete or R1 is structurally inapplicable")

if __name__ == "__main__":
    main()
'''


MERGE = r'''
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

BUNDLE = Path(__file__).resolve().parents[1]
LOCAL = BUNDLE / "local_results" / "completed" / "k3_final_5dataset"
SERVER = BUNDLE / "server_results" / "k3_remaining"
FINAL = BUNDLE / "server_results" / "final" / "k3_final_5dataset"
PANEL = ["email-Eu", "NDC-substances", "tags-math-sx", "tags-ask-ubuntu", "threads-ask-ubuntu"]

def combine(name: str, server_name: str | None = None) -> pd.DataFrame:
    parts = []
    local_path = LOCAL / name
    if local_path.is_file(): parts.append(pd.read_csv(local_path))
    server_path = SERVER / (server_name or name)
    if server_path.is_file(): parts.append(pd.read_csv(server_path))
    if not parts: raise FileNotFoundError(name)
    frame = pd.concat(parts, ignore_index=True)
    key = [value for value in ("dataset", "model", "seed", "reference") if value in frame.columns]
    return frame[frame.dataset.isin(PANEL)].drop_duplicates(key or None, keep="last")

def main() -> None:
    FINAL.mkdir(parents=True, exist_ok=True)
    by_seed = combine("K3_FINAL_5DATASET_SATURATION_BY_SEED.csv", "K3_TASK_CORE_SATURATION_BY_SEED.csv")
    aggregate = combine("K3_FINAL_5DATASET_SATURATION_AGGREGATED.csv", "K3_TASK_CORE_SATURATION_AGGREGATED.csv")
    bootstrap = combine("K3_FINAL_5DATASET_BOOTSTRAP.csv", "K3_TASK_CORE_SATURATION_BOOTSTRAP.csv")
    by_seed.to_csv(FINAL / "K3_FINAL_5DATASET_SATURATION_BY_SEED.csv", index=False)
    aggregate.to_csv(FINAL / "K3_FINAL_5DATASET_SATURATION_AGGREGATED.csv", index=False)
    bootstrap.to_csv(FINAL / "K3_FINAL_5DATASET_BOOTSTRAP.csv", index=False)
    panel = pd.read_csv(LOCAL / "K3_FINAL_5DATASET_PANEL_MANIFEST.csv").set_index("dataset")
    server_panel_path = SERVER / "K3_8DATASET_PANEL_MANIFEST.csv"
    if server_panel_path.is_file():
        server_panel = pd.read_csv(server_panel_path).set_index("dataset")
        for dataset in ("tags-ask-ubuntu", "threads-ask-ubuntu"):
            if dataset in server_panel.index:
                for column, value in server_panel.loc[dataset].items(): panel.loc[dataset, column] = value
    panel.reset_index().to_csv(FINAL / "K3_FINAL_5DATASET_PANEL_MANIFEST.csv", index=False)
    missing = sorted(set(PANEL) - set(aggregate.dataset))
    if missing: raise RuntimeError(f"FINAL_PANEL_INCOMPLETE: {missing}")
    lines = ["# K3 Final Five-Dataset Saturation Report", "", "- K3_FINAL_PANEL_SIZE = 5", "- N_COMPLETED = 5", "- N_PENDING_SERVER = 0", "- Excluded from all aggregates: email-Enron, contact-high-school, contact-primary-school.", "", "| dataset | B0 | B1 | B2 | B3 | B4 | Full | minimum saturation | CI upper | classification |", "|---|---:|---:|---:|---:|---:|---:|---|---:|---|"]
    for row in aggregate.set_index("dataset").loc[PANEL].itertuples():
        lines.append(f"| {row.Index} | {row.LogLoss_B0:.6f} | {row.LogLoss_B1:.6f} | {row.LogLoss_B2:.6f} | {row.LogLoss_B3:.6f} | {row.LogLoss_B4:.6f} | {row.LogLoss_B5:.6f} | {row.minimum_saturation_model} | {row.saturation_ci_upper:.6f} | {row.classification} |")
    (FINAL / "K3_FINAL_5DATASET_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"merged final K3 panel: {len(aggregate)} datasets")

if __name__ == "__main__":
    main()
'''


VALIDATE = r'''
from __future__ import annotations
import argparse, hashlib, importlib, json, os, platform, sys
from pathlib import Path

BUNDLE = Path(__file__).resolve().parents[1]

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-missing-data", action="store_true")
    args = parser.parse_args()
    errors, warnings = [], []
    required = ["README_SERVER.md", "RUN_ALL_SERVER.sh", "run_all_server.py", "configs/experiment_config.json", "manifests/SERVER_REQUIRED_CASES.json", "manifests/LOCAL_COMPLETED_CASES.csv", "manifests/DATA_REQUIREMENTS.json", "manifests/HASHES.json"]
    for name in required:
        if not (BUNDLE / name).is_file(): errors.append(f"missing bundle file: {name}")
    hashes = json.loads((BUNDLE / "manifests" / "HASHES.json").read_text(encoding="utf-8")) if not errors else {"files": {}}
    for relative, expected in hashes.get("files", {}).items():
        path = BUNDLE / relative
        if not path.is_file(): errors.append(f"hash target absent: {relative}")
        elif sha(path) != expected: errors.append(f"bundle hash mismatch: {relative}")
    if not errors:
        config = json.loads((BUNDLE / "configs" / "experiment_config.json").read_text(encoding="utf-8"))
        frozen = config["frozen_hashes"]
        explicit = {
            "feature_manifest": BUNDLE / "local_results/frozen/k3_old/K3_TASK_CORE_FEATURE_MANIFEST.json",
            "split_audit_cache": BUNDLE / "local_results/frozen/timestamp_audit_cache.json",
            "panel_source": BUNDLE / "local_results/frozen/k3_panel_source.json",
        }
        for label, path in explicit.items():
            if sha(path) != frozen[label]: errors.append(f"explicit frozen manifest mismatch: {label}")
        seed_manifest = json.loads((BUNDLE / "configs/seed_manifest.json").read_text(encoding="utf-8"))
        if seed_manifest.get("training_seeds") != [21001, 21002, 21003, 21004, 21005]: errors.append("seed manifest mismatch")
    requirements = json.loads((BUNDLE / "manifests" / "DATA_REQUIREMENTS.json").read_text(encoding="utf-8"))["datasets"]
    data_ready = True
    for row in requirements:
        root = BUNDLE / row["required_relative_path"]
        for name in row["expected_file_names"]:
            path = root / name
            if not path.is_file():
                data_ready = False; warnings.append(f"canonical server input absent: {path.relative_to(BUNDLE)}")
            elif name in row["expected_sha256_if_available"] and sha(path) != row["expected_sha256_if_available"][name]:
                errors.append(f"canonical data hash mismatch: {path.relative_to(BUNDLE)}")
    project = BUNDLE / "src" / "project"
    sys.path.insert(0, str(project)); sys.path.insert(0, str(project / "scripts"))
    for module in ("scripts.run_k3_spatial_order_task_core_v1", "run_strict_timestamp_chronological_core_v1", "run_strict_timestamp_t2_mlp_local_v1"):
        try: importlib.import_module(module)
        except Exception as exc: errors.append(f"import failed: {module}: {exc!r}")
    cases = json.loads((BUNDLE / "manifests" / "SERVER_REQUIRED_CASES.json").read_text(encoding="utf-8"))["cases"]
    ids = [row["case_id"] for row in cases]
    expected = 2 * 6 * 5
    if len(ids) != expected or len(set(ids)) != expected: errors.append(f"server case manifest expected {expected}, got {len(ids)}/{len(set(ids))} unique")
    if any(row["dataset"] not in ("tags-ask-ubuntu", "threads-ask-ubuntu") for row in cases): errors.append("unexpected dataset in server queue")
    if not data_ready and not args.allow_missing_data: errors.append("canonical data incomplete; STOP_BEFORE_SCIENTIFIC_RUN")
    valid = not errors
    report = ["# Server Bundle Validation", "", f"- BUNDLE_VALID = {'YES' if valid else 'NO'}", f"- DATA_READY = {'YES' if data_ready else 'NO'}", f"- DRY_RUN = {'YES' if args.dry_run else 'NO'}", f"- SERVER_CASES_RESOLVED = {len(ids)}", f"- PYTHON = {platform.python_version()}", f"- PLATFORM = {platform.platform()}", "", "## Errors", ""]
    report.extend([f"- {value}" for value in errors] or ["- none"])
    report += ["", "## Warnings", ""]
    report.extend([f"- {value}" for value in warnings] or ["- none"])
    (BUNDLE / "SERVER_BUNDLE_VALIDATION.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"BUNDLE_VALID={'YES' if valid else 'NO'} DATA_READY={'YES' if data_ready else 'NO'} CASES={len(ids)}")
    if not valid: raise SystemExit(2)

if __name__ == "__main__":
    main()
'''


MASTER = r'''
from __future__ import annotations
import json, os, platform, subprocess, sys
from pathlib import Path

BUNDLE = Path(__file__).resolve().parent
LOG = BUNDLE / "logs" / "run_all_server.log"

def run(args: list[str]) -> None:
    print("+", " ".join(args), flush=True)
    with LOG.open("a", encoding="utf-8") as handle:
        subprocess.run(args, cwd=BUNDLE, check=True, stdout=handle, stderr=subprocess.STDOUT)

def main() -> None:
    (BUNDLE / "logs").mkdir(exist_ok=True); (BUNDLE / "server_results").mkdir(exist_ok=True)
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE"
        memory = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
        try:
            pages = os.sysconf("SC_PHYS_PAGES"); page_size = os.sysconf("SC_PAGE_SIZE"); ram = pages * page_size
        except (AttributeError, ValueError):
            try:
                import psutil
                ram = psutil.virtual_memory().total
            except Exception: ram = "UNKNOWN"
        print(f"Python={platform.python_version()} PyTorch={torch.__version__} CUDA={torch.cuda.is_available()} GPU={gpu} GPU_BYTES={memory} CPU={os.cpu_count()} RAM_BYTES={ram}")
    except Exception as exc:
        raise SystemExit(f"STOP_BEFORE_SCIENTIFIC_RUN: PyTorch environment unavailable: {exc!r}")
    py = sys.executable
    run([py, "scripts/validate_bundle.py", "--dry-run"])
    run([py, "scripts/run_k3_remaining.py"])
    run([py, "scripts/run_k4_retention_remaining.py"])
    run([py, "scripts/run_k4_saturation_remaining.py"])
    run([py, "scripts/run_k4_r1_remaining.py"])
    run([py, "scripts/merge_results.py"])
    print("ALL_SERVER_STAGES_COMPLETE")

if __name__ == "__main__":
    main()
'''


README = r'''
# Task-Core Confirmatory v2 Server Bundle

This handoff contains exactly the unfinished formal cases. It does not contain raw datasets and does not rerun locally completed K3/K4 cases.

## Required data placement

Before starting, place the three canonical raw files for each queued dataset at:

- `dataset/raw_data/tags-ask-ubuntu/`
- `dataset/raw_data/threads-ask-ubuntu/`

Exact filenames, schemas, and known SHA-256 values are in `manifests/DATA_REQUIREMENTS.json`. Never substitute another dataset. A missing or mismatched file stops before scientific execution.

## One command

```bash
cd task_core_server_bundle_v2
bash RUN_ALL_SERVER.sh
```

Equivalent: `python run_all_server.py`.

The master validates the environment/data/hashes, runs only the two pending K3 datasets with B0-B5 and seeds 21001-21005, performs the frozen 2,000-replicate bootstrap, and merges the three locally completed datasets into the final five-dataset outputs. K4 retention and saturation are already complete locally. All K4 minimum cores are Full, so K4 R1 is formally `NO_NONTRIVIAL_PERIPHERY_AFTER_SATURATION` and has no queued evaluations.

Execution is resumable: per-case model/prediction markers are stored below `server_results/k3_remaining/local/`, and reruns use `SKIP_COMPLETED=YES` behavior.
'''


def build(replace: bool = False) -> None:
    if BUNDLE.exists() or ZIP_PATH.exists():
        if not replace:
            raise RuntimeError("target bundle already exists; rerun with --replace to rebuild the exact generated handoff")
        if BUNDLE.parent.resolve() != ROOT.resolve() or BUNDLE.name != "task_core_server_bundle_v2":
            raise RuntimeError("refusing to replace an unexpected directory")
        if BUNDLE.exists(): shutil.rmtree(BUNDLE)
        if ZIP_PATH.exists(): ZIP_PATH.unlink()
    for directory in ("configs", "manifests", "scripts", "src/project", "local_results/frozen/k3_old", "local_results/completed", "logs", "server_results"):
        (BUNDLE / directory).mkdir(parents=True, exist_ok=True)

    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    git_status = subprocess.run(["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True).stdout.splitlines()
    write_json(RESULTS / "GIT_STATE_AT_CONFIRMATORY_V2.json", {"git_commit": git_commit, "git_status": git_status})

    copied_modules = copy_import_closure([
        "scripts.run_k3_spatial_order_task_core_v1",
        "scripts.task_core_common_v1",
        "run_k3_residual_lower_order_conditional_redundancy",
        "run_strict_timestamp_chronological_core_v1",
        "run_strict_timestamp_t2_mlp_local_v1",
    ])

    frozen_files = {
        ROOT / "results/k3_all_dataset_learned_smoke/K3_SMOKE_DATASET_MANIFEST.json": BUNDLE / "local_results/frozen/k3_panel_source.json",
        ROOT / "results/strict_timestamp_chronological_core_v1/01_timestamp_audit/audit_cache.json": BUNDLE / "local_results/frozen/timestamp_audit_cache.json",
        ROOT / "results/strict_timestamp_t2_remaining_server_v1/02_k3/tags-ask-ubuntu/T2/case_protocol.json": BUNDLE / "local_results/frozen/tags_ask_protocol.json",
        K3_OLD / "K3_TASK_CORE_FEATURE_MANIFEST.json": BUNDLE / "local_results/frozen/k3_old/K3_TASK_CORE_FEATURE_MANIFEST.json",
        K3_OLD / "K3_FEATURE_PROVENANCE_AUDIT.csv": BUNDLE / "local_results/frozen/k3_old/K3_FEATURE_PROVENANCE_AUDIT.csv",
        K3_OLD / "K3_FEATURE_PROVENANCE_AUDIT.json": BUNDLE / "local_results/frozen/k3_old/K3_FEATURE_PROVENANCE_AUDIT.json",
        K3_OLD / "K3_TASK_CORE_PREDICTIONS.parquet": BUNDLE / "local_results/frozen/k3_old/K3_TASK_CORE_PREDICTIONS.parquet",
        K3_OLD / "K3_L2_CONSTITUENT_PREDICTIONS.parquet": BUNDLE / "local_results/frozen/k3_old/K3_L2_CONSTITUENT_PREDICTIONS.parquet",
    }
    for source, target in frozen_files.items(): copy_file(source, target)

    shutil.copytree(RESULTS / "k3_final_5dataset", BUNDLE / "local_results/completed/k3_final_5dataset", dirs_exist_ok=True)
    k4_target = BUNDLE / "local_results/completed/k4_downward_core"
    k4_target.mkdir(parents=True, exist_ok=True)
    for source in (RESULTS / "k4_downward_core").iterdir():
        if source.suffix.lower() in {".csv", ".json", ".md"}: copy_file(source, k4_target / source.name)
    copy_file(RESULTS / "TASK_CORE_CONFIRMATORY_V2_LOCAL_FINAL_REPORT.md", BUNDLE / "local_results/completed/TASK_CORE_CONFIRMATORY_V2_LOCAL_FINAL_REPORT.md")

    source_feature = BUNDLE / "local_results/frozen/k3_old/K3_TASK_CORE_FEATURE_MANIFEST.json"
    source_panel = BUNDLE / "local_results/frozen/k3_panel_source.json"
    source_audit = BUNDLE / "local_results/frozen/timestamp_audit_cache.json"
    config = {
        "protocol": "TASK_CORE_CONFIRMATORY_V2", "strict_T2": "YES", "full_data": "YES", "full_candidate_universe": "YES",
        "sampling": "NONE", "primary_endpoint": "LogLoss", "noninferiority_margin_LL": 0.02,
        "model": "frozen existing MLP", "training_seeds": SEEDS, "bootstrap_replicates": 2000,
        "model_search": "NO", "hyperparameter_search": "NO", "seed_search": "NO", "skip_completed": "YES",
        "git_commit": git_commit, "copied_project_modules": copied_modules,
        "frozen_hashes": {
            "feature_manifest": sha(source_feature), "split_audit_cache": sha(source_audit), "panel_source": sha(source_panel),
        },
    }
    write_json(BUNDLE / "configs/experiment_config.json", config)
    write_json(BUNDLE / "configs/k3_panel_manifest.json", {"panel_size": 5, "datasets": K3_PANEL, "excluded": ["email-Enron", "contact-high-school", "contact-primary-school"], "server_remaining": K3_PENDING})
    write_json(BUNDLE / "configs/k4_panel_manifest.json", {"panel_size": 3, "datasets": K4_PANEL, "local_retention_complete": K4_PANEL, "local_saturation_complete": K4_PANEL, "server_remaining": []})
    write_json(BUNDLE / "configs/feature_manifest.json", {"K3_models": MODELS, "frozen_source": "local_results/frozen/k3_old/K3_TASK_CORE_FEATURE_MANIFEST.json", "source_sha256": sha(source_feature), "K4_models": ["C0", "C1", "C2", "C3", "Full"], "K4_server_cases": []})
    write_json(BUNDLE / "configs/seed_manifest.json", {"training_seeds": SEEDS, "seed_search": "NO"})
    write_json(BUNDLE / "configs/bootstrap_config.json", {"primary_endpoint": "LogLoss", "replicates": 2000, "paired_by_candidate": "YES", "margin": 0.02})
    write_json(BUNDLE / "configs/r1_config.json", {"doses": [0.20, 0.40, 0.60, 0.80, 0.95], "structural_seeds_per_dose": 10, "training_seeds": SEEDS, "retrain": "NO", "refit_scaler": "NO", "K4_status": {dataset: "NO_NONTRIVIAL_PERIPHERY_AFTER_SATURATION" for dataset in K4_PANEL}, "server_cases": []})

    cases = [{"case_id": f"K3|{dataset}|{model}|{seed}", "experiment": "K3_SATURATION", "dataset": dataset, "model": model, "training_seed": seed, "status": "SERVER_REQUIRED", "dependency_stage": 1} for dataset in K3_PENDING for model in MODELS for seed in SEEDS]
    write_json(BUNDLE / "manifests/SERVER_REQUIRED_CASES.json", {"case_count": len(cases), "dataset_packages": len(K3_PENDING), "cases": cases, "K4_server_case_count": 0})
    write_json(BUNDLE / "manifests/DATA_REQUIREMENTS.json", {"raw_data_in_zip": "NO", "datasets": [data_requirement(dataset) for dataset in K3_PENDING]})

    local_rows = []
    for dataset in K3_PANEL[:3]:
        for model in MODELS:
            for seed in SEEDS: local_rows.append(["K3_SATURATION", dataset, model, seed, "COMPLETED_LOCAL"])
    for dataset in K4_PANEL:
        local_rows.append(["K4_RETENTION_TRAIN", dataset, "NA", "NA", "COMPLETED_LOCAL"])
        local_rows.append(["K4_RETENTION_EVAL", dataset, "NA", "NA", "COMPLETED_LOCAL"])
        for model in ["C0", "C1", "C2", "C3", "Full"]:
            for seed in SEEDS: local_rows.append(["K4_SATURATION", dataset, model, seed, "COMPLETED_LOCAL"])
        local_rows.append(["K4_R1", dataset, "Full", "NA", "NO_NONTRIVIAL_PERIPHERY_AFTER_SATURATION"])
    with (BUNDLE / "manifests/LOCAL_COMPLETED_CASES.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["experiment", "dataset", "model", "seed", "status"]); writer.writerows(local_rows)

    write_text(BUNDLE / "scripts/run_k3_remaining.py", RUN_K3)
    write_text(BUNDLE / "scripts/run_k4_retention_remaining.py", NOOP_TEMPLATE.format(stage="K4_RETENTION"))
    write_text(BUNDLE / "scripts/run_k4_saturation_remaining.py", NOOP_TEMPLATE.format(stage="K4_SATURATION"))
    write_text(BUNDLE / "scripts/run_k4_r1_remaining.py", NOOP_TEMPLATE.format(stage="K4_R1"))
    write_text(BUNDLE / "scripts/merge_results.py", MERGE)
    write_text(BUNDLE / "scripts/validate_bundle.py", VALIDATE)
    write_text(BUNDLE / "run_all_server.py", MASTER)
    write_text(BUNDLE / "RUN_ALL_SERVER.sh", "#!/usr/bin/env bash\nset -euo pipefail\npython3 run_all_server.py \"$@\"\n")
    write_text(BUNDLE / "README_SERVER.md", README)

    ignored_prefixes = ("logs/", "server_results/", "dataset/")
    hashes = {}
    for path in sorted(BUNDLE.rglob("*")):
        relative = path.relative_to(BUNDLE).as_posix()
        if path.is_file() and relative != "manifests/HASHES.json" and relative != "SERVER_BUNDLE_VALIDATION.md" and not relative.startswith(ignored_prefixes):
            hashes[relative] = sha(path)
    write_json(BUNDLE / "manifests/HASHES.json", {"algorithm": "SHA256", "file_count": len(hashes), "files": hashes})

    def make_zip() -> None:
        with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in sorted(BUNDLE.rglob("*")):
                if path.is_file(): archive.write(path, Path(BUNDLE.name) / path.relative_to(BUNDLE))

    make_zip()
    with tempfile.TemporaryDirectory(prefix="task_core_bundle_validate_") as temp_name:
        temp = Path(temp_name)
        with zipfile.ZipFile(ZIP_PATH) as archive: archive.extractall(temp)
        extracted = temp / BUNDLE.name
        result = subprocess.run([sys.executable, str(extracted / "scripts/validate_bundle.py"), "--dry-run", "--allow-missing-data"], cwd=extracted, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"extracted bundle validation failed:\n{result.stdout}\n{result.stderr}")
        copy_file(extracted / "SERVER_BUNDLE_VALIDATION.md", BUNDLE / "SERVER_BUNDLE_VALIDATION.md")
    make_zip()
    # Validate the final archive once more after embedding the validation report.
    with tempfile.TemporaryDirectory(prefix="task_core_bundle_final_") as temp_name:
        temp = Path(temp_name)
        with zipfile.ZipFile(ZIP_PATH) as archive: archive.extractall(temp)
        extracted = temp / BUNDLE.name
        result = subprocess.run([sys.executable, str(extracted / "scripts/validate_bundle.py"), "--dry-run", "--allow-missing-data"], cwd=extracted, capture_output=True, text=True)
        if result.returncode != 0: raise RuntimeError(result.stdout + result.stderr)
        copy_file(extracted / "SERVER_BUNDLE_VALIDATION.md", BUNDLE / "SERVER_BUNDLE_VALIDATION.md")
    print(json.dumps({"bundle": str(BUNDLE), "zip": str(ZIP_PATH), "server_cases": len(cases), "local_case_records": len(local_rows), "copied_modules": len(copied_modules)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--replace", action="store_true", help="replace only the exact generated v2 bundle and zip")
    build(parser.parse_args().replace)
