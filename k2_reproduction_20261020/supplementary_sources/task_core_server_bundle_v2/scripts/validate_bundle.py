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
    if any(row["dataset"] not in ("tags-ask-ubuntu", "threads-math-sx") for row in cases): errors.append("unexpected dataset in server queue")
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
