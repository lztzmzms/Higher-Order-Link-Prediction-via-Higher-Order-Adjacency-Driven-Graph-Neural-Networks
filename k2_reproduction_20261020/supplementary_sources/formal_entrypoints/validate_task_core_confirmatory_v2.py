"""Independent result-shape, identity, and bundle integrity checks for v2."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "task_core_confirmatory_v2"
K3 = OUT / "k3_final_5dataset"
K4 = OUT / "k4_downward_core"
BUNDLE = ROOT / "task_core_server_bundle_v2"
ZIP = ROOT / "task_core_server_bundle_v2.zip"
K3_PANEL = {"email-Eu", "NDC-substances", "tags-math-sx", "tags-ask-ubuntu", "threads-ask-ubuntu"}
K4_PANEL = {"email-Eu", "tags-ask-ubuntu", "tags-math-sx"}


def check(condition: bool, name: str, detail, rows: list[dict]) -> None:
    rows.append({"check": name, "status": "PASS" if condition else "FAIL", "detail": str(detail)})


def main() -> None:
    rows: list[dict] = []
    panel = pd.read_csv(K3 / "K3_FINAL_5DATASET_PANEL_MANIFEST.csv")
    check(len(panel) == 5 and set(panel.dataset) == K3_PANEL, "K3_PANEL_EXACT", list(panel.dataset), rows)
    check(not set(panel.dataset) & {"email-Enron", "contact-high-school", "contact-primary-school"}, "K3_EXCLUSIONS_EXACT", "no excluded dataset", rows)
    k3_seed = pd.read_csv(K3 / "K3_FINAL_5DATASET_SATURATION_BY_SEED.csv")
    check(len(k3_seed) == 90 and k3_seed[["dataset", "model", "seed"]].drop_duplicates().shape[0] == 90, "K3_LOCAL_COMPLETED_CASES", len(k3_seed), rows)

    k4_panel = pd.read_csv(K4 / "K4_FINAL_PANEL_MANIFEST.csv")
    eligible = (k4_panel[["train_candidates", "eval_candidates", "train_positives", "eval_positives"]].to_numpy() > 0).all()
    check(len(k4_panel) == 3 and set(k4_panel.dataset) == K4_PANEL and eligible, "K4_PANEL_ELIGIBLE_EXACT", len(k4_panel), rows)

    retention = pd.read_parquet(K4 / "K4_MULTIORDER_RETENTION_AUDIT_BY_CANDIDATE.parquet")
    expected_candidates = int(k4_panel.train_candidates.sum() + k4_panel.eval_candidates.sum())
    candidate_rows = retention[retention.scope == "candidate_local"] if "scope" in retention.columns else retention[retention.candidate_id.notna()]
    check(len(candidate_rows) == expected_candidates, "K4_RETENTION_CANDIDATE_UNIVERSE", f"{len(candidate_rows)}/{expected_candidates}", rows)
    ratios = [column for column in candidate_rows.columns if column.startswith("local_") and column.endswith("_retention")]
    ratio_ok = all(np.isfinite(candidate_rows[column]).all() and candidate_rows[column].between(0, 1).all() for column in ratios)
    check(ratio_ok, "K4_RETENTION_RATIOS_FINITE_BOUNDED", ratios, rows)

    by_seed = pd.read_csv(K4 / "K4_TASK_CORE_SATURATION_BY_SEED.csv")
    unique_cases = by_seed[["dataset", "model", "seed"]].drop_duplicates()
    check(len(by_seed) == 75 and len(unique_cases) == 75, "K4_SATURATION_CASES_EXACT", len(by_seed), rows)
    predictions = pd.read_parquet(K4 / "K4_TASK_CORE_SATURATION_PREDICTIONS.parquet")
    expected_predictions = int(k4_panel.eval_candidates.sum()) * 5 * 5
    unique_predictions = predictions[["dataset", "candidate_id", "model", "seed"]].drop_duplicates()
    check(len(predictions) == expected_predictions and len(unique_predictions) == expected_predictions, "K4_PREDICTION_ROWS_EXACT", f"{len(predictions)}/{expected_predictions}", rows)
    label_counts = predictions.groupby(["dataset", "candidate_id"]).y_true.nunique()
    check((label_counts == 1).all(), "K4_LABEL_ALIGNMENT_ACROSS_MODELS_SEEDS", int(label_counts.max()), rows)
    check(np.isfinite(predictions.p_pred).all() and predictions.p_pred.between(0, 1).all(), "K4_PREDICTIONS_FINITE_BOUNDED", len(predictions), rows)

    bootstrap = pd.read_csv(K4 / "K4_TASK_CORE_SATURATION_BOOTSTRAP.csv")
    check(len(bootstrap) == 12 and (bootstrap.bootstrap_replicates == 2000).all(), "K4_BOOTSTRAP_EXACT", len(bootstrap), rows)
    core = json.loads((K4 / "K4_FROZEN_SATURATION_CORE_MANIFEST.json").read_text(encoding="utf-8"))
    full = all(core["datasets"][dataset]["minimum_core"] == "Full" for dataset in K4_PANEL)
    check(full, "K4_MINIMUM_CORES_FROZEN_FULL", [core["datasets"][x]["minimum_core"] for x in sorted(K4_PANEL)], rows)
    r1 = pd.read_csv(K4 / "K4_NEW_CORE_R1_LOCAL_STATUS.csv")
    check(len(r1) == 3 and (r1.status == "NO_NONTRIVIAL_PERIPHERY_AFTER_SATURATION").all(), "K4_R1_STRUCTURALLY_INAPPLICABLE", len(r1), rows)

    feature_hashes = {}
    feature_ok = True
    for path in sorted((OUT / "local/k4_features").glob("*/*.npy")):
        array = np.load(path, allow_pickle=False)
        feature_ok &= np.isfinite(array).all()
        feature_hashes[path.relative_to(OUT).as_posix()] = hashlib.sha256(array.tobytes()).hexdigest()
    check(feature_ok and len(feature_hashes) == 30, "K4_FEATURE_MATRICES_FINITE", len(feature_hashes), rows)

    required = json.loads((BUNDLE / "manifests/SERVER_REQUIRED_CASES.json").read_text(encoding="utf-8"))
    ids = [value["case_id"] for value in required["cases"]]
    check(len(ids) == 60 and len(set(ids)) == 60 and required["K4_server_case_count"] == 0, "SERVER_QUEUE_EXACT", f"{len(ids)} K3; 0 K4", rows)
    validation = (BUNDLE / "SERVER_BUNDLE_VALIDATION.md").read_text(encoding="utf-8")
    check("BUNDLE_VALID = YES" in validation and "SERVER_CASES_RESOLVED = 60" in validation, "EXTRACTED_BUNDLE_VALID", "YES", rows)
    with zipfile.ZipFile(ZIP) as archive:
        members = archive.namelist()
        only_root = all(name.startswith("task_core_server_bundle_v2/") for name in members)
        no_forbidden = not any("/.git/" in name or "/dataset/raw_data/" in name or "__pycache__" in name for name in members)
        bad_absolute = []
        for name in members:
            if name.endswith((".py", ".json", ".md", ".sh", ".csv")):
                try: text = archive.read(name).decode("utf-8")
                except UnicodeDecodeError: continue
                if "C:/Users/13307" in text or "C:\\Users\\13307" in text: bad_absolute.append(name)
    check(only_root and no_forbidden and not bad_absolute, "ZIP_SCOPE_AND_PORTABILITY", f"members={len(members)}, absolute_paths={bad_absolute}", rows)

    failures = [row for row in rows if row["status"] != "PASS"]
    payload = {"protocol": "TASK_CORE_CONFIRMATORY_V2", "RESULTS_VALID": "YES" if not failures else "NO", "checks": rows, "feature_matrix_hashes": feature_hashes}
    (OUT / "RESULTS_INTEGRITY_AUDIT.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Task-Core Confirmatory v2 Integrity Audit", "", f"- RESULTS_VALID = {'YES' if not failures else 'NO'}", f"- checks passed = {len(rows)-len(failures)}/{len(rows)}", "", "| check | status | detail |", "|---|---|---|"]
    lines.extend(f"| {row['check']} | {row['status']} | {row['detail']} |" for row in rows)
    (OUT / "RESULTS_INTEGRITY_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if failures: raise SystemExit("; ".join(row["check"] for row in failures))
    print(f"RESULTS_VALID=YES checks={len(rows)}")


if __name__ == "__main__":
    main()
