#!/usr/bin/env python3
"""Fast, training-free integrity checks for the fixed server release."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "protocol_data"
DATASETS = [
    "email-Enron", "contact-high-school", "contact-primary-school", "email-Eu",
    "NDC-substances", "tags-ask-ubuntu", "tags-math-sx", "threads-ask-ubuntu",
]
FULL_DATASETS = {"email-Enron", "contact-high-school", "contact-primary-school"}
SAMPLE_BUDGETS = {
    "email-Eu": 200,
    "NDC-substances": 200,
    "tags-ask-ubuntu": 40,
    "tags-math-sx": 40,
    "threads-ask-ubuntu": 600,
}


def triangles(path: Path):
    return [
        tuple(sorted(map(int, line.split())))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    config = json.loads((ROOT / "configs" / "model_selection.json").read_text(encoding="utf-8"))
    assert config["model_seeds"] == [41, 42, 43]
    dual = config["dual"]["email-Enron"]
    assert dual["graph_triangle_scope"] == "legacy_closed_open"
    assert dual["prior"] == "recent_activity"
    assert dual["temporal_activity_alpha"] == 0.90
    assert dual["validation_select_temporal_alpha"] is False
    mlshol = config["mlshol"]["email-Enron"]
    assert mlshol["rw_source"] == "dual_mapping_full"
    assert mlshol["observed_2_skeleton"] is True
    assert mlshol["strength_edge"] is True
    assert mlshol["activity_prior"] is True
    assert mlshol["temporal_activity_alpha"] == 0.90

    run_manifest = json.loads((DATA_ROOT / "run_manifest.json").read_text(encoding="utf-8"))
    rows = {row["dataset"]: row for row in run_manifest}
    assert set(rows) == set(DATASETS)
    sample_manifest = json.loads((ROOT / "sampling_manifest.json").read_text(encoding="utf-8"))
    sample_rows = {row["dataset"]: row for row in sample_manifest["datasets"]}

    for dataset in DATASETS:
        assert rows[dataset]["reference_match"] is True
        protocol = json.loads(
            (DATA_ROOT / "shared_splits" / dataset / "protocol_manifest.json").read_text(encoding="utf-8")
        )
        assert protocol["reference_match"] is True
        sampled = bool(protocol["sampling"]["sampled"])
        assert sampled is (dataset not in FULL_DATASETS)
        if sampled:
            assert sample_rows[dataset]["selected_node_count"] == SAMPLE_BUDGETS[dataset]

        train_dir = DATA_ROOT / "dataset" / "train"
        opened = triangles(train_dir / f"{dataset}_openTriangles_train.txt")
        positives = set(triangles(train_dir / f"{dataset}_new_closures_train.txt"))
        split = np.load(DATA_ROOT / "shared_splits" / dataset / "fit_val_indices.npz")
        labels = np.asarray(split["train_labels"], dtype=np.int64)
        rebuilt = np.asarray([int(item in positives) for item in opened], dtype=np.int64)
        assert np.array_equal(rebuilt, labels)
        fit = np.asarray(split["fit_indices"], dtype=np.int64)
        val = np.asarray(split["val_indices"], dtype=np.int64)
        assert np.intersect1d(fit, val).size == 0
        assert np.array_equal(np.sort(np.concatenate([fit, val])), np.arange(len(labels)))
        assert np.unique(labels[fit]).size == 2
        assert np.unique(labels[val]).size == 2
        print(
            f"OK {dataset}: sampled={sampled} train={len(labels)} "
            f"fit_pos={int(labels[fit].sum())} val_pos={int(labels[val].sum())} "
            f"test={protocol['test_candidate_count']}/{protocol['test_positive_count']}"
        )

    print("PRECHECK PASSED: fixed protocol, labels, splits, samples, and email-Enron configs")


if __name__ == "__main__":
    main()
