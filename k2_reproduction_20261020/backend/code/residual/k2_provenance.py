"""Strict provenance helpers for leakage-sensitive K=2 feature caches."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path


PROVENANCE_VERSION = "k2_maxsimplex_provenance_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_int_lines(path: Path):
    return [int(line.strip()) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _raw_paths(raw_dir: Path, dataset: str):
    raw_dir = Path(raw_dir)
    return {
        "nverts": raw_dir / f"{dataset}-nverts.txt",
        "simplices": raw_dir / f"{dataset}-simplices.txt",
        "times": raw_dir / f"{dataset}-times.txt",
    }


def _load_events(raw_dir: Path, dataset: str, event_order: str):
    paths = _raw_paths(raw_dir, dataset)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing raw provenance inputs: " + ", ".join(missing))
    counts = _read_int_lines(paths["nverts"])
    flat = _read_int_lines(paths["simplices"])
    times = _read_int_lines(paths["times"])
    if len(counts) != len(times) or sum(counts) != len(flat):
        raise ValueError(f"Malformed raw provenance inputs for {dataset}")
    events = []
    offset = 0
    for row_id, (count, timestamp) in enumerate(zip(counts, times)):
        vertices = tuple(sorted(set(flat[offset : offset + count])))
        offset += count
        if vertices:
            events.append({"time": int(timestamp), "row_id": int(row_id), "vertices": vertices})
    if event_order == "timestamp":
        events.sort(key=lambda event: (event["time"], event["row_id"]))
    elif event_order != "raw":
        raise ValueError(f"Unsupported event_order={event_order!r}")
    return paths, events


def _hash_events(events) -> str:
    digest = hashlib.sha256()
    for event in events:
        digest.update(
            (
                f"{event['time']}|{event['row_id']}|"
                + ",".join(str(node) for node in event["vertices"])
                + "\n"
            ).encode("utf-8")
        )
    return digest.hexdigest()


def gpickle_triplets(gpickle_path: Path):
    with Path(gpickle_path).open("rb") as handle:
        graph = pickle.load(handle)
    triplets = []
    for _, data in graph.nodes(data=True):
        triangle = tuple(sorted(int(value) for value in data["triplet"]))
        if len(triangle) != 3 or len(set(triangle)) != 3:
            raise ValueError(f"Invalid triplet in {gpickle_path}: {data.get('triplet')!r}")
        triplets.append(triangle)
    return triplets


def _hash_triplets(triplets) -> str:
    digest = hashlib.sha256()
    for triangle in triplets:
        digest.update(("|".join(str(value) for value in triangle) + "\n").encode("ascii"))
    return digest.hexdigest()


def build_feature_provenance(
    raw_dir: Path,
    dataset: str,
    event_order: str,
    context_ratio: float,
    gpickle_path: Path,
    feature_header=None,
):
    paths, events = _load_events(raw_dir, dataset, event_order)
    ratio = float(context_ratio)
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"context_ratio must be in [0, 1], got {ratio}")
    context_end = int(ratio * len(events))
    context = events[:context_end]
    triplets = gpickle_triplets(gpickle_path)
    raw_hashes = {name: sha256_file(path) for name, path in paths.items()}
    raw_data_hash = hashlib.sha256(
        json.dumps(raw_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "provenance_version": PROVENANCE_VERSION,
        "dataset": str(dataset),
        "event_order": str(event_order),
        "context_ratio": ratio,
        "raw_data_hash": raw_data_hash,
        "raw_file_sha256": raw_hashes,
        "event_hash": _hash_events(events),
        "context_event_count": int(len(context)),
        "context_max_event_index": int(max((event["row_id"] for event in context), default=-1)),
        "context_max_timestamp": int(max((event["time"] for event in context), default=-1)),
        "gpickle_sha256": sha256_file(gpickle_path),
        "candidate_hash": _hash_triplets(triplets),
        "candidate_count": int(len(triplets)),
        "generator_version": PROVENANCE_VERSION,
        "feature_header": list(feature_header or []),
    }


def sidecar_path(feature_path: Path) -> Path:
    return Path(str(feature_path) + ".meta.json")


def write_feature_provenance(feature_path: Path, provenance: dict) -> Path:
    feature_path = Path(feature_path)
    payload = dict(provenance)
    payload["feature_sha256"] = sha256_file(feature_path)
    target = sidecar_path(feature_path)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def validate_feature_provenance(
    feature_path: Path,
    raw_dir: Path,
    dataset: str,
    event_order: str,
    context_ratio: float,
    gpickle_path: Path,
    feature_header=None,
):
    feature_path = Path(feature_path)
    metadata_path = sidecar_path(feature_path)
    if not metadata_path.is_file():
        raise ValueError(
            f"Missing maximal-simplex provenance sidecar for {feature_path}; "
            "refusing to load an unverified cache."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = build_feature_provenance(
        raw_dir,
        dataset,
        event_order,
        context_ratio,
        gpickle_path,
        feature_header=feature_header,
    )
    expected["feature_sha256"] = sha256_file(feature_path)
    keys = [
        "provenance_version",
        "dataset",
        "event_order",
        "context_ratio",
        "raw_data_hash",
        "event_hash",
        "context_event_count",
        "context_max_event_index",
        "context_max_timestamp",
        "gpickle_sha256",
        "candidate_hash",
        "candidate_count",
        "generator_version",
    ]
    if feature_header is not None:
        keys.append("feature_header")
    mismatches = {
        key: {"cached": metadata.get(key), "expected": expected.get(key)}
        for key in keys
        if metadata.get(key) != expected.get(key)
    }
    if metadata.get("feature_sha256") != expected["feature_sha256"]:
        mismatches["feature_sha256"] = {
            "cached": metadata.get("feature_sha256"),
            "expected": expected["feature_sha256"],
        }
    if mismatches:
        raise ValueError(
            "Maximal-simplex provenance mismatch for "
            f"{feature_path}: {json.dumps(mismatches, ensure_ascii=False, sort_keys=True)}"
        )
    return metadata
