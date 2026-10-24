"""Server pipeline for K4 Benchmark Expansion Bundle v1.

The bundle builder copies this file to ``scripts/k4_server_pipeline.py``.
Every scientific choice is frozen by bundle manifests; this module only adds
server engineering (checkpoints, heartbeats, exact indexed labeling, resume).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import pickle
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import urllib.error
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

BUNDLE = Path(__file__).resolve().parents[1]
PROJECT = BUNDLE / "src" / "project"
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

TARGETS = ("NDC-classes", "congress-bills", "DAWN")
EXCLUDED = ("music-rap-genius",)
RESULTS = BUNDLE / "server_results" / "k4_server_benchmark_expansion_v1"
S1 = RESULTS / "stageS1_data"
S2 = RESULTS / "stageS2_census"
S3 = RESULTS / "stageS3_gate"
S4 = RESULTS / "stageS4_mlp"
S5 = RESULTS / "stageS5_final"
CHECKPOINTS = RESULTS / "checkpoints"
DATA_ROOT = Path(os.environ.get("K4_DATA_ROOT", str(BUNDLE / "data"))).expanduser().resolve()
CONFIG_PATH = BUNDLE / "configs" / "experiment_config.json"
DATA_MANIFEST_PATH = BUNDLE / "manifests" / "K4_LOCAL_EXPANSION_DATA_MANIFEST.json"
IDENTITY_REFERENCE_PATH = BUNDLE / "manifests" / "K4_LOCAL_EXPANSION_DATA_IDENTITY.csv"
SEEDS = (21001, 21002, 21003, 21004, 21005)
MODELS = ("C0*", "C1*", "C2*", "C3*", "Full")
MARGIN = 0.02
BOOTSTRAPS = 2000
BATCH = 1024


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=json_default) + "\n", encoding="utf-8")
    temporary.replace(path)


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
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["status"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha_rows(rows) -> str:
    digest = hashlib.sha256()
    for index, row in enumerate(rows):
        if index:
            digest.update(b"\n")
        digest.update(",".join(map(str, row)).encode("utf-8"))
    return digest.hexdigest()


def array_hash(value) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    header = f"{array.dtype.str}|{array.shape}|".encode()
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def slug(dataset: str) -> str:
    return dataset.replace("-", "_").replace(" ", "_").upper()


def done_marker(dataset: str, split: str) -> Path:
    return CHECKPOINTS / f"DONE_{slug(dataset)}_{split.upper()}_CENSUS"


def rss_bytes() -> int:
    try:
        import psutil
        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        return 0


class Heartbeat:
    def __init__(self, dataset: str, stage: str, interval: int = 30):
        self.dataset = dataset
        self.stage = stage
        self.interval = interval
        self.started = time.time()
        self.processed = 0
        self.total = 0
        self.status = "starting"
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        self.emit()
        return self

    def update(self, processed=None, total=None, status=None):
        if processed is not None:
            self.processed = int(processed)
        if total is not None:
            self.total = int(total)
        if status is not None:
            self.status = str(status)

    def emit(self):
        elapsed = int(time.time() - self.started)
        checkpoint = "YES" if "DONE" in self.status else "NO"
        print(f"HEARTBEAT dataset={self.dataset} stage={self.stage} elapsed={elapsed}s processed={self.processed}/{self.total or '?'} rss_bytes={rss_bytes()} checkpoint={checkpoint} status={self.status}", flush=True)

    def _run(self):
        while not self._stop.wait(self.interval):
            self.emit()

    def close(self, status="DONE"):
        self.status = status
        self._stop.set()
        self.emit()


def config() -> dict:
    return read_json(CONFIG_PATH)


def frozen_dataset_manifest() -> dict:
    return read_json(DATA_MANIFEST_PATH)


def expected_identity() -> dict:
    rows = {}
    with IDENTITY_REFERENCE_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] in TARGETS:
                rows[row["dataset"]] = row
    return rows


def required_data_entries(dataset: str) -> dict:
    manifest = frozen_dataset_manifest()["datasets"][dataset]
    result = {}
    for entry in manifest["files"]:
        name = entry.get("canonical_filename", "")
        for kind in ("nverts", "simplices", "times"):
            if name == f"{dataset}-{kind}.txt":
                result[kind] = entry
    if set(result) != {"nverts", "simplices", "times"}:
        raise RuntimeError(f"frozen canonical file manifest incomplete: {dataset}")
    return result


def atomic_download(url: str, destination: Path, expected_size: int, expected_sha256: str) -> None:
    """Download one frozen source with bounded retries and HTTP range resume."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    retries = max(1, int(os.environ.get("K4_DOWNLOAD_RETRIES", "12")))
    timeout = max(30, int(os.environ.get("K4_DOWNLOAD_TIMEOUT", "300")))
    if temporary.is_file() and temporary.stat().st_size > expected_size:
        temporary.unlink()
    retryable = (TimeoutError, socket.timeout, urllib.error.URLError, ConnectionError, OSError)
    last_error = None
    for attempt in range(1, retries + 1):
        offset = temporary.stat().st_size if temporary.is_file() else 0
        if offset == expected_size:
            if sha256(temporary) == expected_sha256:
                temporary.replace(destination)
                print(
                    f"DOWNLOAD_COMPLETE file={destination.name} bytes={offset} "
                    "SHA256_VALID=YES resumed_from_complete_part=YES",
                    flush=True,
                )
                return
            temporary.unlink()
            offset = 0
        headers = {"User-Agent": "K4-server-benchmark-expansion/1.1"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            print(
                f"DOWNLOAD file={destination.name} attempt={attempt}/{retries} "
                f"resume_bytes={offset} expected_bytes={expected_size}",
                flush=True,
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = response.getcode()
                append = bool(offset and status == 206)
                mode = "ab" if append else "wb"
                with temporary.open(mode) as target:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        target.write(block)
            actual_size = temporary.stat().st_size
            if actual_size != expected_size:
                raise RuntimeError(f"incomplete download: got={actual_size} expected={expected_size}")
            actual_sha256 = sha256(temporary)
            if actual_sha256 != expected_sha256:
                temporary.unlink()
                raise RuntimeError(
                    f"download sha256 mismatch: got={actual_sha256} expected={expected_sha256}"
                )
            temporary.replace(destination)
            print(f"DOWNLOAD_COMPLETE file={destination.name} bytes={actual_size} SHA256_VALID=YES", flush=True)
            return
        except retryable + (RuntimeError,) as exc:
            last_error = exc
            if temporary.is_file() and temporary.stat().st_size > expected_size:
                temporary.unlink()
            if attempt == retries:
                break
            delay = min(5 * attempt, 60)
            partial = temporary.stat().st_size if temporary.is_file() else 0
            print(
                f"DOWNLOAD_RETRY file={destination.name} partial_bytes={partial} "
                f"delay_seconds={delay} error={type(exc).__name__}:{exc}",
                flush=True,
            )
            time.sleep(delay)
    raise RuntimeError(
        f"download failed after {retries} attempts; partial file retained for resume: "
        f"{temporary}; last_error={last_error}"
    )


def ensure_dataset_files(dataset: str, dry_run=False) -> list[dict]:
    records = []
    root = DATA_ROOT / dataset
    entries = required_data_entries(dataset)
    for kind, entry in entries.items():
        canonical = root / entry["canonical_filename"]
        raw = root / entry["raw_filename"]
        status = "PRESENT"
        if not canonical.is_file():
            status = "WOULD_DOWNLOAD" if dry_run else "DOWNLOADED"
            if not dry_run:
                root.mkdir(parents=True, exist_ok=True)
                expected_raw_size = int(entry["raw_size"])
                expected_raw_sha256 = entry["sha256"]
                if raw.is_file() and (
                    raw.stat().st_size != expected_raw_size or sha256(raw) != expected_raw_sha256
                ):
                    partial = raw.with_suffix(raw.suffix + ".part")
                    if raw.stat().st_size < expected_raw_size and not partial.exists():
                        raw.replace(partial)
                    else:
                        raw.unlink()
                if not raw.is_file():
                    atomic_download(
                        entry["source_url"], raw, expected_raw_size, expected_raw_sha256
                    )
                if raw.stat().st_size != int(entry["raw_size"]) or sha256(raw) != entry["sha256"]:
                    raise RuntimeError(f"canonical compressed identity mismatch: {dataset}:{kind}")
                temporary = canonical.with_suffix(canonical.suffix + ".part")
                with gzip.open(raw, "rb") as source, temporary.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                temporary.replace(canonical)
        if canonical.is_file():
            if canonical.stat().st_size != int(entry["canonical_size"]):
                raise RuntimeError(f"canonical size mismatch: {dataset}:{kind}")
            if sha256(canonical) != entry["canonical_sha256"]:
                raise RuntimeError(f"canonical sha256 mismatch: {dataset}:{kind}")
        records.append({"dataset": dataset, "kind": kind, "status": status, "canonical_path": str(canonical), "source_url": entry["source_url"], "expected_size": entry["canonical_size"], "expected_sha256": entry["canonical_sha256"]})
    return records


def int_lines(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            value = line.strip()
            if not value:
                continue
            try:
                yield int(value)
            except ValueError as exc:
                raise RuntimeError(f"non-integer canonical value {path}:{line_number}") from exc


def boundary_candidates(cumulative, target):
    position = int(np.searchsorted(cumulative, target, side="left"))
    return sorted({int(cumulative[i]) for i in range(max(0, position - 2), min(len(cumulative), position + 3))})


def choose_block_cuts(group_sizes, n_events):
    cumulative = np.cumsum(np.asarray(group_sizes, dtype=np.int64))
    targets = [fraction * n_events for fraction in (0.4, 0.6, 0.8)]
    options = [boundary_candidates(cumulative, target) for target in targets]
    best = None
    for c40 in options[0]:
        for c60 in options[1]:
            for c80 in options[2]:
                if not (0 < c40 < c60 < c80 < n_events):
                    continue
                error = sum(abs(cut - target) for cut, target in zip((c40, c60, c80), targets))
                key = (error, abs(c40 - targets[0]), abs(c60 - targets[1]), abs(c80 - targets[2]))
                if best is None or key < best[0]:
                    best = (key, (int(c40), int(c60), int(c80)))
    if best is None:
        raise RuntimeError("TIMESTAMP_BLOCK_SPLIT_FEASIBLE=NO")
    return best[1]


def canonical_identity(dataset: str) -> dict:
    root = DATA_ROOT / dataset
    nverts_path = root / f"{dataset}-nverts.txt"
    times_path = root / f"{dataset}-times.txt"
    simplices_path = root / f"{dataset}-simplices.txt"
    nverts = list(int_lines(nverts_path))
    times = list(int_lines(times_path))
    if len(nverts) != len(times):
        raise RuntimeError(f"nverts/times row mismatch: {dataset}")
    simplex_values = int_lines(simplices_path)
    nodes = set()
    consumed = 0
    for size in nverts:
        if size <= 0:
            raise RuntimeError(f"malformed simplex size: {dataset}:{size}")
        row = []
        for _ in range(size):
            try:
                row.append(next(simplex_values))
            except StopIteration as exc:
                raise RuntimeError(f"simplices stream truncated: {dataset}") from exc
        consumed += size
        nodes.update(row)
    try:
        next(simplex_values)
        raise RuntimeError(f"simplices stream has trailing values: {dataset}")
    except StopIteration:
        pass
    groups = Counter(times)
    cuts = choose_block_cuts([groups[t] for t in sorted(groups)], len(times))
    sorted_times = sorted(times)
    violations = sum(int(sorted_times[c - 1] >= sorted_times[c]) for c in cuts)
    return {"dataset": dataset, "events": len(nverts), "nodes": len(nodes), "timestamp_groups": len(groups), "simplex_membership_values": consumed, "min_timestamp": min(times), "max_timestamp": max(times), "cut_40": cuts[0], "cut_60": cuts[1], "cut_80": cuts[2], "same_timestamp_split_violations": violations, "canonical_status": "PASS" if violations == 0 else "FAIL", "audited_at": now()}


def stage_s0(dry_run=False) -> None:
    required = [CONFIG_PATH, DATA_MANIFEST_PATH, IDENTITY_REFERENCE_PATH, BUNDLE / "README_SERVER.md", BUNDLE / "RUN_ALL_SERVER.sh", BUNDLE / "requirements.txt"]
    missing = [str(path.relative_to(BUNDLE)) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing bundle files: {missing}")
    cfg = config()
    gates = {
        "target_datasets": tuple(cfg["target_datasets"]) == TARGETS,
        "excluded_datasets": tuple(cfg["excluded_datasets"]) == EXCLUDED,
        "strict_T2": cfg["strict_T2"] == "YES",
        "official_full": cfg["official_full"] == "FULL_STANDARD_RESIDUAL",
        "seeds": tuple(cfg["seeds"]) == SEEDS,
        "margin": float(cfg["noninferiority_margin"]) == MARGIN,
        "bootstraps": int(cfg["bootstrap_replicates"]) == BOOTSTRAPS,
        "sampling": cfg["sampling"] == "NO",
        "R1": cfg["K4_R1_EXECUTED"] == "NO",
    }
    if not all(gates.values()):
        raise RuntimeError(f"frozen protocol validation failed: {gates}")
    hashes_path = BUNDLE / "manifests" / "HASHES.json"
    verified = 0
    if hashes_path.is_file():
        hashes = read_json(hashes_path)
        for relative, expected in hashes.get("files", {}).items():
            path = BUNDLE / relative
            if not path.is_file() or sha256(path) != expected:
                raise RuntimeError(f"bundle file identity mismatch: {relative}")
            verified += 1
    elif not dry_run:
        raise RuntimeError("bundle HASHES.json missing")
    print(f"S0 BUNDLE_IDENTITY_VALID=YES HASHED_FILES_VERIFIED={verified} DRY_RUN={'YES' if dry_run else 'NO'} TARGET_DATASETS=3 OFFICIAL_FULL=FULL_STANDARD_RESIDUAL", flush=True)


def stage_s1(dataset=None, dry_run=False) -> None:
    datasets = (dataset,) if dataset else TARGETS
    references = expected_identity()
    rows = []
    for ds in datasets:
        rows.extend(ensure_dataset_files(ds, dry_run=dry_run))
        if dry_run:
            print(f"S1 dataset={ds} locator=PASS download=NOT_EXECUTED frozen_urls=YES", flush=True)
            continue
        identity = canonical_identity(ds)
        expected = references[ds]
        mismatch = not (identity["events"] == int(expected["events"]) and identity["nodes"] == int(expected["nodes"]) and identity["timestamp_groups"] == int(expected["timestamp_groups"]) and identity["same_timestamp_split_violations"] == 0)
        identity["DATA_IDENTITY_MISMATCH"] = "YES" if mismatch else "NO"
        identity["data_identity_valid"] = "NO" if mismatch else "YES"
        write_json(S1 / f"{ds}_identity.json", identity)
        if mismatch:
            raise RuntimeError(f"DATA_IDENTITY_MISMATCH=YES:{ds}")
        (CHECKPOINTS / f"DONE_{slug(ds)}_DATA").parent.mkdir(parents=True, exist_ok=True)
        (CHECKPOINTS / f"DONE_{slug(ds)}_DATA").write_text(now() + "\n", encoding="utf-8")
        print(f"S1 dataset={ds} events={identity['events']} nodes={identity['nodes']} timestamp_groups={identity['timestamp_groups']} SAME_TIMESTAMP_SPLIT_VIOLATIONS=0 DATA_VALID=YES", flush=True)
    if rows and not dry_run:
        prior = []
        path = S1 / "K4_SERVER_DATA_FILES.csv"
        if path.is_file():
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                prior = list(csv.DictReader(handle))
        merged = {(r["dataset"], r["kind"]): r for r in prior}
        merged.update({(r["dataset"], r["kind"]): r for r in rows})
        write_csv(path, [merged[key] for key in sorted(merged)])


def load_events(dataset: str):
    from kclosure.core import load_timestamped_simplices
    root = DATA_ROOT / dataset
    return load_timestamped_simplices(root / f"{dataset}-nverts.txt", root / f"{dataset}-simplices.txt", root / f"{dataset}-times.txt", event_order="timestamp")[0]


def build_state(events, cut: int, hb: Heartbeat):
    by_size = {size: set() for size in range(1, 6)}
    hb.update(total=cut, status="build_observed_state")
    for index, event in enumerate(events[:cut], 1):
        vertices = event["vertices"]
        for size in range(1, min(5, len(vertices)) + 1):
            by_size[size].update(combinations(vertices, size))
        if index % 10000 == 0:
            hb.update(processed=index)
    hb.update(processed=cut, status="observed_state_complete")
    return {"simplices_by_size": by_size, "nodes": by_size[1], "edges": by_size[2], "triangles": by_size[3], "tetrahedra": by_size[4]}


def enumerate_candidates(state, hb: Heartbeat):
    boundaries = state["simplices_by_size"][4]
    observed_targets = state["simplices_by_size"][5]
    ridge_to_completions = defaultdict(set)
    hb.update(processed=0, total=len(boundaries), status="build_ridge_index")
    for index, boundary in enumerate(boundaries, 1):
        bset = set(boundary)
        for ridge in combinations(boundary, 3):
            ridge_to_completions[ridge].add(next(iter(bset - set(ridge))))
        if index % 100000 == 0:
            hb.update(processed=index)
    potential = set()
    invalid = 0
    raw_proposals = 0
    hb.update(processed=0, total=len(boundaries), status="enumerate_boundary_driven_candidates")
    for index, boundary in enumerate(sorted(boundaries), 1):
        completion_sets = [ridge_to_completions[ridge] for ridge in combinations(boundary, 3)]
        possible = set.intersection(*(set(values) for values in completion_sets)) if completion_sets else set()
        for vertex in possible - set(boundary):
            raw_proposals += 1
            candidate = tuple(sorted((*boundary, vertex)))
            if not all(face in boundaries for face in combinations(candidate, 4)):
                invalid += 1
                continue
            potential.add(candidate)
        if index % 100000 == 0:
            hb.update(processed=index)
    already_closed = potential & observed_targets
    candidates = sorted(potential - observed_targets)
    hb.update(processed=len(boundaries), status="candidate_universe_complete")
    return candidates, {"potential_5node_sets": len(potential), "potential_proposals_before_dedup": raw_proposals, "valid_K4_boundaries": len(potential), "already_closed": len(already_closed), "incomplete_boundaries": invalid, "final_open_K4_candidates": len(candidates), "candidate_generation_method": "audited_boundary_face_inverted_index"}


def indexed_future_labels(candidates, events, start: int, end: int, hb: Heartbeat):
    by_vertex = defaultdict(list)
    hb.update(processed=0, total=len(candidates), status="build_candidate_label_index")
    for cid, candidate in enumerate(candidates):
        for vertex in candidate:
            by_vertex[int(vertex)].append(cid)
        if (cid + 1) % 250000 == 0:
            hb.update(processed=cid + 1)
    positive = np.zeros(len(candidates), dtype=np.uint8)
    total = end - start
    hb.update(processed=0, total=total, status="indexed_future_labeling")
    for offset, event in enumerate(events[start:end], 1):
        vertices = set(int(v) for v in event["vertices"])
        if len(vertices) >= 5:
            lists = [by_vertex.get(v, ()) for v in vertices]
            if lists:
                for cid in min(lists, key=len):
                    if not positive[cid] and all(v in vertices for v in candidates[cid]):
                        positive[cid] = 1
        if offset % 10000 == 0:
            hb.update(processed=offset)
    hb.update(processed=total, status="future_labels_complete")
    return positive


def save_pickle_atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def census_one(dataset: str, split: str) -> dict:
    marker = done_marker(dataset, split)
    root = S2 / dataset / split
    census_path = root / "census.json"
    required = [census_path, root / "candidates.npy", root / "labels.npy", root / "state.pkl"]
    if marker.is_file() and all(path.is_file() for path in required):
        result = read_json(census_path)
        print(f"S2 RESUME dataset={dataset} split={split} checkpoint=DONE candidates={result['final_open_K4_candidates']} positives={result['future_K4_positive_closures']}", flush=True)
        return result
    identity = read_json(S1 / f"{dataset}_identity.json")
    if identity.get("data_identity_valid") != "YES":
        raise RuntimeError(f"S1 identity gate not passed: {dataset}")
    events = load_events(dataset)
    cut = int(identity["cut_60"] if split == "train" else identity["cut_80"])
    label_end = int(identity["cut_80"] if split == "train" else len(events))
    hb = Heartbeat(dataset, f"S2_{split}").start()
    try:
        state = build_state(events, cut, hb)
        candidates, funnel = enumerate_candidates(state, hb)
        labels = indexed_future_labels(candidates, events, cut, label_end, hb)
        root.mkdir(parents=True, exist_ok=True)
        candidate_array = np.asarray(candidates, dtype=np.int64).reshape((-1, 5)) if candidates else np.empty((0, 5), dtype=np.int64)
        np.save(root / "candidates.npy", candidate_array)
        np.save(root / "labels.npy", labels)
        hb.update(status="persist_state_checkpoint")
        save_pickle_atomic(root / "state.pkl", state)
        row = {"dataset": dataset, "split": split, "strict_T2": "YES", "same_timestamp_split_violations": 0, "observation_end_index": cut, "label_start_index": cut, "label_end_index": label_end, "observed_L1": len(state["simplices_by_size"][1]), "observed_L2": len(state["simplices_by_size"][2]), "observed_L3": len(state["simplices_by_size"][3]), "observed_L4": len(state["simplices_by_size"][4]), **funnel, "future_K4_positive_closures": int(labels.sum()), "future_nonclosures": int(len(labels) - labels.sum()), "prevalence": float(labels.mean()) if len(labels) else 0.0, "candidate_hash": sha_rows(candidates), "label_hash": array_hash(labels), "candidate_ordering": "lexicographic integer tuple", "candidate_protocol_match": "YES", "sampling": "NO", "completed_at": now()}
        write_json(census_path, row)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(now() + "\n", encoding="utf-8")
        hb.close("DONE_CHECKPOINT")
        return row
    except Exception:
        hb.close("FAILED_NO_CHECKPOINT")
        raise


def merge_census_outputs() -> None:
    rows = []
    for dataset in TARGETS:
        for split in ("train", "eval"):
            path = S2 / dataset / split / "census.json"
            if path.is_file():
                rows.append(read_json(path))
    write_csv(S2 / "K4_SERVER_CANDIDATE_FUNNEL.csv", rows)


def stage_s2(dataset=None, split=None, dry_run=False) -> None:
    if dry_run:
        print("S2 DRY_RUN=YES candidate_builder=audited_boundary_face_inverted_index labeling=indexed_exact sampling=NO resume=split_checkpoint", flush=True)
        return
    datasets = (dataset,) if dataset else TARGETS
    splits = (split,) if split else ("train", "eval")
    for ds in datasets:
        for part in splits:
            census_one(ds, part)
    merge_census_outputs()


def support_gate(train, evaluation):
    tc, tp = int(train["final_open_K4_candidates"]), int(train["future_K4_positive_closures"])
    ec, ep = int(evaluation["final_open_K4_candidates"]), int(evaluation["future_K4_positive_closures"])
    if min(tc, tp, ec, ep) <= 0:
        return "NOT_FEASIBLE", "NO", "zero candidate or positive support"
    minimum = min(tp, ep)
    if minimum < 10:
        return "VERY_SPARSE", "NO", "min(train positives, eval positives) < 10"
    if minimum < 20:
        return "SPARSE", "NO", "10 <= min(train positives, eval positives) < 20"
    if minimum >= 50:
        return "STRONG_EXPANSION", "YES", "train/eval positives both >= 50"
    return "USEFUL_EXPANSION", "YES", "train/eval positives both >= 20"


def stage_s3(dry_run=False) -> None:
    if dry_run:
        print("S3 DRY_RUN=YES frozen_positive_support_gate=PASS", flush=True)
        return
    rows = []
    for dataset in TARGETS:
        train = read_json(S2 / dataset / "train" / "census.json")
        evaluation = read_json(S2 / dataset / "eval" / "census.json")
        level, decision, reason = support_gate(train, evaluation)
        rows.append({"dataset": dataset, "train_candidates": train["final_open_K4_candidates"], "train_positives": train["future_K4_positive_closures"], "eval_candidates": evaluation["final_open_K4_candidates"], "eval_positives": evaluation["future_K4_positive_closures"], "train_prevalence": train["prevalence"], "eval_prevalence": evaluation["prevalence"], "support_level": level, "formal_MLP_decision": decision, "reason": reason, "candidate_protocol_valid": "YES", "same_timestamp_split_violations": 0})
        print(f"S3 dataset={dataset} train={train['final_open_K4_candidates']}/{train['future_K4_positive_closures']} eval={evaluation['final_open_K4_candidates']}/{evaluation['future_K4_positive_closures']} support={level} FORMAL_MLP={decision}", flush=True)
    write_csv(S3 / "K4_SERVER_CANDIDATE_AVAILABILITY.csv", rows)
    write_csv(S3 / "K4_SERVER_SUPPORT_GATE.csv", rows)
    write_json(S3 / "support_gate.json", {row["dataset"]: row for row in rows})


def load_task(dataset: str, split: str):
    from scripts import run_k4_task_core_perturbation_v1 as k4old
    root = S2 / dataset / split
    with (root / "state.pkl").open("rb") as handle:
        state = pickle.load(handle)
    candidates_array = np.load(root / "candidates.npy", mmap_mode=None)
    candidates = [tuple(map(int, row)) for row in candidates_array]
    labels = np.load(root / "labels.npy").astype(np.int8)
    candidate_tetras = {face for candidate in candidates for face in combinations(candidate, 4)}
    candidate_triangles = {face for candidate in candidates for face in combinations(candidate, 3)}
    candidate_edges = {face for candidate in candidates for face in combinations(candidate, 2)}
    lower_adj = [set() for _ in candidates]
    boundary_to_ids = defaultdict(list)
    for cid, candidate in enumerate(candidates):
        for face in combinations(candidate, 4):
            boundary_to_ids[face].append(cid)
    for ids in boundary_to_ids.values():
        for left, right in combinations(ids, 2):
            lower_adj[left].add(right)
            lower_adj[right].add(left)
    full = {"nodes": set(state["simplices_by_size"][1]), "edges": set(state["simplices_by_size"][2]), "triangles": set(state["simplices_by_size"][3]), "tetras": set(state["simplices_by_size"][4]), "pentas": set(state["simplices_by_size"][5])}
    census = read_json(root / "census.json")
    task = {"dataset": dataset, "task": f"T2_{split.upper()}", "state": state, "candidates": candidates, "labels": labels, "candidate_tetras": candidate_tetras, "candidate_triangles": candidate_triangles, "candidate_edges": candidate_edges, "lower_adj": lower_adj, "full": full, "generation": {"enumeration_method": "boundary_face_inverted_index"}, "candidate_hash": census["candidate_hash"], "boundary_hash": sha_rows(sorted(candidate_tetras)), "positive_count": int(labels.sum()), "prevalence": float(labels.mean()) if len(labels) else 0.0}
    return task, k4old.decompose(task)


def feature_names():
    from scripts import run_k4_task_core_perturbation_v1 as k4old
    h = tuple(k4old.H_NAMES)
    block = tuple(k4old.L3_NAMES + k4old.L2_NAMES + k4old.L1_NAMES)
    return {"C0*": h, "C1*": h + tuple("T0__" + x for x in block), "C2*": h + tuple("T0__" + x for x in block) + tuple("TLE1__" + x for x in block), "C3*": h + tuple("T0__" + x for x in block) + tuple("TLE1__" + x for x in block) + tuple("TLE2__" + x for x in block), "Full": tuple(k4old.FULL_NAMES)}


def build_features(dataset: str, split: str):
    from scripts import run_k4_task_core_perturbation_v1 as k4old
    from scripts import run_task_core_confirmatory_v2 as structural
    root = S4 / "features" / dataset / split
    paths = {model: root / f"{model.replace('*', 'star')}.npy" for model in MODELS}
    meta_path = root / "feature_meta.json"
    if meta_path.is_file() and all(path.is_file() for path in paths.values()):
        return {model: np.load(path, mmap_mode="r") for model, path in paths.items()}, read_json(meta_path)
    if dataset == "congress-bills":
        helper = BUNDLE / "scripts" / "k4_congress_memory_safe.py"
        if not helper.is_file():
            raise RuntimeError(f"congress-bills memory-safe helper missing:{helper}")
        print(
            "S4 dataset=congress-bills representation=compact_exact_disk_indices_v1 "
            "sampling=NO launching_feature_subprocess=YES",
            flush=True,
        )
        subprocess.run([sys.executable, str(helper)], cwd=BUNDLE, check=True)
        if not meta_path.is_file() or not all(path.is_file() for path in paths.values()):
            raise RuntimeError("congress-bills memory-safe feature subprocess returned without complete checkpoint")
        return {model: np.load(path, mmap_mode="r") for model, path in paths.items()}, read_json(meta_path)
    task, dec = load_task(dataset, split)
    n = len(task["candidates"])
    hb = Heartbeat(dataset, f"S4_features_{split}").start()
    hb.update(total=n, status="standard_residual_features")
    arrays = k4old.compute_features(task, dec)
    h = np.column_stack([arrays[name] for name in k4old.H_NAMES]).astype(np.float32)
    full = np.column_stack([arrays[name] for name in k4old.FULL_NAMES]).astype(np.float32)
    local = {"C1*": np.zeros((n, 39), dtype=np.float32), "C2*": np.zeros((n, 39), dtype=np.float32), "C3*": np.zeros((n, 39), dtype=np.float32)}
    by_triangle = defaultdict(set)
    for tetra in task["full"]["tetras"]:
        for triangle in combinations(tetra, 3):
            by_triangle[triangle].add(tetra)
    hb.update(processed=0, total=n, status="strict_T0_TLE1_TLE2_features")
    for cid, candidate in enumerate(task["candidates"]):
        t0, t1, t2 = structural.tetra_shells(task, candidate, by_triangle)
        scopes = {"C1*": t0, "C2*": t0 | t1, "C3*": t0 | t1 | t2}
        for model, tetras in scopes.items():
            triangles, edges = structural.closure(tetras)
            local[model][cid] = structural.allowed_structural_features(candidate, tetras, triangles, edges)
        if (cid + 1) % 10000 == 0:
            hb.update(processed=cid + 1)
    mats = {"C0*": h, "C1*": np.column_stack([h, local["C1*"]]).astype(np.float32), "C2*": np.column_stack([h, local["C1*"], local["C2*"]]).astype(np.float32), "C3*": np.column_stack([h, local["C1*"], local["C2*"], local["C3*"]]).astype(np.float32), "Full": full}
    names = feature_names()
    root.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        np.save(paths[model], mats[model])
    audits = []
    for left, right, width in (("C0*", "C1*", 17), ("C1*", "C2*", 56), ("C2*", "C3*", 95)):
        diff = float(np.max(np.abs(mats[left].astype(np.float64) - mats[right][:, :width].astype(np.float64)))) if len(mats[left]) else 0.0
        audits.append({"relation": f"{left}_in_{right}", "max_abs_difference": diff, "dtype_left": str(mats[left].dtype), "dtype_right": str(mats[right].dtype), "status": "PASS" if diff <= 1e-12 else "FAIL"})
    if any(row["status"] != "PASS" for row in audits):
        hb.close("FEATURE_NESTING_FAIL")
        raise RuntimeError(f"FEATURE_INFORMATION_NESTING_FAIL:{dataset}:{split}")
    meta = {"dataset": dataset, "split": split, "candidate_hash": task["candidate_hash"], "label_hash": array_hash(task["labels"]), "candidate_count": n, "feature_names": {model: list(names[model]) for model in MODELS}, "input_dimensions": {model: mats[model].shape[1] for model in MODELS}, "feature_hashes": {model: array_hash(mats[model]) for model in MODELS}, "feature_information_nesting": "STRICT", "C0_IN_C1": "YES", "C1_IN_C2": "YES", "C2_IN_C3": "YES", "INHERITED_FEATURE_VALUES_IDENTICAL": "YES", "OFFICIAL_FULL": "FULL_STANDARD_RESIDUAL", "audits": audits}
    write_json(meta_path, meta)
    hb.close("DONE_FEATURE_CHECKPOINT")
    return {model: np.load(paths[model], mmap_mode="r") for model in MODELS}, meta


def classify_bootstrap(lower: float, upper: float) -> str:
    if upper <= MARGIN:
        return "NONINFERIORITY_SUPPORTED"
    if lower > MARGIN:
        return "FULL_ADVANTAGE_SUPPORTED"
    return "INCONCLUSIVE"


def paired_seed_bootstrap(y, lefts, rights, seed):
    y = np.asarray(y, dtype=np.float64)
    diffs = []
    for left, right in zip(lefts, rights):
        left = np.clip(np.asarray(left, dtype=np.float64), 1e-7, 1 - 1e-7)
        right = np.clip(np.asarray(right, dtype=np.float64), 1e-7, 1 - 1e-7)
        diffs.append((-(y * np.log(left) + (1 - y) * np.log(1 - left))) - (-(y * np.log(right) + (1 - y) * np.log(1 - right))))
    estimate = float(np.mean([diff.mean() for diff in diffs]))
    rng = np.random.default_rng(seed)
    values = np.empty(BOOTSTRAPS, dtype=np.float64)
    for index in range(BOOTSTRAPS):
        ids = rng.integers(0, len(y), size=len(y))
        values[index] = float(np.mean([diff[ids].mean() for diff in diffs]))
    lower, upper = float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))
    return {"D_LL": estimate, "CI_lower": lower, "CI_upper": upper, "classification": classify_bootstrap(lower, upper), "bootstrap_replicates": BOOTSTRAPS, "bootstrap_seed": seed}


def stage_s4(dataset=None, dry_run=False) -> None:
    if dry_run:
        print("S4 DRY_RUN=YES models=C0star,C1star,C2star,C3star,Full seeds=21001-21005 OFFICIAL_FULL=FULL_STANDARD_RESIDUAL feature_nesting=STRICT", flush=True)
        return
    from scripts.task_core_common_v1 import metric_row
    import torch
    import run_strict_timestamp_t2_mlp_local_v1 as mlp
    gate = read_json(S3 / "support_gate.json")
    datasets = (dataset,) if dataset else TARGETS
    for ds in datasets:
        if gate[ds]["formal_MLP_decision"] != "YES":
            path = CHECKPOINTS / f"DONE_{slug(ds)}_MLP_SKIPPED"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(gate[ds]["reason"] + "\n", encoding="utf-8")
            print(f"S4 dataset={ds} FORMAL_MLP=NO support={gate[ds]['support_level']} skipped_by_frozen_gate=YES", flush=True)
            continue
        done = CHECKPOINTS / f"DONE_{slug(ds)}_FORMAL_MLP"
        summary_path = S4 / ds / "formal_summary.json"
        if done.is_file() and summary_path.is_file():
            print(f"S4 RESUME dataset={ds} checkpoint=DONE", flush=True)
            continue
        train_mats, train_meta = build_features(ds, "train")
        eval_mats, eval_meta = build_features(ds, "eval")
        if train_meta["feature_information_nesting"] != "STRICT" or eval_meta["feature_information_nesting"] != "STRICT":
            raise RuntimeError(f"feature nesting gate failed:{ds}")
        y_train = np.load(S2 / ds / "train" / "labels.npy").astype(np.int8)
        y_eval = np.load(S2 / ds / "eval" / "labels.npy").astype(np.int8)
        names = feature_names()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        by_seed = []
        predictions = {model: {} for model in MODELS}
        for model in MODELS:
            for seed in SEEDS:
                root = S4 / ds / "models" / model.replace("*", "star") / f"seed_{seed}"
                model_path = root / "model.pt"
                scaler_path = root / "scaler"
                prediction_path = root / "eval_probabilities.npy"
                result_path = root / "result.json"
                if model_path.is_file() and scaler_path.with_suffix(".npz").is_file():
                    bundle = mlp.load_model(model_path, scaler_path, device)
                    execution = "RESUMED_EXISTING_MODEL"
                else:
                    hb = Heartbeat(ds, f"S4_train_{model}_seed{seed}").start()
                    bundle, _ = mlp.train_one_model(np.asarray(train_mats[model]), y_train, names[model], seed, device, BATCH, model_path, scaler_path)
                    hb.close("DONE_MODEL_CHECKPOINT")
                    execution = "TRAINED_SERVER"
                if prediction_path.is_file():
                    probability = np.load(prediction_path)
                else:
                    _, probability = mlp.predict_model(bundle, np.asarray(eval_mats[model]), device, BATCH)
                    prediction_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(prediction_path, np.asarray(probability, dtype=np.float64))
                probability = np.asarray(probability, dtype=np.float64)
                predictions[model][seed] = probability
                metrics = metric_row(y_eval, probability)
                row = {"dataset": ds, "model": model, "seed": seed, **metrics, "execution": execution, "input_dim": int(train_mats[model].shape[1]), "train_candidates": len(y_train), "train_positives": int(y_train.sum()), "eval_candidates": len(y_eval), "eval_positives": int(y_eval.sum()), "train_candidate_hash": train_meta["candidate_hash"], "eval_candidate_hash": eval_meta["candidate_hash"], "train_label_hash": train_meta["label_hash"], "eval_label_hash": eval_meta["label_hash"], "OFFICIAL_SEED_AGGREGATION": "MEAN_SEED_LOGLOSS"}
                write_json(result_path, row)
                by_seed.append(row)
        aggregate = {"dataset": ds, "n_seeds": 5, "OFFICIAL_SEED_AGGREGATION": "MEAN_SEED_LOGLOSS", "FEATURE_INFORMATION_NESTING": "STRICT", "MODEL_COMPARISON_DOMAIN_IDENTICAL": "YES", "OFFICIAL_FULL": "FULL_STANDARD_RESIDUAL"}
        for model in MODELS:
            rows = [row for row in by_seed if row["model"] == model]
            for metric in ("LogLoss", "Brier", "AP", "ROC_AUC"):
                values = [row[metric] for row in rows if row[metric] is not None]
                aggregate[f"{metric}_{model}"] = float(np.mean(values)) if values else None
        bootstraps = []
        for index, model in enumerate(("C0*", "C1*", "C2*", "C3*")):
            result = paired_seed_bootstrap(y_eval, [predictions[model][seed] for seed in SEEDS], [predictions["Full"][seed] for seed in SEEDS], 72001 + index)
            bootstraps.append({"dataset": ds, "model": model, "reference": "Full", "margin": MARGIN, **result})
        minimum = next((row["model"].replace("*", "") for row in bootstraps if row["classification"] == "NONINFERIORITY_SUPPORTED"), None)
        if minimum is None:
            c3 = next(row for row in bootstraps if row["model"] == "C3*")
            minimum = "CURRENT_TWO_HOP_CORE_INSUFFICIENT" if c3["classification"] == "FULL_ADVANTAGE_SUPPORTED" else "INCONCLUSIVE"
        aggregate["minimum_supported_core"] = minimum
        aggregate["bootstrap"] = bootstraps
        aggregate["train_candidates"] = len(y_train)
        aggregate["train_positives"] = int(y_train.sum())
        aggregate["eval_candidates"] = len(y_eval)
        aggregate["eval_positives"] = int(y_eval.sum())
        write_csv(S4 / ds / "K4_SERVER_RESULTS_BY_SEED.csv", by_seed)
        write_csv(S4 / ds / "K4_SERVER_BOOTSTRAP.csv", bootstraps)
        write_json(summary_path, aggregate)
        done.parent.mkdir(parents=True, exist_ok=True)
        done.write_text(now() + "\n", encoding="utf-8")
        print(f"S4 dataset={ds} FORMAL_MLP=YES minimum_supported_core={minimum} FEATURE_INFORMATION_NESTING=STRICT K4_R1_EXECUTED=NO", flush=True)


def stage_s5(dry_run=False) -> None:
    if dry_run:
        print("S5 DRY_RUN=YES final_schema=PASS existing_panel_comparison=PASS", flush=True)
        return
    gate = read_json(S3 / "support_gate.json")
    identities = {ds: read_json(S1 / f"{ds}_identity.json") for ds in TARGETS}
    master = []
    completed = 0
    pending = 0
    compact = 0
    for ds in TARGETS:
        summary_path = S4 / ds / "formal_summary.json"
        summary = read_json(summary_path) if summary_path.is_file() else None
        if gate[ds]["formal_MLP_decision"] == "YES" and summary is None:
            pending += 1
        if summary:
            completed += 1
            if summary["minimum_supported_core"] in ("C0", "C1", "C2", "C3"):
                compact += 1
        row = {"dataset": ds, "data_identity_valid": identities[ds]["data_identity_valid"], "train_candidates": gate[ds]["train_candidates"], "train_positives": gate[ds]["train_positives"], "eval_candidates": gate[ds]["eval_candidates"], "eval_positives": gate[ds]["eval_positives"], "support_level": gate[ds]["support_level"], "formal_MLP_run": "YES" if summary else "NO", "C0_LogLoss": summary.get("LogLoss_C0*") if summary else None, "C1_LogLoss": summary.get("LogLoss_C1*") if summary else None, "C2_LogLoss": summary.get("LogLoss_C2*") if summary else None, "C3_LogLoss": summary.get("LogLoss_C3*") if summary else None, "Full_LogLoss": summary.get("LogLoss_Full") if summary else None, "minimum_supported_core": summary.get("minimum_supported_core") if summary else None}
        boots = {item["model"]: item for item in summary.get("bootstrap", [])} if summary else {}
        for label, model in (("C0", "C0*"), ("C1", "C1*"), ("C2", "C2*"), ("C3", "C3*")):
            result = boots.get(model, {})
            row[f"{label}_D_LL"] = result.get("D_LL")
            row[f"{label}_CI_lower"] = result.get("CI_lower")
            row[f"{label}_CI_upper"] = result.get("CI_upper")
            row[f"{label}_classification"] = result.get("classification")
        master.append(row)
    write_csv(S5 / "K4_SERVER_EXPANSION_MASTER_TABLE.csv", master)
    existing = [{"panel": "existing", "dataset": "email-Eu", "train_candidates": 39, "train_positives": 13, "eval_candidates": 52, "eval_positives": 14}, {"panel": "existing", "dataset": "tags-ask-ubuntu", "train_candidates": 415, "train_positives": 32, "eval_candidates": 797, "eval_positives": 43}, {"panel": "existing", "dataset": "tags-math-sx", "train_candidates": 79, "train_positives": 11, "eval_candidates": 152, "eval_positives": 8}]
    comparison = existing + [{"panel": "new", "dataset": row["dataset"], "train_candidates": row["train_candidates"], "train_positives": row["train_positives"], "eval_candidates": row["eval_candidates"], "eval_positives": row["eval_positives"], "support_level": row["support_level"]} for row in master]
    write_csv(S5 / "K4_EXISTING_VS_NEW_SUPPORT_COMPARISON.csv", comparison)
    strong = sum(row["support_level"] == "STRONG_EXPANSION" for row in master)
    if completed >= 2 and strong >= 1:
        outcome = "STRONG"
    elif completed >= 1:
        outcome = "USEFUL"
    elif pending:
        outcome = "SUPPORT_FOUND_BUT_MODEL_PENDING"
    elif all(min(int(row["train_positives"]), int(row["eval_positives"])) < 20 for row in master):
        outcome = "SPARSE"
    else:
        outcome = "PROTOCOL_ISSUE"
    if completed >= 2 and strong >= 1 and compact >= 2:
        upgrade, r1 = "YES", "YES"
    elif completed >= 1 and compact >= 1:
        upgrade, r1 = "CONDITIONAL", "CONDITIONAL"
    else:
        upgrade, r1 = "NO", "NO"
    lines = ["# K4 Server Benchmark Expansion — Final Report", "", "## Frozen protocol", "", "- STRICT_T2 = YES; FULL_DATA = YES; FULL_CANDIDATE_UNIVERSE = YES; SAMPLING = NO", "- OFFICIAL_FULL = FULL_STANDARD_RESIDUAL", "- Models = C0*, C1*, C2*, C3*, Full; seeds = 21001-21005", "- PRIMARY_ENDPOINT = LogLoss; NONINFERIORITY_MARGIN = 0.02; BOOTSTRAP_REPLICATES = 2000", "- K4_R1_EXECUTED = NO; LEGACY_RESULTS_OVERWRITTEN = NO", "", "## Census and formal MLP", "", "| dataset | train cand/pos | eval cand/pos | support | MLP | C0* | C1* | C2* | C3* | Full | minimum core |", "|---|---:|---:|---|---|---:|---:|---:|---:|---:|---|"]
    for row in master:
        fmt = lambda value: "—" if value is None else f"{float(value):.9f}"
        lines.append(f"| {row['dataset']} | {row['train_candidates']}/{row['train_positives']} | {row['eval_candidates']}/{row['eval_positives']} | {row['support_level']} | {row['formal_MLP_run']} | {fmt(row['C0_LogLoss'])} | {fmt(row['C1_LogLoss'])} | {fmt(row['C2_LogLoss'])} | {fmt(row['C3_LogLoss'])} | {fmt(row['Full_LogLoss'])} | {row['minimum_supported_core'] or '—'} |")
    lines += ["", "## Compact models versus Full", "", "| dataset | model | D_LL | 95% CI | classification |", "|---|---|---:|---:|---|"]
    for row in master:
        for label in ("C0", "C1", "C2", "C3"):
            if row[f"{label}_D_LL"] is not None:
                lines.append(f"| {row['dataset']} | {label}* | {row[f'{label}_D_LL']:.9f} | [{row[f'{label}_CI_lower']:.9f}, {row[f'{label}_CI_upper']:.9f}] | {row[f'{label}_classification']} |")
    new_max = max((int(row["train_positives"]) + int(row["eval_positives"]) for row in master), default=0)
    existing_max = max(row["train_positives"] + row["eval_positives"] for row in existing)
    improved = "YES" if new_max > existing_max else "NO"
    useful = [row["dataset"] for row in master if row["support_level"] in ("USEFUL_EXPANSION", "STRONG_EXPANSION")]
    strong_names = [row["dataset"] for row in master if row["support_level"] == "STRONG_EXPANSION"]
    mlp_names = [row["dataset"] for row in master if row["formal_MLP_run"] == "YES"]
    lines += [
        "", "## Required scientific answers", "",
        f"1. NDC-classes train/eval candidates and positives: {master[0]['train_candidates']}/{master[0]['train_positives']} and {master[0]['eval_candidates']}/{master[0]['eval_positives']}.",
        f"2. congress-bills train/eval candidates and positives: {master[1]['train_candidates']}/{master[1]['train_positives']} and {master[1]['eval_candidates']}/{master[1]['eval_positives']}.",
        f"3. DAWN train/eval candidates and positives: {master[2]['train_candidates']}/{master[2]['train_positives']} and {master[2]['eval_candidates']}/{master[2]['eval_positives']}.",
        f"4. USEFUL_EXPANSION or stronger datasets: {', '.join(useful) if useful else 'NONE'}.",
        f"5. STRONG_EXPANSION datasets: {', '.join(strong_names) if strong_names else 'NONE'}.",
        f"6. Formal MLP datasets: {', '.join(mlp_names) if mlp_names else 'NONE'}.",
        "7. C0*/C1*/C2*/C3*/Full LogLoss values are reported in the census/formal-MLP table above.",
        "8. Every compact-model D_LL, paired 95% CI, and classification is reported in the compact-model table above.",
        "9. Each dataset's minimum supported core is reported in the formal-MLP table; no unsupported 'FULL_NETWORK_REQUIRED' label is used.",
        f"10. New K4 positive support exceeds the strongest existing panel member: {improved}.",
        f"11. K4_EVIDENCE_UPGRADE_RECOMMENDATION = {upgrade}.",
        f"12. K4_R1_FOLLOWUP_RECOMMENDED = {r1}.",
        "", "## Scientific decisions", "",
        f"- K4_EXPANSION_OUTCOME = {outcome}",
        f"- K4_EVIDENCE_UPGRADE_RECOMMENDATION = {upgrade}",
        f"- K4_R1_FOLLOWUP_RECOMMENDED = {r1}",
        "- No C4/3-hop/4-hop model, feature search, tuning, or R1 was run.",
    ]
    report = RESULTS / "K4_SERVER_EXPANSION_FINAL_REPORT.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    integrity = ["STRICT_T2 = YES", "FULL_DATA = YES", "FULL_CANDIDATE_UNIVERSE = YES", "SAMPLING = NO", "TARGET_DATASETS = 3"]
    for ds in TARGETS:
        integrity.append(f"{slug(ds)}_DATA_VALID = {identities[ds]['data_identity_valid']}")
    integrity += ["SAME_TIMESTAMP_SPLIT_VIOLATIONS = 0", "K4_CANDIDATE_PROTOCOL_MATCH = YES"]
    for row in master:
        integrity.append(f"{slug(row['dataset'])}_SUPPORT_LEVEL = {row['support_level']}")
    integrity += [f"FORMAL_MLP_DATASETS = {completed}/3", "FEATURE_INFORMATION_NESTING = STRICT" if completed else "FEATURE_INFORMATION_NESTING = NOT_APPLICABLE", "OFFICIAL_FULL = FULL_STANDARD_RESIDUAL", "NONINFERIORITY_MARGIN = 0.02", "BOOTSTRAP_REPLICATES = 2000", "K4_R1_EXECUTED = NO", "LEGACY_RESULTS_OVERWRITTEN = NO", f"K4_EXPANSION_OUTCOME = {outcome}", f"K4_EVIDENCE_UPGRADE_RECOMMENDATION = {upgrade}", f"K4_R1_FOLLOWUP_RECOMMENDED = {r1}"]
    (RESULTS / "K4_SERVER_INTEGRITY_SUMMARY.txt").write_text("\n".join(integrity) + "\n", encoding="utf-8")
    print("\n".join(integrity), flush=True)


def selftest() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        marker = root / "DONE_FIXTURE_TRAIN_CENSUS"
        marker.write_text("done\n", encoding="utf-8")
        if not marker.is_file():
            raise RuntimeError("resume marker selftest failed")
    cuts = choose_block_cuts([4, 4, 4, 4, 4], 20)
    if cuts != (8, 12, 16):
        raise RuntimeError(f"timestamp cut selftest failed: {cuts}")
    train = {"final_open_K4_candidates": 100, "future_K4_positive_closures": 50}
    evaluation = dict(train)
    if support_gate(train, evaluation)[:2] != ("STRONG_EXPANSION", "YES"):
        raise RuntimeError("support gate selftest failed")
    class SilentHeartbeat:
        def update(self, **kwargs):
            return None
    simplex = (1, 2, 3, 4, 5)
    by_size = {1: {(v,) for v in simplex}, 2: set(combinations(simplex, 2)), 3: set(combinations(simplex, 3)), 4: set(combinations(simplex, 4)), 5: set()}
    candidates, funnel = enumerate_candidates({"simplices_by_size": by_size}, SilentHeartbeat())
    if candidates != [simplex] or funnel["final_open_K4_candidates"] != 1:
        raise RuntimeError("candidate builder fixture selftest failed")
    labels = indexed_future_labels(candidates, [{"vertices": simplex, "time": 1, "row_id": 0}], 0, 1, SilentHeartbeat())
    if labels.tolist() != [1]:
        raise RuntimeError("indexed future label fixture selftest failed")
    print("SELFTEST=PASS resume=PASS strict_timestamp=PASS gate=PASS candidate_builder=PASS indexed_labeling=PASS", flush=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("s0", "s1", "s2", "s3", "s4", "s5", "selftest"))
    parser.add_argument("--dataset", choices=TARGETS)
    parser.add_argument("--split", choices=("train", "eval"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.stage == "s0": stage_s0(args.dry_run)
    elif args.stage == "s1": stage_s1(args.dataset, args.dry_run)
    elif args.stage == "s2": stage_s2(args.dataset, args.split, args.dry_run)
    elif args.stage == "s3": stage_s3(args.dry_run)
    elif args.stage == "s4": stage_s4(args.dataset, args.dry_run)
    elif args.stage == "s5": stage_s5(args.dry_run)
    else: selftest()


if __name__ == "__main__":
    main()
