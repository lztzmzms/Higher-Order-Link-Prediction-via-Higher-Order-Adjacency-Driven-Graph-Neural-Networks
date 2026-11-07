from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import pickle
import shutil
import sys
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

DATASET = "congress-bills"
MODELS = ("C0*", "C1*", "C2*", "C3*", "Full")
MODEL_WIDTHS = {"C0*": 17, "C1*": 56, "C2*": 95, "C3*": 134, "Full": 56}
ROW_CHUNK = max(1000, int(os.environ.get("K4_CONGRESS_ROW_CHUNK", "10000")))
INDEX_CHUNK = max(10000, int(os.environ.get("K4_CONGRESS_INDEX_CHUNK", "250000")))
MIN_FREE_GIB = max(1, int(os.environ.get("K4_CONGRESS_MIN_FREE_GIB", "60")))
U32_MAX = np.iinfo(np.uint32).max
RECORD_DTYPE = np.dtype([("key", "<u8"), ("item", "<u4")])
EDGE_DTYPE = np.dtype([("src", "<u4"), ("dst", "<u4")])

def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")

def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)

def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def _heartbeat(stage: str, processed: int, total: int, started: float, detail: str = "") -> None:
    elapsed = int(time.time() - started)
    suffix = f" detail={detail}" if detail else ""
    print(
        f"HEARTBEAT dataset={DATASET} stage={stage} elapsed={elapsed}s "
        f"processed={processed}/{total} rss_bytes={_rss_bytes()} checkpoint=NO{suffix}",
        flush=True,
    )

def _rss_bytes() -> int:
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        return 0

def _open_npy(path: Path, mode: str, dtype=None, shape=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if mode.startswith("w"):
        return np.lib.format.open_memmap(path, mode=mode, dtype=dtype, shape=shape)
    return np.load(path, mmap_mode=mode)

def _unlink(path: Path) -> int:
    pass
    try:
        size = path.stat().st_size
    except OSError:
        return 0
    try:
        path.unlink()
    except OSError:
        return 0
    return int(size)

def _subset_index_complete(root: Path, stem: str) -> bool:
    return all(
        path.is_file()
        for path in (
            root / f"{stem}.complete",
            root / f"{stem}_keys.npy",
            root / f"{stem}_counts.npy",
        )
    )

def _reclaim_completed_intermediates(bundle: Path) -> int:
    pass
    results = bundle / "server_results" / "k4_server_benchmark_expansion_v1"
    reclaimed = 0
    for split in ("train", "eval"):
        index_root = results / "stageS4_mlp" / "compact_index" / DATASET / split
        feature_root = results / "stageS4_mlp" / "features" / DATASET / split
        feature_complete = (feature_root / "feature_meta.json").is_file() and all(
            (feature_root / f"{model.replace('*', 'star')}.npy").is_file() for model in MODELS
        )
        if feature_complete and index_root.is_dir():
            before = sum(path.stat().st_size for path in index_root.rglob("*") if path.is_file())
            try:
                shutil.rmtree(index_root)
                reclaimed += int(before)
            except OSError:
                pass
            continue
        if not index_root.is_dir():
            continue
        for stem in (
            "penta_tetra",
            "residual_edge",
            "residual_triangle",
            "all_tetra_triangle",
            "pure_l2_edge",
            "all_triangle_edge",
        ):
            if _subset_index_complete(index_root, stem):
                reclaimed += _unlink(index_root / f"{stem}.records")
        candidate_csr_complete = all(
            path.is_file()
            for path in (
                index_root / "candidate_csr.complete",
                index_root / "candidate_indptr.npy",
                index_root / "candidate_indices.npy",
                index_root / "candidate_face_groups.complete",
                index_root / "candidate_face_keys.npy",
                index_root / "candidate_face_counts.npy",
            )
        )
        if candidate_csr_complete:
            reclaimed += _unlink(index_root / "candidate_faces.records")
            reclaimed += _unlink(index_root / "candidate_edges.records")
        if _subset_index_complete(index_root, "penta_tetra"):
            reclaimed += _unlink(index_root / "full_pentas.npy")
        if not (index_root / "residual_tetras.complete").is_file():
            reclaimed += _unlink(index_root / "residual_tetras.npy")
            reclaimed += _unlink(index_root / "residual_mask.raw")
    if reclaimed:
        print(f"DISK_RECLAIM dataset={DATASET} bytes={reclaimed} status=DONE", flush=True)
    return reclaimed

def _disk_preflight(bundle: Path) -> None:
    results = bundle / "server_results" / "k4_server_benchmark_expansion_v1"
    results.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(results)
    required = MIN_FREE_GIB * (1024**3)
    print(
        f"DISK_PREFLIGHT dataset={DATASET} free_bytes={usage.free} "
        f"required_bytes={required} required_gib={MIN_FREE_GIB}",
        flush=True,
    )
    if usage.free < required:
        raise RuntimeError(
            f"insufficient free disk for exact congress-bills S4: "
            f"free={usage.free / 1024**3:.1f} GiB required>={MIN_FREE_GIB} GiB; "
            "free space or quota, then rerun the same S4 command"
        )

def _encode_rows(rows, radix: int) -> np.ndarray:
    values = np.asarray(rows, dtype=np.uint64)
    if values.ndim != 2:
        raise ValueError("simplex rows must be a matrix")
    result = np.zeros(len(values), dtype=np.uint64)
    for column in range(values.shape[1]):
        result = result * np.uint64(radix) + values[:, column] + np.uint64(1)
    return result

def _decode_codes(codes, width: int, radix: int) -> np.ndarray:
    values = np.asarray(codes, dtype=np.uint64).copy()
    rows = np.empty((len(values), width), dtype=np.uint32)
    for column in range(width - 1, -1, -1):
        rows[:, column] = (values % np.uint64(radix)).astype(np.uint32) - np.uint32(1)
        values //= np.uint64(radix)
    return rows

def _face_codes(rows, face_width: int, radix: int) -> np.ndarray:
    values = np.asarray(rows)
    result = np.empty((len(values), math.comb(values.shape[1], face_width)), dtype=np.uint64)
    for index, columns in enumerate(combinations(range(values.shape[1]), face_width)):
        result[:, index] = _encode_rows(values[:, columns], radix)
    return result

def _rows_from_set(path: Path, values, width: int, marker: Path) -> np.memmap:
    if marker.is_file() and path.is_file():
        return _open_npy(path, "r")
    count = len(values)
    if count > U32_MAX:
        raise RuntimeError(f"compact row id overflow: {path.name}:{count}")
    matrix = _open_npy(path, "w+", np.uint32, (count, width))
    buffer = []
    offset = 0
    started = time.time()
    for row in values:
        buffer.append(row)
        if len(buffer) >= INDEX_CHUNK:
            block = np.asarray(buffer, dtype=np.uint32)
            matrix[offset : offset + len(block)] = block
            offset += len(block)
            buffer.clear()
            _heartbeat("S4_compact_state", offset, count, started, path.stem)
    if buffer:
        block = np.asarray(buffer, dtype=np.uint32)
        matrix[offset : offset + len(block)] = block
        offset += len(block)
    if offset != count:
        raise RuntimeError(f"compact state row mismatch:{path}:{offset}!={count}")
    matrix.flush()
    marker.write_text(_now() + "\n", encoding="utf-8")
    return matrix

def _sort_records(path: Path, marker: Path) -> np.memmap:
    records = np.memmap(path, mode="r+", dtype=RECORD_DTYPE)
    if not marker.is_file():
        records.sort(order=("key", "item"), kind="quicksort")
        records.flush()
        marker.write_text(_now() + "\n", encoding="utf-8")
    return records

def _unique_keys_from_sorted(records, keys_path: Path, counts_path: Path, marker: Path):
    if marker.is_file() and keys_path.is_file() and counts_path.is_file():
        return _open_npy(keys_path, "r"), _open_npy(counts_path, "r")
    keys = records["key"]
    if len(keys) == 0:
        unique = _open_npy(keys_path, "w+", np.uint64, (0,))
        counts = _open_npy(counts_path, "w+", np.uint32, (0,))
    else:
        change = np.empty(len(keys), dtype=np.bool_)
        change[0] = True
        for start in range(1, len(keys), INDEX_CHUNK):
            end = min(len(keys), start + INDEX_CHUNK)
            change[start:end] = keys[start:end] != keys[start - 1 : end - 1]
        starts = np.flatnonzero(change).astype(np.int64)
        ends = np.empty_like(starts)
        ends[:-1] = starts[1:]
        ends[-1] = len(keys)
        unique = _open_npy(keys_path, "w+", np.uint64, (len(starts),))
        counts = _open_npy(counts_path, "w+", np.uint32, (len(starts),))
        unique[:] = keys[starts]
        sizes = ends - starts
        if sizes.max(initial=0) > U32_MAX:
            raise RuntimeError("compact incidence count overflow")
        counts[:] = sizes.astype(np.uint32)
        unique.flush()
        counts.flush()
        del change, starts, ends, sizes
    marker.write_text(_now() + "\n", encoding="utf-8")
    return unique, counts

def _lookup_counts(keys, counts, query) -> np.ndarray:
    query_array = np.asarray(query, dtype=np.uint64)
    flat = query_array.reshape(-1)
    positions = np.searchsorted(keys, flat)
    result = np.zeros(len(flat), dtype=np.uint32)
    valid = positions < len(keys)
    if np.any(valid):
        valid_ids = np.flatnonzero(valid)
        matches = keys[positions[valid_ids]] == flat[valid_ids]
        matched_ids = valid_ids[matches]
        result[matched_ids] = counts[positions[matched_ids]]
    return result.reshape(query_array.shape)

def _lookup_membership(keys, query) -> np.ndarray:
    query_array = np.asarray(query, dtype=np.uint64)
    flat = query_array.reshape(-1)
    positions = np.searchsorted(keys, flat)
    result = np.zeros(len(flat), dtype=np.bool_)
    valid = positions < len(keys)
    if np.any(valid):
        valid_ids = np.flatnonzero(valid)
        result[valid_ids] = keys[positions[valid_ids]] == flat[valid_ids]
    return result.reshape(query_array.shape)

def _build_candidate_face_records(candidates, radix: int, root: Path):
    raw = root / "candidate_faces.records"
    sorted_marker = root / "candidate_faces.sorted"
    complete = root / "candidate_faces.complete"
    groups_complete = root / "candidate_face_groups.complete"
    if all(
        path.is_file()
        for path in (
            root / "candidate_csr.complete",
            groups_complete,
            root / "candidate_face_keys.npy",
            root / "candidate_face_counts.npy",
        )
    ):
        _unlink(raw)
        return None, _open_npy(root / "candidate_face_keys.npy", "r"), _open_npy(root / "candidate_face_counts.npy", "r")
    total = len(candidates) * 5
    if not complete.is_file():
        records = np.memmap(raw, mode="w+", dtype=RECORD_DTYPE, shape=(total,))
        started = time.time()
        for start in range(0, len(candidates), INDEX_CHUNK):
            end = min(len(candidates), start + INDEX_CHUNK)
            faces = _face_codes(candidates[start:end], 4, radix)
            block = slice(start * 5, end * 5)
            records[block]["key"] = faces.reshape(-1)
            records[block]["item"] = np.repeat(np.arange(start, end, dtype=np.uint32), 5)
            _heartbeat("S4_candidate_face_index", end, len(candidates), started)
        records.flush()
        complete.write_text(_now() + "\n", encoding="utf-8")
    records = _sort_records(raw, sorted_marker)
    keys, counts = _unique_keys_from_sorted(
        records,
        root / "candidate_face_keys.npy",
        root / "candidate_face_counts.npy",
        groups_complete,
    )
    return records, keys, counts

def _build_subset_count_index(rows, subset_width: int, radix: int, stem: str, root: Path):
    raw = root / f"{stem}.records"
    generated = root / f"{stem}.generated"
    complete = root / f"{stem}.complete"
    keys_path = root / f"{stem}_keys.npy"
    counts_path = root / f"{stem}_counts.npy"
    if complete.is_file() and keys_path.is_file() and counts_path.is_file():
        _unlink(raw)
        return _open_npy(keys_path, "r"), _open_npy(counts_path, "r")
    if rows is None:
        raise RuntimeError(f"source rows missing for incomplete compact index:{stem}")
    if generated.is_file() and not raw.is_file():
        _unlink(generated)
        _unlink(root / f"{stem}.sorted")
    count = math.comb(rows.shape[1], subset_width)
    if not generated.is_file():
        records = np.memmap(raw, mode="w+", dtype=RECORD_DTYPE, shape=(len(rows) * count,))
        started = time.time()
        for start in range(0, len(rows), INDEX_CHUNK):
            end = min(len(rows), start + INDEX_CHUNK)
            codes = _face_codes(rows[start:end], subset_width, radix)
            block = slice(start * count, end * count)
            records[block]["key"] = codes.reshape(-1)
            records[block]["item"] = 0
            _heartbeat("S4_compact_subset_index", end, len(rows), started, stem)
        records.flush()
        generated.write_text(_now() + "\n", encoding="utf-8")
    records = _sort_records(raw, root / f"{stem}.sorted")
    keys, counts = _unique_keys_from_sorted(
        records,
        keys_path,
        counts_path,
        complete,
    )
    del records
    gc.collect()
    _unlink(raw)
    return keys, counts

def _build_candidate_csr(records, candidate_count: int, root: Path):
    marker = root / "candidate_csr.complete"
    indptr_path = root / "candidate_indptr.npy"
    indices_path = root / "candidate_indices.npy"
    if marker.is_file() and indptr_path.is_file() and indices_path.is_file():
        return _open_npy(indptr_path, "r"), _open_npy(indices_path, "r")
    keys = records["key"]
    if not len(keys):
        indptr = _open_npy(indptr_path, "w+", np.uint64, (candidate_count + 1,))
        indices = _open_npy(indices_path, "w+", np.uint32, (0,))
        marker.write_text(_now() + "\n", encoding="utf-8")
        return indptr, indices
    change = np.empty(len(keys), dtype=np.bool_)
    change[0] = True
    for start in range(1, len(keys), INDEX_CHUNK):
        end = min(len(keys), start + INDEX_CHUNK)
        change[start:end] = keys[start:end] != keys[start - 1 : end - 1]
    starts = np.flatnonzero(change).astype(np.int64)
    ends = np.empty_like(starts)
    ends[:-1] = starts[1:]
    ends[-1] = len(keys)
    sizes = ends - starts
    shared = sizes > 1
    starts, ends, sizes = starts[shared], ends[shared], sizes[shared]
    directed_count = int(np.sum(sizes * (sizes - 1), dtype=np.int64))
    if directed_count > U32_MAX * 16:
        raise RuntimeError(f"candidate adjacency unexpectedly large:{directed_count}")
    edge_raw = root / "candidate_edges.records"
    edges = np.memmap(edge_raw, mode="w+", dtype=EDGE_DTYPE, shape=(directed_count,))
    offset = 0
    started = time.time()
    for group_index, (left, right, size) in enumerate(zip(starts, ends, sizes), 1):
        ids = np.asarray(records["item"][left:right], dtype=np.uint32)
        src = np.repeat(ids, int(size) - 1)
        tiled = np.tile(ids, int(size))
        mask = tiled != np.repeat(ids, int(size))
        dst = tiled[mask]
        length = len(src)
        edges[offset : offset + length]["src"] = src
        edges[offset : offset + length]["dst"] = dst
        offset += length
        if group_index % 100000 == 0:
            _heartbeat("S4_candidate_csr", group_index, len(starts), started, f"edges={offset}")
    if offset != directed_count:
        raise RuntimeError(f"candidate edge count mismatch:{offset}!={directed_count}")
    edges.flush()
    edges.sort(order=("src", "dst"), kind="quicksort")
    edges.flush()
    src = np.asarray(edges["src"])
    degrees = np.bincount(src.astype(np.int64), minlength=candidate_count).astype(np.uint64)
    indptr = _open_npy(indptr_path, "w+", np.uint64, (candidate_count + 1,))
    indptr[0] = 0
    np.cumsum(degrees, out=indptr[1:])
    indices = _open_npy(indices_path, "w+", np.uint32, (directed_count,))
    indices[:] = edges["dst"]
    indptr.flush()
    indices.flush()
    marker.write_text(_now() + "\n", encoding="utf-8")
    del change, starts, ends, sizes, shared, edges, src, degrees
    gc.collect()
    _unlink(edge_raw)
    return indptr, indices

def _setdiff_sorted(left, right, output: Path) -> np.memmap:
    if output.is_file():
        return _open_npy(output, "r")
    values = np.setdiff1d(np.asarray(left), np.asarray(right), assume_unique=True)
    target = _open_npy(output, "w+", values.dtype, values.shape)
    target[:] = values
    target.flush()
    del values
    return target

def _build_compact_context(bundle: Path, split: str):
    results = bundle / "server_results" / "k4_server_benchmark_expansion_v1"
    s2root = results / "stageS2_census" / DATASET / split
    index_root = results / "stageS4_mlp" / "compact_index" / DATASET / split
    index_root.mkdir(parents=True, exist_ok=True)
    meta_path = index_root / "context_meta.json"
    candidates = np.load(s2root / "candidates.npy", mmap_mode="r")
    labels = np.load(s2root / "labels.npy", mmap_mode="r")
    if len(candidates) > U32_MAX:
        raise RuntimeError("candidate ids exceed uint32 compact index")
    state_marker = index_root / "state_compaction.complete"
    if not state_marker.is_file():
        started = time.time()
        with (s2root / "state.pkl").open("rb") as handle:
            state = pickle.load(handle)
        node_values = state["simplices_by_size"][1]
        max_node = max(max(row) for row in node_values)
        radix = int(max_node) + 2
        if radix**5 >= 2**64:
            raise RuntimeError(f"node radix cannot be encoded in uint64:{radix}")
        _write_json(index_root / "radix.json", {"radix": radix, "max_node": int(max_node)})
        full_rows = {}
        for width, name in ((2, "full_edges"), (3, "full_triangles"), (4, "full_tetras"), (5, "full_pentas")):
            full_rows[name] = _rows_from_set(
                index_root / f"{name}.npy",
                state["simplices_by_size"][width],
                width,
                index_root / f"{name}.complete",
            )
            state["simplices_by_size"][width] = None
            gc.collect()
        state = None
        del full_rows
        gc.collect()
        _heartbeat("S4_compact_state", 1, 1, started, "python_state_released")
        state_marker.write_text(_now() + "\n", encoding="utf-8")
    radix = int(_read_json(index_root / "radix.json")["radix"])
    full_edges = _open_npy(index_root / "full_edges.npy", "r")
    full_triangles = _open_npy(index_root / "full_triangles.npy", "r")
    full_tetras = _open_npy(index_root / "full_tetras.npy", "r")
    full_pentas_path = index_root / "full_pentas.npy"
    penta_index_ready = _subset_index_complete(index_root, "penta_tetra")
    full_pentas = _open_npy(full_pentas_path, "r") if full_pentas_path.is_file() else None
    if full_pentas is None and not penta_index_ready:
        raise RuntimeError("full_pentas compact state missing before penta_tetra index completion")

    candidate_records, candidate_face_keys, _ = _build_candidate_face_records(candidates, radix, index_root)
    candidate_indptr, candidate_indices = _build_candidate_csr(candidate_records, len(candidates), index_root)
    del candidate_records
    gc.collect()
    _unlink(index_root / "candidate_faces.records")
    penta_tetra_keys, penta_tetra_counts = _build_subset_count_index(full_pentas, 4, radix, "penta_tetra", index_root)
    if full_pentas is not None:
        del full_pentas
        gc.collect()
        _unlink(full_pentas_path)

    residual_path = index_root / "residual_tetras.npy"
    residual_marker = index_root / "residual_tetras.complete"
    if not (residual_marker.is_file() and residual_path.is_file()):
        _unlink(residual_path)
        residual_mask_path = index_root / "residual_mask.raw"
        mask = np.memmap(residual_mask_path, mode="w+", dtype=np.bool_, shape=(len(full_tetras),))
        started = time.time()
        residual_count = 0
        for start in range(0, len(full_tetras), INDEX_CHUNK):
            end = min(len(full_tetras), start + INDEX_CHUNK)
            codes = _encode_rows(full_tetras[start:end], radix)
            is_h = _lookup_membership(candidate_face_keys, codes) | _lookup_membership(penta_tetra_keys, codes)
            mask[start:end] = ~is_h
            residual_count += int(np.count_nonzero(~is_h))
            _heartbeat("S4_residual_partition", end, len(full_tetras), started)
        residual = _open_npy(residual_path, "w+", np.uint32, (residual_count, 4))
        offset = 0
        for start in range(0, len(full_tetras), INDEX_CHUNK):
            end = min(len(full_tetras), start + INDEX_CHUNK)
            block = np.asarray(full_tetras[start:end])[np.asarray(mask[start:end])]
            residual[offset : offset + len(block)] = block
            offset += len(block)
        residual.flush()
        residual_marker.write_text(_now() + "\n", encoding="utf-8")
        del mask
        try:
            residual_mask_path.unlink()
        except OSError:
            pass
    residual = _open_npy(residual_path, "r")
    residual_edge_keys, residual_edge_counts = _build_subset_count_index(residual, 2, radix, "residual_edge", index_root)
    residual_triangle_keys, residual_triangle_counts = _build_subset_count_index(residual, 3, radix, "residual_triangle", index_root)
    residual_vertex_counts_path = index_root / "residual_vertex_counts.npy"
    if not residual_vertex_counts_path.is_file():
        counts = np.bincount(np.asarray(residual).reshape(-1).astype(np.int64), minlength=radix).astype(np.uint64)
        target = _open_npy(residual_vertex_counts_path, "w+", np.uint64, counts.shape)
        target[:] = counts
        target.flush()
    residual_vertex_counts = _open_npy(residual_vertex_counts_path, "r")

    tetra_triangle_keys, _ = _build_subset_count_index(full_tetras, 3, radix, "all_tetra_triangle", index_root)
    full_triangle_codes_path = index_root / "full_triangle_codes.npy"
    if not full_triangle_codes_path.is_file():
        values = np.sort(_encode_rows(full_triangles, radix))
        target = _open_npy(full_triangle_codes_path, "w+", np.uint64, values.shape)
        target[:] = values
        target.flush()
    full_triangle_codes = _open_npy(full_triangle_codes_path, "r")
    pure_l2_codes = _setdiff_sorted(full_triangle_codes, tetra_triangle_keys, index_root / "pure_l2_triangle_codes.npy")
    pure_l2_rows_path = index_root / "pure_l2_triangles.npy"
    if not pure_l2_rows_path.is_file():
        rows = _decode_codes(pure_l2_codes, 3, radix)
        target = _open_npy(pure_l2_rows_path, "w+", np.uint32, rows.shape)
        target[:] = rows
        target.flush()
    pure_l2 = _open_npy(pure_l2_rows_path, "r")
    pure_l2_edge_keys, pure_l2_edge_counts = _build_subset_count_index(pure_l2, 2, radix, "pure_l2_edge", index_root)
    pure_l2_vertex_counts_path = index_root / "pure_l2_vertex_counts.npy"
    if not pure_l2_vertex_counts_path.is_file():
        counts = np.bincount(np.asarray(pure_l2).reshape(-1).astype(np.int64), minlength=radix).astype(np.uint64)
        target = _open_npy(pure_l2_vertex_counts_path, "w+", np.uint64, counts.shape)
        target[:] = counts
        target.flush()
    pure_l2_vertex_counts = _open_npy(pure_l2_vertex_counts_path, "r")

    triangle_edge_keys, _ = _build_subset_count_index(full_triangles, 2, radix, "all_triangle_edge", index_root)
    full_edge_codes_path = index_root / "full_edge_codes.npy"
    if not full_edge_codes_path.is_file():
        values = np.sort(_encode_rows(full_edges, radix))
        target = _open_npy(full_edge_codes_path, "w+", np.uint64, values.shape)
        target[:] = values
        target.flush()
    full_edge_codes = _open_npy(full_edge_codes_path, "r")
    pure_l1_codes = _setdiff_sorted(full_edge_codes, triangle_edge_keys, index_root / "pure_l1_edge_codes.npy")
    pure_l1_rows_path = index_root / "pure_l1_edges.npy"
    if not pure_l1_rows_path.is_file():
        rows = _decode_codes(pure_l1_codes, 2, radix)
        target = _open_npy(pure_l1_rows_path, "w+", np.uint32, rows.shape)
        target[:] = rows
        target.flush()
    pure_l1 = _open_npy(pure_l1_rows_path, "r")

    tetra_triangle_records_path = index_root / "tetra_triangle.records"
    tetra_triangle_marker = index_root / "tetra_triangle.generated"
    if not tetra_triangle_marker.is_file():
        records = np.memmap(tetra_triangle_records_path, mode="w+", dtype=RECORD_DTYPE, shape=(len(full_tetras) * 4,))
        started = time.time()
        for start in range(0, len(full_tetras), INDEX_CHUNK):
            end = min(len(full_tetras), start + INDEX_CHUNK)
            codes = _face_codes(full_tetras[start:end], 3, radix)
            block = slice(start * 4, end * 4)
            records[block]["key"] = codes.reshape(-1)
            records[block]["item"] = np.repeat(np.arange(start, end, dtype=np.uint32), 4)
            _heartbeat("S4_tetra_triangle_index", end, len(full_tetras), started)
        records.flush()
        tetra_triangle_marker.write_text(_now() + "\n", encoding="utf-8")
    tetra_triangle_records = _sort_records(tetra_triangle_records_path, index_root / "tetra_triangle.sorted")

    if not meta_path.is_file():
        _write_json(
            meta_path,
            {
                "dataset": DATASET,
                "split": split,
                "radix": radix,
                "candidate_count": int(len(candidates)),
                "residual_tetra_count": int(len(residual)),
                "pure_l2_triangle_count": int(len(pure_l2)),
                "pure_l1_edge_count": int(len(pure_l1)),
                "representation": "compact_exact_disk_indices_v1",
                "sampling": "NO",
                "completed_at": _now(),
            },
        )
    return {
        "root": index_root,
        "radix": radix,
        "candidates": candidates,
        "labels": labels,
        "candidate_indptr": candidate_indptr,
        "candidate_indices": candidate_indices,
        "penta_tetra_keys": penta_tetra_keys,
        "penta_tetra_counts": penta_tetra_counts,
        "residual_vertex_counts": residual_vertex_counts,
        "residual_edge_keys": residual_edge_keys,
        "residual_edge_counts": residual_edge_counts,
        "residual_triangle_keys": residual_triangle_keys,
        "residual_triangle_counts": residual_triangle_counts,
        "pure_l2_vertex_counts": pure_l2_vertex_counts,
        "pure_l2_edge_keys": pure_l2_edge_keys,
        "pure_l2_edge_counts": pure_l2_edge_counts,
        "pure_l2_codes": pure_l2_codes,
        "pure_l1": pure_l1,
        "full_tetras": full_tetras,
        "tetra_triangle_records": tetra_triangle_records,
    }

def _candidate_graph_values(cid: int, indptr, indices):
    start, end = int(indptr[cid]), int(indptr[cid + 1])
    neighbors_array = np.asarray(indices[start:end], dtype=np.uint32)
    neighbors = set(map(int, neighbors_array))
    common_sum = 0
    two_hop = set()
    for neighbor in neighbors:
        left, right = int(indptr[neighbor]), int(indptr[neighbor + 1])
        other = set(map(int, np.asarray(indices[left:right], dtype=np.uint32)))
        common_sum += len(neighbors & other)
        two_hop.update(other)
    two_hop.discard(cid)
    two_hop.difference_update(neighbors)
    return float(len(neighbors)), float(common_sum), 0.0, float(len(two_hop))

def _pure_l1_context(edges, radix: int):
    adjacency = [set() for _ in range(radix)]
    for left, right in np.asarray(edges):
        adjacency[int(left)].add(int(right))
        adjacency[int(right)].add(int(left))

    pair_cn = np.zeros((radix, radix), dtype=np.float64)
    pair_ra = np.zeros((radix, radix), dtype=np.float64)
    pair_aa = np.zeros((radix, radix), dtype=np.float64)
    pair_jac = np.zeros((radix, radix), dtype=np.float64)
    for left in range(radix):
        for right in range(left + 1, radix):
            ln = adjacency[left] - {right}
            rn = adjacency[right] - {left}
            common = ln & rn
            union = ln | rn
            pair_cn[left, right] = pair_cn[right, left] = len(common)
            pair_ra[left, right] = pair_ra[right, left] = sum(1.0 / len(adjacency[node]) for node in common if adjacency[node])
            pair_aa[left, right] = pair_aa[right, left] = sum(1.0 / math.log(len(adjacency[node]) + 1.0) for node in common if adjacency[node])
            pair_jac[left, right] = pair_jac[right, left] = len(common) / len(union) if union else 0.0
    return adjacency, pair_cn, pair_ra, pair_aa, pair_jac

def _l1_values(candidate, l1_context):
    adjacency, pair_cn, pair_ra, pair_aa, pair_jac = l1_context
    vertices = tuple(map(int, candidate))
    vset = set(vertices)
    pairs = tuple(combinations(vertices, 2))
    degrees = [len(adjacency[v]) for v in vertices]
    one = set().union(*(adjacency[v] for v in vertices)) - vset
    two = set().union(*(adjacency[v] for v in one)) - vset - one if one else set()
    external = sum(len(adjacency[v] - vset) for v in vertices)
    return np.asarray(
        (
            sum(degrees), np.mean(degrees), max(degrees), min(degrees), np.std(degrees),
            sum(pair_cn[a, b] for a, b in pairs),
            sum(pair_ra[a, b] for a, b in pairs),
            sum(pair_aa[a, b] for a, b in pairs),
            sum(pair_jac[a, b] for a, b in pairs) / 10.0,
            len(one), len(two), external,
        ),
        dtype=np.float32,
    )

def _triangle_group(records, triangle_code: int) -> np.ndarray:
    keys = records["key"]
    left = int(np.searchsorted(keys, np.uint64(triangle_code), side="left"))
    right = int(np.searchsorted(keys, np.uint64(triangle_code), side="right"))
    return np.asarray(records["item"][left:right], dtype=np.uint32)

def _tetra_shell_rows(candidate, context):
    radix = context["radix"]
    full_tetras = context["full_tetras"]
    records = context["tetra_triangle_records"]
    t0_rows = np.asarray(list(combinations(map(int, candidate), 4)), dtype=np.uint32)

    t0_ids = set()
    for tetra in t0_rows:
        tri_code = int(_encode_rows(tetra[:3].reshape(1, 3), radix)[0])
        for row_id in _triangle_group(records, tri_code):
            if np.array_equal(full_tetras[int(row_id)], tetra):
                t0_ids.add(int(row_id))
                break
    if len(t0_ids) != 5:
        raise RuntimeError("compact T0 boundary lookup failed")

    def expand(row_ids):
        result = set()
        for row_id in row_ids:
            tetra = np.asarray(full_tetras[row_id], dtype=np.uint32).reshape(1, 4)
            for tri_code in _face_codes(tetra, 3, radix)[0]:
                result.update(map(int, _triangle_group(records, int(tri_code))))
        return result

    t1_ids = expand(t0_ids) - t0_ids
    t2_ids = expand(t1_ids) - t0_ids - t1_ids
    return (
        np.asarray(full_tetras[sorted(t0_ids)], dtype=np.uint32),
        np.asarray(full_tetras[sorted(t0_ids | t1_ids)], dtype=np.uint32),
        np.asarray(full_tetras[sorted(t0_ids | t1_ids | t2_ids)], dtype=np.uint32),
    )

def _streaming_array_hash(array, rows_per_chunk: int = 100000) -> str:
    values = np.asarray(array)
    digest = hashlib.sha256()
    digest.update(f"{values.dtype.str}|{values.shape}|".encode())
    for start in range(0, len(values), rows_per_chunk):
        digest.update(np.ascontiguousarray(values[start : start + rows_per_chunk]).tobytes())
    return digest.hexdigest()

def _build_feature_rows(context, outputs, progress_path: Path, structural) -> None:
    candidates = context["candidates"]
    radix = context["radix"]
    completed = int(_read_json(progress_path).get("completed_rows", 0)) if progress_path.is_file() else 0
    l1_context = _pure_l1_context(context["pure_l1"], radix)
    started = time.time()
    for start in range(completed, len(candidates), ROW_CHUNK):
        end = min(len(candidates), start + ROW_CHUNK)
        block = np.asarray(candidates[start:end], dtype=np.uint32)
        tetra_codes = _face_codes(block, 4, radix)
        triangle_codes = _face_codes(block, 3, radix)
        edge_codes = _face_codes(block, 2, radix)

        cofaces = _lookup_counts(context["penta_tetra_keys"], context["penta_tetra_counts"], tetra_codes).astype(np.float64)
        coface_sum = cofaces.sum(axis=1)
        h = np.zeros((len(block), 17), dtype=np.float32)
        h[:, 3] = ((coface_sum * coface_sum - np.square(cofaces).sum(axis=1)) / 2.0).astype(np.float32)
        h[:, 4] = coface_sum
        h[:, 5] = cofaces.mean(axis=1)
        h[:, 6] = cofaces.min(axis=1)
        h[:, 7] = cofaces.max(axis=1)
        h[:, 8] = cofaces.std(axis=1)
        boundary = cofaces + 1.0
        h[:, 13] = boundary.sum(axis=1)
        h[:, 14] = boundary.mean(axis=1)
        h[:, 15] = boundary.max(axis=1)
        h[:, 16] = boundary.std(axis=1)

        rv = np.asarray(context["residual_vertex_counts"])[block]
        re = _lookup_counts(context["residual_edge_keys"], context["residual_edge_counts"], edge_codes)
        rt = _lookup_counts(context["residual_triangle_keys"], context["residual_triangle_counts"], triangle_codes)
        f1 = rv.sum(axis=1).astype(np.int64)
        f2 = re.sum(axis=1).astype(np.int64)
        f3 = rt.sum(axis=1).astype(np.int64)
        n3 = f3
        n2 = f2 - 3 * n3
        n1 = f1 - 2 * n2 - 3 * n3
        if np.any(n1 < 0) or np.any(n2 < 0):
            raise RuntimeError("residual tetra inclusion-count audit failed")
        l3 = np.column_stack(
            (
                rv.sum(axis=1), rv.mean(axis=1), rv.max(axis=1), rv.min(axis=1), rv.std(axis=1),
                rt.sum(axis=1), rt.mean(axis=1), rt.max(axis=1), rt.std(axis=1),
                n1, n2, n3, np.zeros(len(block), dtype=np.int64), f1,
            )
        ).astype(np.float32)

        lv = np.asarray(context["pure_l2_vertex_counts"])[block]
        le = _lookup_counts(context["pure_l2_edge_keys"], context["pure_l2_edge_counts"], edge_codes)
        lt = _lookup_membership(context["pure_l2_codes"], triangle_codes).astype(np.uint32)
        g1 = lv.sum(axis=1).astype(np.int64)
        g2 = le.sum(axis=1).astype(np.int64)
        g3 = lt.sum(axis=1).astype(np.int64)
        m3 = g3
        m2 = g2 - 3 * m3
        m1 = g1 - 2 * m2 - 3 * m3
        if np.any(m1 < 0) or np.any(m2 < 0):
            raise RuntimeError("pure L2 inclusion-count audit failed")
        l2 = np.column_stack(
            (
                lv.sum(axis=1), lv.mean(axis=1), lv.max(axis=1), lv.min(axis=1), lv.std(axis=1),
                le.sum(axis=1), le.mean(axis=1), le.max(axis=1), le.std(axis=1),
                m1, m2, m3, g1,
            )
        ).astype(np.float32)

        local1 = np.empty((len(block), 39), dtype=np.float32)
        local2 = np.empty((len(block), 39), dtype=np.float32)
        local3 = np.empty((len(block), 39), dtype=np.float32)
        l1 = np.empty((len(block), 12), dtype=np.float32)
        for offset, candidate in enumerate(block):
            cid = start + offset
            h[offset, 9:13] = _candidate_graph_values(cid, context["candidate_indptr"], context["candidate_indices"])
            l1[offset] = _l1_values(candidate, l1_context)
            shells = _tetra_shell_rows(candidate, context)
            for target, tetras in zip((local1, local2, local3), shells):
                tetra_tuples = {tuple(map(int, row)) for row in tetras}
                triangles = {face for tetra in tetra_tuples for face in combinations(tetra, 3)}
                edges = {face for triangle in triangles for face in combinations(triangle, 2)}
                target[offset] = structural.allowed_structural_features(tuple(map(int, candidate)), tetra_tuples, triangles, edges)

        outputs["C0*"][start:end] = h
        outputs["C1*"][start:end] = np.column_stack((h, local1))
        outputs["C2*"][start:end] = np.column_stack((h, local1, local2))
        outputs["C3*"][start:end] = np.column_stack((h, local1, local2, local3))
        outputs["Full"][start:end] = np.column_stack((h, l3, l2, l1))
        for output in outputs.values():
            output.flush()
        _write_json(progress_path, {"dataset": DATASET, "completed_rows": end, "total_rows": len(candidates), "updated_at": _now()})
        _heartbeat("S4_features_memory_safe", end, len(candidates), started)

def build_features(bundle: Path, split: str, feature_names: dict[str, tuple[str, ...]]):
    if split not in ("train", "eval"):
        raise ValueError(split)
    project = bundle / "src" / "project"
    sys.path.insert(0, str(project))
    sys.path.insert(0, str(project / "scripts"))
    from scripts import run_task_core_confirmatory_v2 as structural

    results = bundle / "server_results" / "k4_server_benchmark_expansion_v1"
    feature_root = results / "stageS4_mlp" / "features" / DATASET / split
    paths = {model: feature_root / f"{model.replace('*', 'star')}.npy" for model in MODELS}
    meta_path = feature_root / "feature_meta.json"
    if meta_path.is_file() and all(path.is_file() for path in paths.values()):
        return
    context = _build_compact_context(bundle, split)
    feature_root.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for model in MODELS:
        path = paths[model]
        if path.is_file():
            matrix = _open_npy(path, "r+")
            if matrix.shape != (len(context["candidates"]), MODEL_WIDTHS[model]) or matrix.dtype != np.float32:
                raise RuntimeError(f"partial feature matrix identity mismatch:{model}")
        else:
            matrix = _open_npy(path, "w+", np.float32, (len(context["candidates"]), MODEL_WIDTHS[model]))
        outputs[model] = matrix
    progress_path = feature_root / "feature_progress.json"
    _build_feature_rows(context, outputs, progress_path, structural)

    audits = []
    for left, right, width in (("C0*", "C1*", 17), ("C1*", "C2*", 56), ("C2*", "C3*", 95)):
        maximum = 0.0
        for start in range(0, len(outputs[left]), ROW_CHUNK):
            end = min(len(outputs[left]), start + ROW_CHUNK)
            maximum = max(maximum, float(np.max(np.abs(outputs[left][start:end].astype(np.float64) - outputs[right][start:end, :width].astype(np.float64)))))
        audits.append({"relation": f"{left}_in_{right}", "max_abs_difference": maximum, "status": "PASS" if maximum <= 1e-12 else "FAIL"})
    if any(row["status"] != "PASS" for row in audits):
        raise RuntimeError("FEATURE_INFORMATION_NESTING_FAIL:congress-bills:" + split)
    census = _read_json(results / "stageS2_census" / DATASET / split / "census.json")
    labels = np.load(results / "stageS2_census" / DATASET / split / "labels.npy", mmap_mode="r")
    meta = {
        "dataset": DATASET,
        "split": split,
        "candidate_hash": census["candidate_hash"],
        "label_hash": _streaming_array_hash(labels),
        "candidate_count": int(len(labels)),
        "feature_names": {model: list(feature_names[model]) for model in MODELS},
        "input_dimensions": MODEL_WIDTHS,
        "feature_hashes": {model: _streaming_array_hash(outputs[model]) for model in MODELS},
        "feature_information_nesting": "STRICT",
        "C0_IN_C1": "YES",
        "C1_IN_C2": "YES",
        "C2_IN_C3": "YES",
        "INHERITED_FEATURE_VALUES_IDENTICAL": "YES",
        "OFFICIAL_FULL": "FULL_STANDARD_RESIDUAL",
        "representation": "compact_exact_disk_indices_v1",
        "sampling": "NO",
        "audits": audits,
        "completed_at": _now(),
    }
    _write_json(meta_path, meta)
    print(f"S4_FEATURES dataset={DATASET} split={split} rows={len(labels)} MEMORY_SAFE=YES SAMPLING=NO checkpoint=DONE", flush=True)
    index_root = results / "stageS4_mlp" / "compact_index" / DATASET / split
    del labels, context, outputs
    gc.collect()
    if os.environ.get("K4_CONGRESS_KEEP_COMPACT_INDEX", "0") != "1" and index_root.is_dir():
        try:
            shutil.rmtree(index_root)
            print(f"DISK_RECLAIM dataset={DATASET} split={split} compact_index=REMOVED", flush=True)
        except OSError as exc:
            print(f"DISK_RECLAIM dataset={DATASET} split={split} compact_index=RETAINED error={exc}", flush=True)

def run(bundle: Path) -> None:
    _reclaim_completed_intermediates(bundle)
    _disk_preflight(bundle)
    project = bundle / "src" / "project"
    sys.path.insert(0, str(project))
    sys.path.insert(0, str(project / "scripts"))
    from scripts import run_k4_task_core_perturbation_v1 as k4old

    h = tuple(k4old.H_NAMES)
    block = tuple(k4old.L3_NAMES + k4old.L2_NAMES + k4old.L1_NAMES)
    names = {
        "C0*": h,
        "C1*": h + tuple("T0__" + name for name in block),
        "C2*": h + tuple("T0__" + name for name in block) + tuple("TLE1__" + name for name in block),
        "C3*": h + tuple("T0__" + name for name in block) + tuple("TLE1__" + name for name in block) + tuple("TLE2__" + name for name in block),
        "Full": tuple(k4old.FULL_NAMES),
    }
    for split in ("train", "eval"):
        build_features(bundle, split, names)

def selftest() -> None:
    radix = 19
    rows = np.asarray([(0, 2, 5, 9), (1, 3, 7, 11)], dtype=np.uint32)
    if not np.array_equal(_decode_codes(_encode_rows(rows, radix), 4, radix), rows):
        raise AssertionError("compact simplex codec failed")

    candidate = (0, 1, 2, 3, 4)
    residual = {
        (0, 1, 5, 6), (0, 2, 3, 6), (1, 2, 3, 7),
        (0, 5, 6, 7), (1, 4, 6, 7), (2, 4, 5, 7),
    }
    vertex_counts = Counter(v for tetra in residual for v in tetra)
    edge_counts = Counter(edge for tetra in residual for edge in combinations(tetra, 2))
    triangle_counts = Counter(triangle for tetra in residual for triangle in combinations(tetra, 3))
    f1 = sum(vertex_counts[v] for v in candidate)
    f2 = sum(edge_counts[edge] for edge in combinations(candidate, 2))
    f3 = sum(triangle_counts[triangle] for triangle in combinations(candidate, 3))
    n3 = f3
    n2 = f2 - 3 * n3
    n1 = f1 - 2 * n2 - 3 * n3
    direct = Counter(len(set(candidate) & set(tetra)) for tetra in residual)
    if (n1, n2, n3, 0, f1) != (direct[1], direct[2], direct[3], direct[4], sum(k * v for k, v in direct.items())):
        raise AssertionError("exact residual inclusion identities failed")

    adjacency = ({1, 2}, {0, 2}, {0, 1, 3}, {2})
    indptr = np.asarray([0, 2, 4, 7, 8], dtype=np.uint64)
    indices = np.asarray([1, 2, 0, 2, 0, 1, 3, 2], dtype=np.uint32)
    observed = _candidate_graph_values(0, indptr, indices)
    neighbors = set(adjacency[0])
    two_hop = set().union(*(adjacency[n] for n in neighbors)) - {0} - neighbors
    expected = (
        float(len(neighbors)),
        float(sum(len(neighbors & set(adjacency[n])) for n in neighbors)),
        0.0,
        float(len(two_hop)),
    )
    if observed != expected:
        raise AssertionError(f"compact candidate CSR failed:{observed}!={expected}")
    print("CONGRESS_MEMORY_SAFE_SELFTEST=PASS", flush=True)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return
    bundle = Path(__file__).resolve().parents[1]
    run(bundle)

if __name__ == "__main__":
    main()
