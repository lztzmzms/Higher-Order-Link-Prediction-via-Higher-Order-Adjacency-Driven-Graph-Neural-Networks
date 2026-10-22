"""Leakage-safe legacy 31-feature candidate-law prior for K=2 closure."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score


IMPL_VERSION = "legacy31_linear_v1"


def _hash_array(value):
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def _load_events(data_root, dataset, event_order="raw"):
    raw_dir = Path(data_root) / "dataset" / "raw_data" / str(dataset)
    counts = np.loadtxt(raw_dir / f"{dataset}-nverts.txt", dtype=np.int64, ndmin=1)
    flat = np.loadtxt(raw_dir / f"{dataset}-simplices.txt", dtype=np.int64, ndmin=1)
    times = np.loadtxt(raw_dir / f"{dataset}-times.txt", dtype=np.int64, ndmin=1)
    if counts.size != times.size or int(counts.sum()) != flat.size:
        raise ValueError(f"Malformed candidate-law raw data for {dataset}")
    rows, offset = [], 0
    for row_id, (count, timestamp) in enumerate(zip(counts.tolist(), times.tolist())):
        nodes = tuple(sorted(set(map(int, flat[offset:offset + int(count)].tolist()))))
        offset += int(count)
        rows.append((int(timestamp), int(row_id), nodes))
    if event_order == "timestamp":
        rows.sort(key=lambda row: (row[0], row[1]))
    elif event_order != "raw":
        raise ValueError(f"Unsupported event_order={event_order}")
    return [(row[0], row[2]) for row in rows]


def _history_statistics(events, history_ratio):
    history = events[:int(len(events) * float(history_ratio))]
    recent20_start = int(0.8 * len(history))
    recent40_start = int(0.6 * len(history))
    total, recent20, recent40 = Counter(), Counter(), Counter()
    last_time, observed_triangles = {}, set()
    for index, (timestamp, nodes) in enumerate(history):
        for edge in combinations(nodes, 2):
            total[edge] += 1
            if index >= recent20_start:
                recent20[edge] += 1
            if index >= recent40_start:
                recent40[edge] += 1
            last_time[edge] = timestamp
        observed_triangles.update(combinations(nodes, 3))
    adjacency = defaultdict(set)
    for left, right in total:
        adjacency[left].add(right)
        adjacency[right].add(left)
    upper = Counter()
    for triangle in observed_triangles:
        for edge in combinations(triangle, 2):
            upper[edge] += 1
    return {
        "total": total, "recent20": recent20, "recent40": recent40,
        "last_time": last_time, "current_time": int(history[-1][0]),
        "adjacency": adjacency, "upper": upper,
    }


def _summary(values):
    array = np.asarray(values, dtype=np.float64)
    return float(array.min()), float(array.mean()), float(array.max())


def _feature_row(triangle, stats):
    triangle = tuple(sorted(map(int, triangle)))
    edges = list(combinations(triangle, 2))
    total = [float(stats["total"].get(edge, 0)) for edge in edges]
    recent20 = [float(stats["recent20"].get(edge, 0)) for edge in edges]
    recent40 = [float(stats["recent40"].get(edge, 0)) for edge in edges]
    last = [float(stats["last_time"].get(edge, stats["current_time"])) for edge in edges]
    ages = [float(max(0.0, stats["current_time"] - value)) for value in last]
    common = set(stats["adjacency"][triangle[0]])
    common.intersection_update(stats["adjacency"][triangle[1]])
    common.intersection_update(stats["adjacency"][triangle[2]])
    common.difference_update(triangle)
    upper = [float(stats["upper"].get(edge, 0)) for edge in edges]
    return [
        *[math.log1p(value) for value in total], *_summary(total),
        *[math.log1p(value) for value in recent20], *_summary(recent20),
        *[math.log1p(value) for value in recent40], *_summary(recent40),
        *_summary(ages), *_summary(last), float(len(common)),
        *[math.log1p(value) for value in upper], *_summary(upper),
    ]


def _feature_matrix(triplets, stats):
    return np.asarray([_feature_row(triangle, stats) for triangle in triplets], dtype=np.float32)


def _split_indices(labels, supervision_mask, val_seed, val_ratio=0.2):
    labels = np.asarray(labels, dtype=np.int64)
    eligible = np.flatnonzero(np.asarray(supervision_mask, dtype=bool))
    positive = eligible[labels[eligible] == 1]
    negative = eligible[labels[eligible] != 1]
    rng = np.random.default_rng(int(val_seed))
    rng.shuffle(positive)
    rng.shuffle(negative)
    val = np.concatenate([
        positive[:int(positive.size * val_ratio)],
        negative[:int(negative.size * val_ratio)],
    ])
    fit = np.setdiff1d(eligible, val, assume_unique=False)
    return fit.astype(np.int64), val.astype(np.int64)


def _fit_linear_prior(x_fit, y_fit, x_val, y_val, seed, device):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    device = torch.device(device)
    model = nn.Linear(x_fit.shape[1], 1).to(device)
    x_fit_t = torch.from_numpy(x_fit.astype(np.float32, copy=False)).to(device)
    y_fit_t = torch.from_numpy(y_fit.astype(np.float32, copy=False)).to(device)
    x_val_t = torch.from_numpy(x_val.astype(np.float32, copy=False)).to(device)
    positive_weight = float((y_fit == 0).sum() / max(1, (y_fit == 1).sum()))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-3)
    best_ap, best_epoch, best_state = -1.0, 0, None
    for epoch in range(400):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x_fit_t).view(-1), y_fit_t)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_score = torch.sigmoid(model(x_val_t).view(-1)).cpu().numpy()
        val_ap = float(average_precision_score(y_val, val_score))
        if val_ap > best_ap:
            best_ap, best_epoch = val_ap, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch - best_epoch >= 50:
            break
    if best_state is None:
        raise RuntimeError("candidate-law prior did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model, best_ap, best_epoch, device


def _predict(model, device, features):
    with torch.no_grad():
        tensor = torch.from_numpy(features.astype(np.float32, copy=False)).to(device)
        return torch.sigmoid(model(tensor).view(-1)).cpu().numpy().astype(np.float64)


def build_or_load_candidate_law_prior(
    dataset, train_triplets, train_labels, supervision_mask,
    test_target_triplets, data_root, event_order, val_seed,
    cache_dir, device="cpu", seed=41,
):
    train_triplets = np.asarray([tuple(map(int, row)) for row in train_triplets], dtype=np.int64)
    test_target_triplets = np.asarray(
        [tuple(map(int, row)) for row in test_target_triplets], dtype=np.int64
    )
    train_labels = np.asarray(train_labels, dtype=np.int64)
    supervision_mask = np.asarray(supervision_mask, dtype=bool)
    payload = {
        "impl": IMPL_VERSION, "dataset": str(dataset), "event_order": str(event_order),
        "val_seed": int(val_seed), "seed": int(seed),
        "train_triplet_hash": _hash_array(train_triplets),
        "train_label_hash": _hash_array(train_labels),
        "supervision_hash": _hash_array(supervision_mask),
        "test_triplet_hash": _hash_array(test_target_triplets),
    }
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:20]
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"candidate_law_{dataset}_{key}.npz"
    audit_path = cache_dir / f"candidate_law_{dataset}_{key}.json"
    if cache_path.is_file() and audit_path.is_file():
        cached = np.load(cache_path)
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        return cached["train_scores"], cached["test_scores"], audit

    events = _load_events(data_root, dataset, event_order=event_order)
    train_features = _feature_matrix(train_triplets, _history_statistics(events, 0.60))
    test_features = _feature_matrix(test_target_triplets, _history_statistics(events, 0.80))
    fit_idx, val_idx = _split_indices(train_labels, supervision_mask, val_seed)
    mean = train_features[fit_idx].mean(axis=0, keepdims=True)
    std = train_features[fit_idx].std(axis=0, keepdims=True)
    std = np.where(std > 1e-6, std, 1.0)
    standardized_train = ((train_features - mean) / std).astype(np.float32)
    standardized_test = ((test_features - mean) / std).astype(np.float32)
    model, best_val_ap, best_epoch, model_device = _fit_linear_prior(
        standardized_train[fit_idx], train_labels[fit_idx],
        standardized_train[val_idx], train_labels[val_idx],
        seed=seed, device=device,
    )
    train_scores = _predict(model, model_device, standardized_train)
    test_scores = _predict(model, model_device, standardized_test)
    audit = {
        **payload,
        "cache_key": key,
        "feature_dim": int(train_features.shape[1]),
        "fit_count": int(fit_idx.size), "validation_count": int(val_idx.size),
        "fit_positives": int(train_labels[fit_idx].sum()),
        "validation_positives": int(train_labels[val_idx].sum()),
        "validation_ap": float(best_val_ap), "best_epoch": int(best_epoch),
        "train_score_hash": _hash_array(train_scores),
        "test_score_hash": _hash_array(test_scores),
        "history_only": True,
    }
    np.savez_compressed(cache_path, train_scores=train_scores, test_scores=test_scores)
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return train_scores, test_scores, audit
