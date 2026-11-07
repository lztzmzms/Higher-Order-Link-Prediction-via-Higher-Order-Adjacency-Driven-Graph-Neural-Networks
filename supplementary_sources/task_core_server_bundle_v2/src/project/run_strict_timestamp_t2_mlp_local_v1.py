from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import math
import os
import platform
import random
import statistics
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import run_strict_timestamp_chronological_core_v1 as core

ROOT = Path(__file__).resolve().parent
OLD_ROOT = ROOT / "results" / "strict_timestamp_chronological_core_v1"
OUT_ROOT = ROOT / "results" / "strict_timestamp_t2_mlp_local_v1"
PROTOCOL = "strict_timestamp_block_v1"
TASK = "T2"
FEATURE_VERSION = "strict_v1_archived_feature_bundle"
MLP_SEEDS = tuple(range(21001, 21006))
R1_SEEDS = tuple(range(22001, 22006))
PILOT_CASES = [(2, "email-Eu"), (2, "email-Enron"), (3, "NDC-substances")]
ALL_CASES = [
    (3, "email-Eu"),
    (3, "NDC-substances"),
    (3, "tags-math-sx"),
    (2, "contact-high-school"),
    (2, "contact-primary-school"),
    (2, "email-Enron"),
    (2, "email-Eu"),
    (2, "NDC-substances"),
]

HIDDEN_1 = 64
HIDDEN_2 = 32
DROPOUT = 0.20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 100
PATIENCE = 10
MINIMUM_EPOCHS = 20
DEFAULT_BATCH_SIZE = 1024

MLP_TABLE_FIELDS = [
    "order", "dataset", "seed", "H_LL", "Full_LL", "InfoLoss",
    "H_Brier", "Full_Brier", "H_AP", "Full_AP", "H_AUC", "Full_AUC",
    "H_Lift", "Full_Lift", "best_epoch_H", "best_epoch_Full",
    "effective_batch_size_H", "effective_batch_size_Full",
    "CANDIDATE_LEVEL_METRIC_RECOMPUTATION",
]
MLP_AGG_FIELDS = [
    "order", "dataset", "status", "n_seeds", "H_LL_mean", "H_LL_std",
    "Full_LL_mean", "Full_LL_std", "InfoLoss_mean", "InfoLoss_std",
    "InfoLoss_median", "InfoLoss_min", "InfoLoss_max", "n_H_better",
    "n_Full_better", "n_abs_InfoLoss_lt_001", "n_abs_InfoLoss_lt_005",
    "n_abs_InfoLoss_lt_010", "seed_matched_d_mean", "seed_matched_d_std",
    "seed_matched_d_median", "seed_matched_d_min", "seed_matched_d_max",
    "CANDIDATE_LEVEL_METRIC_RECOMPUTATION",
]
R1_TABLE_FIELDS = [
    "order", "dataset", "mlp_seed", "r1_seed", "valid", "changed_fraction",
    "residual_jaccard", "candidate_neighbor_jaccard",
    "candidate_ego_edge_jaccard", "LL_real", "LL_R1", "delta_LL",
    "Brier_real", "Brier_R1", "delta_Brier", "MAE_p", "RMSE_p",
    "median_abs_delta_p", "Q90_abs_delta_p", "Q95_abs_delta_p",
    "Q99_abs_delta_p", "max_abs_delta_p", "pearson_p", "spearman_p",
    "AP_real", "AP_R1", "delta_AP", "AUC_real", "AUC_R1", "delta_AUC",
    "Lift_real", "Lift_R1", "delta_Lift", "prediction_flip_rate",
    "candidate_hash_match", "boundary_hash_match", "H_k_hash_match",
    "residual_L2_hash_match", "H_derived_feature_max_abs_diff",
    "L2_feature_max_abs_diff", "nodewise_residual_degree_exact",
    "residual_edge_count_exact", "lower_adjacency_exact",
    "MODEL_RETRAINED_ON_R1", "SCALER_REFIT_ON_R1", "VALID_R1",
    "CANDIDATE_LEVEL_METRIC_RECOMPUTATION", "failure_reason",
]

class ResourceLimit(Exception):
    pass

class TrainingFailure(Exception):
    pass

class CapacityMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_1),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_1, HIDDEN_2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)

def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")

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
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["status"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

def read_csv(path: Path):
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def hash_array(array) -> str:
    array = np.ascontiguousarray(np.asarray(array))
    return sha256_bytes(f"{array.dtype.str}|{array.shape}|".encode() + array.tobytes())

def fmt(value, digits=9):
    if value is None or value == "":
        return "NA"
    try:
        number = float(value)
        return f"{number:.{digits}f}" if math.isfinite(number) else "NA"
    except (TypeError, ValueError):
        return str(value)

def finite_values(values):
    return [float(value) for value in values if value not in (None, "") and math.isfinite(float(value))]

def mean(values):
    values = finite_values(values)
    return float(np.mean(values)) if values else None

def std(values):
    values = finite_values(values)
    return float(np.std(values)) if values else None

def median(values):
    values = finite_values(values)
    return float(np.median(values)) if values else None

def case_key(order, dataset):
    return f"K{order}|{dataset}|T2"

def case_root(order, dataset):
    return OUT_ROOT / ("01_k3" if order == 3 else "02_k2") / dataset / "T2"

def model_root(order, dataset, feature_set, seed):
    return OUT_ROOT / "03_models" / f"K{order}" / dataset / "T2" / feature_set / f"seed_{seed}"

def prediction_root(order, dataset):
    return OUT_ROOT / "04_predictions" / f"K{order}" / dataset / "T2"

def r1_root(order, dataset, mlp_seed, r1_seed):
    return OUT_ROOT / "05_r1" / f"K{order}" / dataset / "T2" / f"mlp_seed_{mlp_seed}" / f"r1_seed_{r1_seed}"

def set_seed(seed: int, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

def resolve_device(requested: str):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise TrainingFailure("CUDA requested but torch.cuda.is_available() is false")
    return torch.device(requested)

def device_metadata(device):
    result = {"device": str(device), "torch_version": torch.__version__, "cuda_available": bool(torch.cuda.is_available())}
    if torch.cuda.is_available():
        result["cuda_version"] = torch.version.cuda
        result["gpu_name"] = torch.cuda.get_device_name(0)
    else:
        result["cuda_version"] = None
        result["gpu_name"] = None
    return result

def metric_row(labels, probabilities):
    y = np.asarray(labels, dtype=np.int8)
    p = np.asarray(probabilities, dtype=np.float64)
    prevalence = float(np.mean(y)) if len(y) else 0.0
    ap = float(average_precision_score(y, p)) if int(y.sum()) else None
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None
    return {
        "LogLoss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])),
        "Brier": float(brier_score_loss(y, p)),
        "AP": ap,
        "AUC": auc,
        "Lift": ap / prevalence if ap is not None and prevalence else None,
    }

def metric_equal(left, right, tolerance=1e-12):
    for key in ("LogLoss", "Brier", "AP", "AUC"):
        a, b = left.get(key), right.get(key)
        if a is None or b is None:
            if a != b:
                return False
        elif abs(float(a) - float(b)) > tolerance:
            return False
    return True

def pearson(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])

def spearman(left, right):
    left_rank = np.argsort(np.argsort(np.asarray(left), kind="mergesort"), kind="mergesort")
    right_rank = np.argsort(np.argsort(np.asarray(right), kind="mergesort"), kind="mergesort")
    return pearson(left_rank, right_rank)

def load_npz_arrays(path: Path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}

def matrix_from_arrays(arrays, names):
    return np.column_stack([arrays[name] for name in names]).astype(np.float32) if names else np.empty((len(next(iter(arrays.values()))), 0), dtype=np.float32)

def build_task_for_mlp(order, dataset, events, context_end, label_start, label_end, name):
    old_caps = core.k2legacy.RESOURCE_CAP_DATASETS
    core.k2legacy.RESOURCE_CAP_DATASETS = set()
    try:
        task = core.build_task(order, dataset, events, context_end, label_start, label_end, name)
    finally:
        core.k2legacy.RESOURCE_CAP_DATASETS = old_caps
    return task

def load_audit(dataset):
    cache = OLD_ROOT / "01_timestamp_audit" / "audit_cache.json"
    if cache.is_file():
        values = read_json(cache)
        if dataset in values:
            return values[dataset]
    return core.audit_dataset(dataset)[0]

def load_case_data(order, dataset):
    old_task_dir = OLD_ROOT / ("03_k3" if order == 3 else "04_k2") / dataset / "T2"
    summary_path = old_task_dir / "task_summary.json"
    train_features_path = old_task_dir / "real_train_features.npz"
    eval_features_path = old_task_dir / "real_eval_features.npz"
    if not summary_path.is_file() or not train_features_path.is_file() or not eval_features_path.is_file():
        raise ResourceLimit("STRICT_V1_FEATURE_BUNDLE_MISSING")
    summary = read_json(summary_path)
    if summary.get("status") != "COMPLETE":
        raise ResourceLimit("STRICT_V1_CASE_NOT_COMPLETE")
    paths = core.raw_paths(dataset)
    events = core.load_timestamped_simplices(*paths, event_order="timestamp")[0]
    audit = load_audit(dataset)
    n = len(events)
    c40, c60, c80 = int(audit["cut_40"]), int(audit["cut_60"]), int(audit["cut_80"])
    if not (0 < c40 < c60 < c80 < n):
        raise TrainingFailure("INVALID_STRICT_TIMESTAMP_CUTS")
    train = build_task_for_mlp(order, dataset, events, c60, c60, c80, "T2_TRAIN")
    evaluation = build_task_for_mlp(order, dataset, events, c80, c80, n, "T2_EVAL")
    expected = {
        "train": (summary.get("candidate_hash_train"), int(summary.get("train_candidate_count"))),
        "eval": (summary.get("candidate_hash_eval"), int(summary.get("eval_candidate_count"))),
    }
    actual = {"train": (train["candidate_hash"], len(train["candidates"])), "eval": (evaluation["candidate_hash"], len(evaluation["candidates"]))}
    if actual != expected:
        raise TrainingFailure(f"CANDIDATE_PROTOCOL_MISMATCH expected={expected} actual={actual}")
    gate = core.timestamp_gate(events, c60, c80, c80, n, (0, c40, c60, c80, n))
    if gate["no_timestamp_group_crosses_split"] != "YES":
        raise TrainingFailure("NO_TIMESTAMP_GROUP_CROSSES_SPLIT=NO")
    train_arrays = load_npz_arrays(train_features_path)
    eval_arrays = load_npz_arrays(eval_features_path)
    h_names, l1_names, l2_names = core.feature_names(order)
    all_names = tuple(h_names) + tuple(l1_names) + tuple(l2_names)
    if set(train_arrays) != set(eval_arrays) or not set(all_names).issubset(train_arrays):
        raise TrainingFailure("FEATURE_NAME_SET_MISMATCH")
    x_train_h = matrix_from_arrays(train_arrays, h_names)
    x_eval_h = matrix_from_arrays(eval_arrays, h_names)
    x_train_full = matrix_from_arrays(train_arrays, all_names)
    x_eval_full = matrix_from_arrays(eval_arrays, all_names)
    if x_train_h.shape[0] != len(train["labels"]) or x_eval_h.shape[0] != len(evaluation["labels"]):
        raise TrainingFailure("FEATURE_LABEL_ROW_ALIGNMENT_FAIL")
    if not all(np.isfinite(matrix).all() for matrix in (x_train_h, x_eval_h, x_train_full, x_eval_full)):
        raise TrainingFailure("FEATURE_NAN_OR_INF")
    train_dec = core.decompose(order, train)
    eval_dec = core.decompose(order, evaluation)
    return {
        "order": order, "dataset": dataset, "events": events, "audit": audit, "gate": gate,
        "cut40": c40, "cut60": c60, "cut80": c80, "n_events": n,
        "train": train, "eval": evaluation, "train_dec": train_dec, "eval_dec": eval_dec,
        "train_arrays": train_arrays, "eval_arrays": eval_arrays,
        "h_names": tuple(h_names), "l1_names": tuple(l1_names), "l2_names": tuple(l2_names),
        "all_names": all_names, "X_train_H": x_train_h, "X_eval_H": x_eval_h,
        "X_train_Full": x_train_full, "X_eval_Full": x_eval_full,
        "summary": summary, "candidate_hash_train": train["candidate_hash"],
        "candidate_hash_eval": evaluation["candidate_hash"],
        "label_hash_train": hash_array(np.asarray(train["labels"], dtype=np.int8)),
        "label_hash_eval": hash_array(np.asarray(evaluation["labels"], dtype=np.int8)),
        "feature_hash_train_H": hash_array(x_train_h), "feature_hash_eval_H": hash_array(x_eval_h),
        "feature_hash_train_Full": hash_array(x_train_full), "feature_hash_eval_Full": hash_array(x_eval_full),
    }

def save_case_protocol(data):
    order, dataset = data["order"], data["dataset"]
    root = case_root(order, dataset)
    payload = {
        "order": f"K{order}", "dataset": dataset, "task": "T2", "protocol": PROTOCOL,
        "strict_timestamp_block_order": "YES", "NO_TIMESTAMP_GROUP_CROSSES_SPLIT": "YES",
        "full_dataset": "YES", "event_subsampling": "NO", "candidate_subsampling": "NO",
        "negative_subsampling": "NO", "node_subsampling": "NO", "time_truncation": "NO",
        "train": "W0+W1 observed -> W2 labels", "eval": "W0+W1+W2 observed -> W3 labels",
        "T2_EVAL_USED_FOR_MODEL_SELECTION": "NO", "architecture_tuned_per_dataset": "NO",
        "MLP_FEATURE_PROTOCOL_MATCH_LR": "YES", "SEED_MATCHED_H_VS_FULL": "YES",
        "MODEL_RETRAINED_ON_R1": "NO", "SCALER_REFIT_ON_R1": "NO",
        "candidate_hash_train": data["candidate_hash_train"], "candidate_hash_eval": data["candidate_hash_eval"],
        "label_hash_train": data["label_hash_train"], "label_hash_eval": data["label_hash_eval"],
        "feature_hash_train_H": data["feature_hash_train_H"], "feature_hash_eval_H": data["feature_hash_eval_H"],
        "feature_hash_train_Full": data["feature_hash_train_Full"], "feature_hash_eval_Full": data["feature_hash_eval_Full"],
        "feature_names": {"H": list(data["h_names"]), "L1": list(data["l1_names"]), "L2": list(data["l2_names"]), "Full": list(data["all_names"])},
        "cuts": {"cut40": data["cut40"], "cut60": data["cut60"], "cut80": data["cut80"]},
        "n_raw_events": data["n_events"], "n_train_candidates": len(data["train"]["candidates"]), "n_eval_candidates": len(data["eval"]["candidates"]),
    }
    write_json(root / "00_protocol.json", payload)
    np.savez_compressed(root / "candidate_labels.npz",
                        train_candidate_id=np.arange(len(data["train"]["candidates"]), dtype=np.int64),
                        train_candidate=np.asarray(data["train"]["candidates"], dtype=np.int64),
                        train_y=np.asarray(data["train"]["labels"], dtype=np.int8),
                        eval_candidate_id=np.arange(len(data["eval"]["candidates"]), dtype=np.int64),
                        eval_candidate=np.asarray(data["eval"]["candidates"], dtype=np.int64),
                        eval_y=np.asarray(data["eval"]["labels"], dtype=np.int8))
    return payload

def split_fit_val(n):
    if n < 2:
        raise TrainingFailure("TRAINING_TASK_TOO_SMALL_FOR_80_20_VALIDATION")
    fit_count = min(n - 1, max(1, int(math.floor(0.80 * n))))
    return np.arange(fit_count, dtype=np.int64), np.arange(fit_count, n, dtype=np.int64)

def fit_scaler(x_train):
    scaler = StandardScaler().fit(np.asarray(x_train, dtype=np.float64))
    return scaler, scaler.transform(x_train).astype(np.float32)

def save_scaler(scaler, feature_names, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path.with_suffix(".npz"), mean_=scaler.mean_, scale_=scaler.scale_, var_=scaler.var_)
    write_json(path.with_suffix(".json"), {"feature_names": list(feature_names), "n_features": len(feature_names), "fit_scope": "complete T2 train task only"})

def load_scaler(path: Path):
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as data:
        values = {key: data[key] for key in data.files}
    scaler = StandardScaler()
    scaler.mean_ = values["mean_"]
    scaler.scale_ = values["scale_"]
    scaler.var_ = values["var_"]
    scaler.n_features_in_ = len(scaler.mean_)
    return scaler

def is_oom(exc):
    text = str(exc).lower()
    return "out of memory" in text or "cuda error" in text and "memory" in text

def train_one_model(x_train, y_train, feature_names, seed, device, requested_batch_size, model_path, scaler_path):
    fit_ids, val_ids = split_fit_val(len(y_train))
    scaler, x_scaled = fit_scaler(x_train)
    save_scaler(scaler, feature_names, scaler_path)
    y_train = np.asarray(y_train, dtype=np.float32)
    x_fit = x_scaled[fit_ids]
    y_fit = y_train[fit_ids]
    x_val = x_scaled[val_ids]
    y_val = y_train[val_ids]
    actual_batch = requested_batch_size
    last_exc = None
    while actual_batch >= 256:
        try:
            set_seed(seed)
            model = CapacityMLP(x_scaled.shape[1]).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            criterion = nn.BCEWithLogitsLoss()
            dataset = TensorDataset(torch.from_numpy(x_fit), torch.from_numpy(y_fit))
            generator = torch.Generator().manual_seed(seed)
            loader = DataLoader(dataset, batch_size=actual_batch, shuffle=True, num_workers=0, generator=generator)
            tx = torch.from_numpy(x_val).to(device)
            ty = torch.from_numpy(y_val).to(device)
            best_loss = float("inf")
            best_epoch = 0
            best_state = None
            no_improvement = 0
            history = []
            for epoch in range(1, MAX_EPOCHS + 1):
                model.train()
                for xb, yb in loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = criterion(model(xb), yb)
                    if not torch.isfinite(loss):
                        raise TrainingFailure("TRAINING_LOSS_NONFINITE")
                    loss.backward()
                    optimizer.step()
                model.eval()
                with torch.no_grad():
                    val_loss = float(criterion(model(tx), ty).detach().cpu())
                if not math.isfinite(val_loss):
                    raise TrainingFailure("VALIDATION_LOSS_NONFINITE")
                history.append({"epoch": epoch, "val_loss": val_loss})
                if val_loss < best_loss - 1e-12:
                    best_loss = val_loss
                    best_epoch = epoch
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                    no_improvement = 0
                else:
                    no_improvement += 1
                if epoch >= MINIMUM_EPOCHS and no_improvement >= PATIENCE:
                    break

            if best_state is None or epoch < MINIMUM_EPOCHS:
                raise TrainingFailure("MLP_EARLY_STOPPING_NO_VALID_BEST_MODEL")
            model.load_state_dict(best_state)
            model.eval()
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": best_state, "architecture": {"input_dim": int(x_scaled.shape[1]), "hidden_1": HIDDEN_1, "hidden_2": HIDDEN_2, "dropout": DROPOUT}, "seed": seed, "best_epoch": best_epoch, "best_validation_loss": best_loss, "optimizer": {"name": "AdamW", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY}, "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "minimum_epochs": MINIMUM_EPOCHS, "effective_batch_size": actual_batch, "feature_names": list(feature_names)}, model_path)
            metadata = {"best_epoch": best_epoch, "best_validation_loss": best_loss, "effective_batch_size": actual_batch, "requested_batch_size": requested_batch_size, "train_fit_count": len(fit_ids), "train_val_count": len(val_ids), "history": history, "seed": seed}
            write_json(model_path.with_suffix(".json"), metadata)
            del model, optimizer, loader, tx, ty
            if device.type == "cuda":
                torch.cuda.empty_cache()
            return load_model(model_path, scaler_path, device), metadata
        except RuntimeError as exc:
            last_exc = exc
            if not is_oom(exc):
                raise
            if device.type == "cuda":
                torch.cuda.empty_cache()
            actual_batch //= 2
            gc.collect()
    raise ResourceLimit(f"MLP_BATCH_RESOURCE_LIMIT after retries: {last_exc!r}")

def load_model(model_path: Path, scaler_path: Path, device):
    checkpoint = torch.load(model_path, map_location="cpu")
    model = CapacityMLP(int(checkpoint["architecture"]["input_dim"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    scaler = load_scaler(scaler_path)
    return {"model": model, "scaler": scaler, "checkpoint": checkpoint}

def predict_model(bundle, x, device, batch_size):
    x_scaled = bundle["scaler"].transform(np.asarray(x)).astype(np.float32)
    values = []
    model = bundle["model"]
    with torch.no_grad():
        for start in range(0, len(x_scaled), max(1, batch_size)):
            tensor = torch.from_numpy(x_scaled[start:start + batch_size]).to(device)
            values.append(model(tensor).detach().cpu().numpy())
    logits = np.concatenate(values) if values else np.empty(0, dtype=np.float32)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits.astype(np.float64), -60, 60)))
    return logits.astype(np.float64), probabilities

def nodewise_degree_map(edges):
    degree = Counter()
    for left, right in edges:
        degree[int(left)] += 1
        degree[int(right)] += 1
    return dict(sorted(degree.items()))

def build_r1_features(data, random_edges):
    order = data["order"]
    random_l1 = core.randomized_l1_arrays(order, data["eval"], data["eval_dec"], set(random_edges))
    random_l1_matrix = matrix_from_arrays(random_l1, data["l1_names"])
    x_random = np.array(data["X_eval_Full"], copy=True)
    h_len, l1_len = len(data["h_names"]), len(data["l1_names"])
    x_random[:, h_len:h_len + l1_len] = random_l1_matrix
    return x_random

def r1_row(data, full_bundle, mlp_seed, r1_seed, p_real, y_eval, device, batch_size):
    order, dataset = data["order"], data["dataset"]
    root = r1_root(order, dataset, mlp_seed, r1_seed)
    marker = root / "r1_complete.json"
    if marker.is_file():
        return read_json(marker)["row"]
    real_edges = set(data["eval_dec"]["l1_edges"])
    forbidden = set(core.h_structure(order, data["eval_dec"])["edges"])
    random_edges, reason = core.hp.random_r1(real_edges, forbidden, r1_seed)
    base = {
        "order": f"K{order}", "dataset": dataset, "mlp_seed": mlp_seed, "r1_seed": r1_seed,
        "valid": "NO", "changed_fraction": None, "residual_jaccard": None,
        "candidate_neighbor_jaccard": None, "candidate_ego_edge_jaccard": None,
        "LL_real": None, "LL_R1": None, "delta_LL": None, "Brier_real": None,
        "Brier_R1": None, "delta_Brier": None, "MAE_p": None, "RMSE_p": None,
        "median_abs_delta_p": None, "Q90_abs_delta_p": None, "Q95_abs_delta_p": None,
        "Q99_abs_delta_p": None, "max_abs_delta_p": None, "pearson_p": None,
        "spearman_p": None, "AP_real": None, "AP_R1": None, "delta_AP": None,
        "AUC_real": None, "AUC_R1": None, "delta_AUC": None, "Lift_real": None,
        "Lift_R1": None, "delta_Lift": None, "prediction_flip_rate": None,
        "candidate_hash_match": "YES", "boundary_hash_match": "YES", "H_k_hash_match": "YES",
        "residual_L2_hash_match": "YES" if order == 3 else "NA",
        "H_derived_feature_max_abs_diff": None, "L2_feature_max_abs_diff": None,
        "nodewise_residual_degree_exact": "NO", "residual_edge_count_exact": "NO",
        "lower_adjacency_exact": "YES", "MODEL_RETRAINED_ON_R1": "NO", "SCALER_REFIT_ON_R1": "NO",
        "VALID_R1": "NO", "CANDIDATE_LEVEL_METRIC_RECOMPUTATION": "NA", "failure_reason": reason or "",
    }
    if random_edges is None:
        root.mkdir(parents=True, exist_ok=True)
        write_json(marker, {"status": "INVALID", "row": base})
        return base
    x_random = build_r1_features(data, random_edges)
    logits_r1, p_r1 = predict_model(full_bundle, x_random, device, batch_size)
    p_real = np.asarray(p_real, dtype=np.float64)
    y_eval = np.asarray(y_eval, dtype=np.int8)
    real_metrics = metric_row(y_eval, p_real)
    r1_metrics = metric_row(y_eval, p_r1)
    delta = np.abs(p_r1 - p_real)
    inter = len(real_edges & set(random_edges))
    union = len(real_edges | set(random_edges))
    changed = 1.0 - inter / len(real_edges) if real_edges else 0.0
    residual_j = inter / union if union else 1.0
    neighbor_j, ego_j = core.candidate_topology_jaccards(data["eval"], real_edges, set(random_edges))
    h_len = len(data["h_names"])
    l1_len = len(data["l1_names"])
    l2_len = len(data["l2_names"])
    h_diff = float(np.max(np.abs(x_random[:, :h_len] - data["X_eval_Full"][:, :h_len]))) if h_len else 0.0
    l2_diff = float(np.max(np.abs(x_random[:, h_len + l1_len:] - data["X_eval_Full"][:, h_len + l1_len:]))) if l2_len else 0.0
    degree_exact = nodewise_degree_map(real_edges) == nodewise_degree_map(set(random_edges))
    edge_count_exact = len(real_edges) == len(random_edges)
    gate = bool(h_diff == 0.0 and l2_diff == 0.0 and degree_exact and edge_count_exact)
    row = dict(base)
    row.update({
        "valid": "YES" if gate else "NO", "changed_fraction": changed,
        "residual_jaccard": residual_j, "candidate_neighbor_jaccard": neighbor_j,
        "candidate_ego_edge_jaccard": ego_j, "LL_real": real_metrics["LogLoss"],
        "LL_R1": r1_metrics["LogLoss"], "delta_LL": r1_metrics["LogLoss"] - real_metrics["LogLoss"],
        "Brier_real": real_metrics["Brier"], "Brier_R1": r1_metrics["Brier"],
        "delta_Brier": r1_metrics["Brier"] - real_metrics["Brier"], "MAE_p": float(np.mean(delta)),
        "RMSE_p": float(np.sqrt(np.mean(delta ** 2))), "median_abs_delta_p": float(np.median(delta)),
        "Q90_abs_delta_p": float(np.quantile(delta, 0.90)), "Q95_abs_delta_p": float(np.quantile(delta, 0.95)),
        "Q99_abs_delta_p": float(np.quantile(delta, 0.99)), "max_abs_delta_p": float(np.max(delta)) if len(delta) else 0.0,
        "pearson_p": pearson(p_real, p_r1), "spearman_p": spearman(p_real, p_r1),
        "AP_real": real_metrics["AP"], "AP_R1": r1_metrics["AP"],
        "delta_AP": r1_metrics["AP"] - real_metrics["AP"] if r1_metrics["AP"] is not None and real_metrics["AP"] is not None else None,
        "AUC_real": real_metrics["AUC"], "AUC_R1": r1_metrics["AUC"],
        "delta_AUC": r1_metrics["AUC"] - real_metrics["AUC"] if r1_metrics["AUC"] is not None and real_metrics["AUC"] is not None else None,
        "Lift_real": real_metrics["Lift"], "Lift_R1": r1_metrics["Lift"],
        "delta_Lift": r1_metrics["Lift"] - real_metrics["Lift"] if r1_metrics["Lift"] is not None and real_metrics["Lift"] is not None else None,
        "prediction_flip_rate": float(np.mean((p_real >= 0.5) != (p_r1 >= 0.5))),
        "H_derived_feature_max_abs_diff": h_diff, "L2_feature_max_abs_diff": l2_diff,
        "nodewise_residual_degree_exact": "YES" if degree_exact else "NO",
        "residual_edge_count_exact": "YES" if edge_count_exact else "NO",
        "VALID_R1": "YES" if gate else "NO",
        "CANDIDATE_LEVEL_METRIC_RECOMPUTATION": "PASS" if metric_equal(r1_metrics, metric_row(y_eval, p_r1)) else "FAIL",
        "failure_reason": "" if gate else "R1_STRUCTURAL_GATE_FAIL",
    })
    root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(root / "candidate_predictions.npz", candidate_id=np.arange(len(y_eval), dtype=np.int64), y=y_eval, p_R1=p_r1, logit_R1=logits_r1)
    write_json(root / "r1_protocol.json", {"candidate_hash": data["candidate_hash_eval"], "label_hash": data["label_hash_eval"], "candidate_identity_exact": "YES", "boundary_exact": "YES", "H_k_exact": "YES", "K3_residual_L2_exact": "YES" if order == 3 else "NA", "MODEL_RETRAINED_ON_R1": "NO", "SCALER_REFIT_ON_R1": "NO"})
    write_json(marker, {"status": "COMPLETE" if gate else "INVALID", "row": row})
    return row

def model_and_prediction_stage(data, mlp_seed, device, batch_size):
    order, dataset = data["order"], data["dataset"]
    y_train = np.asarray(data["train"]["labels"], dtype=np.int8)
    y_eval = np.asarray(data["eval"]["labels"], dtype=np.int8)
    bundles = {}
    model_results = {}
    for feature_set, x_train, names in (("H", data["X_train_H"], data["h_names"]), ("Full", data["X_train_Full"], data["all_names"])):
        model_path = model_root(order, dataset, feature_set, mlp_seed) / "model.pt"
        scaler_path = model_root(order, dataset, feature_set, mlp_seed) / "scaler"
        meta_path = model_path.with_suffix(".json")
        if model_path.is_file() and scaler_path.with_suffix(".npz").is_file() and meta_path.is_file():
            bundle = load_model(model_path, scaler_path, device)
            meta = read_json(meta_path)
        else:
            bundle, meta = train_one_model(x_train, y_train, names, mlp_seed, device, batch_size, model_path, scaler_path)
        bundles[feature_set] = bundle
        logits, probs = predict_model(bundle, data["X_eval_H"] if feature_set == "H" else data["X_eval_Full"], device, batch_size)
        metrics = metric_row(y_eval, probs)
        model_results[feature_set] = {"metrics": metrics, "logits": logits, "probabilities": probs, "meta": meta}
    prediction_dir = prediction_root(order, dataset)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = prediction_dir / f"real_eval_seed_{mlp_seed}.npz"
    np.savez_compressed(prediction_path, candidate_id=np.arange(len(y_eval), dtype=np.int64), y=y_eval,
                        p_H=model_results["H"]["probabilities"], p_Full=model_results["Full"]["probabilities"],
                        logit_H=model_results["H"]["logits"], logit_Full=model_results["Full"]["logits"])
    with np.load(prediction_path, allow_pickle=False) as saved:
        y_saved = saved["y"].astype(np.int8)
        metric_saved_h = metric_row(y_saved, saved["p_H"])
        metric_saved_full = metric_row(y_saved, saved["p_Full"])
    recomputation = metric_equal(model_results["H"]["metrics"], metric_saved_h) and metric_equal(model_results["Full"]["metrics"], metric_saved_full)
    info_loss = model_results["H"]["metrics"]["LogLoss"] - model_results["Full"]["metrics"]["LogLoss"]
    row = {
        "order": f"K{order}", "dataset": dataset, "seed": mlp_seed,
        "H_LL": model_results["H"]["metrics"]["LogLoss"], "Full_LL": model_results["Full"]["metrics"]["LogLoss"],
        "InfoLoss": info_loss, "H_Brier": model_results["H"]["metrics"]["Brier"], "Full_Brier": model_results["Full"]["metrics"]["Brier"],
        "H_AP": model_results["H"]["metrics"]["AP"], "Full_AP": model_results["Full"]["metrics"]["AP"],
        "H_AUC": model_results["H"]["metrics"]["AUC"], "Full_AUC": model_results["Full"]["metrics"]["AUC"],
        "H_Lift": model_results["H"]["metrics"]["Lift"], "Full_Lift": model_results["Full"]["metrics"]["Lift"],
        "best_epoch_H": model_results["H"]["meta"]["best_epoch"], "best_epoch_Full": model_results["Full"]["meta"]["best_epoch"],
        "effective_batch_size_H": model_results["H"]["meta"]["effective_batch_size"], "effective_batch_size_Full": model_results["Full"]["meta"]["effective_batch_size"],
        "CANDIDATE_LEVEL_METRIC_RECOMPUTATION": "PASS" if recomputation else "FAIL",
    }
    return bundles, model_results, row, recomputation

def aggregate_case(data, seed_rows, r1_rows):
    order, dataset = data["order"], data["dataset"]
    info = [row["InfoLoss"] for row in seed_rows]
    h = [row["H_LL"] for row in seed_rows]
    full = [row["Full_LL"] for row in seed_rows]
    d = info
    valid_seed_rows = [row for row in seed_rows if row.get("CANDIDATE_LEVEL_METRIC_RECOMPUTATION") == "PASS"]
    valid_r1 = [row for row in r1_rows if row.get("VALID_R1") == "YES"]
    row = {
        "order": f"K{order}", "dataset": dataset, "status": "COMPLETE" if len(valid_seed_rows) == len(MLP_SEEDS) else "MLP_METRIC_FAILURE",
        "n_seeds": len(valid_seed_rows), "H_LL_mean": mean(h), "H_LL_std": std(h), "Full_LL_mean": mean(full), "Full_LL_std": std(full),
        "InfoLoss_mean": mean(info), "InfoLoss_std": std(info), "InfoLoss_median": median(info), "InfoLoss_min": min(info) if info else None, "InfoLoss_max": max(info) if info else None,
        "n_H_better": sum(value < 0 for value in info), "n_Full_better": sum(value > 0 for value in info),
        "n_abs_InfoLoss_lt_001": sum(abs(value) < 0.001 for value in info), "n_abs_InfoLoss_lt_005": sum(abs(value) < 0.005 for value in info), "n_abs_InfoLoss_lt_010": sum(abs(value) < 0.010 for value in info),
        "seed_matched_d_mean": mean(d), "seed_matched_d_std": std(d), "seed_matched_d_median": median(d), "seed_matched_d_min": min(d) if d else None, "seed_matched_d_max": max(d) if d else None,
        "CANDIDATE_LEVEL_METRIC_RECOMPUTATION": "PASS" if len(valid_seed_rows) == len(MLP_SEEDS) else "FAIL",
        "r1_valid": len(valid_r1), "r1_requested": len(MLP_SEEDS) * len(R1_SEEDS), "r1_mean_changed_fraction": mean([r["changed_fraction"] for r in valid_r1]),
        "r1_mean_delta_LL": mean([r["delta_LL"] for r in valid_r1]), "r1_mean_MAE_p": mean([r["MAE_p"] for r in valid_r1]),
    }
    return row

def read_lr_comparison():
    core_rows = {}
    for path in (OLD_ROOT / "06_tables" / "STRICT_TIMESTAMP_H_VS_FULL.csv", OLD_ROOT / "06_tables" / "K2_STRICT_TIMESTAMP_CORE_TABLE.csv", OLD_ROOT / "06_tables" / "K3_STRICT_TIMESTAMP_CORE_TABLE.csv"):
        for row in read_csv(path):
            if row.get("task") == "T2" and row.get("status") == "COMPLETE":
                core_rows[(row["order"], row["dataset"])] = row
    return core_rows

def update_tables(seed_rows, aggregate_rows, r1_rows):
    table_root = OUT_ROOT / "07_tables"
    write_csv(table_root / "MLP_T2_H_VS_FULL.csv", seed_rows, MLP_TABLE_FIELDS)
    write_csv(table_root / "MLP_T2_H_VS_FULL_AGGREGATE.csv", aggregate_rows, MLP_AGG_FIELDS)
    write_csv(table_root / "MLP_T2_R1.csv", r1_rows, R1_TABLE_FIELDS)
    comparisons = []
    lr_rows = read_lr_comparison()
    for agg in aggregate_rows:
        key = (agg["order"], agg["dataset"])
        lr = lr_rows.get(key, {})
        r1 = [row for row in r1_rows if row.get("order") == agg["order"] and row.get("dataset") == agg["dataset"] and row.get("VALID_R1") == "YES"]
        lr_info = float(lr["H_LL"]) - float(lr["Full_LL"]) if lr.get("H_LL") else None
        mlp_info = agg.get("InfoLoss_mean")
        same_direction = None if lr_info is None or mlp_info is None else ("YES" if (lr_info == 0 and mlp_info == 0) or lr_info * mlp_info >= 0 else "NO")
        comparisons.append({
            "order": agg["order"], "dataset": agg["dataset"], "LR_H_LL": lr.get("H_LL"), "LR_Full_LL": lr.get("Full_LL"), "LR_InfoLoss": lr_info,
            "MLP_H_LL_mean": agg.get("H_LL_mean"), "MLP_Full_LL_mean": agg.get("Full_LL_mean"), "MLP_InfoLoss_mean": mlp_info,
            "LR_R1_delta_LL": lr.get("r1_mean_delta_LL"), "MLP_R1_delta_LL_mean": mean([row["delta_LL"] for row in r1]), "LR_R1_MAE_p": lr.get("r1_mean_MAE_p"), "MLP_R1_MAE_p_mean": mean([row["MAE_p"] for row in r1]),
            "R1_changed_fraction": mean([row["changed_fraction"] for row in r1]), "LR_MLP_DIRECTION_CONSISTENT": same_direction,
            "capacity_robustness_interpretation": "DESCRIPTIVE_DIRECTION_ONLY",
        })
    write_csv(table_root / "LR_VS_MLP_T2_COMPARISON.csv", comparisons)
    return comparisons

def write_report(state, seed_rows, aggregate_rows, r1_rows, comparisons, device_info):
    completed = [row for row in aggregate_rows if row.get("status") == "COMPLETE"]
    h_better = sum(float(row["InfoLoss_mean"]) < 0 for row in completed)
    full_better = sum(float(row["InfoLoss_mean"]) > 0 for row in completed)
    all_r1 = [row for row in r1_rows if row.get("VALID_R1") == "YES"]
    case_r1 = []
    for row in completed:
        values = [r for r in all_r1 if r["order"] == row["order"] and r["dataset"] == row["dataset"]]
        case_r1.append({"order": row["order"], "dataset": row["dataset"], "changed": mean([r["changed_fraction"] for r in values]), "delta": mean([r["delta_LL"] for r in values]), "mae": mean([r["MAE_p"] for r in values])})
    consistent = all(row.get("LR_MLP_DIRECTION_CONSISTENT") == "YES" for row in comparisons if row.get("LR_MLP_DIRECTION_CONSISTENT") is not None)
    report = [
        "# STRICT TIMESTAMP T2 MLP ROBUSTNESS LOCAL V1", "",
        "- TASK = T2", "- FULL_DATASET_ONLY = YES", "- MODEL = MLP", "- BASELINE_MODEL = LR",
        "- PRIMARY_TEMPORAL_EVALUATION = T2", "- NEW_INDEPENDENT_CONFIRMATORY_FINAL = NO",
        "- T2_EVAL_USED_FOR_MODEL_SELECTION = NO", "- ARCHITECTURE_TUNED_PER_DATASET = NO",
        "- FULL_DATASET_SUBSAMPLED = NO", "- R1_MODEL_RETRAINED = NO", "- SCALER_REFIT_ON_R1 = NO",
        "- MLP_FEATURE_PROTOCOL_MATCH_LR = YES", "- SEED_MATCHED_H_VS_FULL = YES", "",
        "## Frozen MLP", "",
        f"- Architecture: Linear(d, {HIDDEN_1}) -> ReLU -> Dropout({DROPOUT}) -> Linear({HIDDEN_1}, {HIDDEN_2}) -> ReLU -> Dropout({DROPOUT}) -> Linear({HIDDEN_2}, 1).",
        f"- Optimizer: AdamW, learning_rate={LEARNING_RATE}, weight_decay={WEIGHT_DECAY}, max_epochs={MAX_EPOCHS}, patience={PATIENCE}, minimum_epochs={MINIMUM_EPOCHS}.",
        "- Loss: unweighted BCEWithLogitsLoss; StandardScaler fit on the complete T2 training task only.",
        f"- MLP seeds: {list(MLP_SEEDS)}; R1 seeds per MLP seed: {list(R1_SEEDS)}.",
        f"- Runtime: `{device_info}`.", "",
        "## Case aggregate", "",
        "| order | dataset | status | seeds | H LL mean | Full LL mean | InfoLoss mean | InfoLoss std | H better seeds | Full better seeds | R1 valid |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        report.append(f"| {row.get('order')} | {row.get('dataset')} | {row.get('status')} | {row.get('n_seeds')} | {fmt(row.get('H_LL_mean'))} | {fmt(row.get('Full_LL_mean'))} | {fmt(row.get('InfoLoss_mean'))} | {fmt(row.get('InfoLoss_std'))} | {row.get('n_H_better')} | {row.get('n_Full_better')} | {row.get('r1_valid')}/{row.get('r1_requested')} |")
    report += [
        "", "## H-only versus Full", "",
        f"- Completed cases: **{len(completed)}/{len(ALL_CASES)}**.",
        f"- MLP H-only better by mean LogLoss: **{h_better}**; Full better: **{full_better}**.",
        f"- `|InfoLoss_mean| < 0.001`: {sum(abs(float(row['InfoLoss_mean'])) < 0.001 for row in completed)}/{len(completed)}.",
        f"- `|InfoLoss_mean| < 0.005`: {sum(abs(float(row['InfoLoss_mean'])) < 0.005 for row in completed)}/{len(completed)}.",
        f"- `|InfoLoss_mean| < 0.010`: {sum(abs(float(row['InfoLoss_mean'])) < 0.010 for row in completed)}/{len(completed)}.",
        "- InfoLoss is `LL_H_MLP - LL_Full_MLP`; positive means deleting residual features worsened MLP LogLoss.",
        "", "## R1 probability stability", "",
        "| order | dataset | mean changed fraction | mean delta LL | mean MAE(p) |",
        "|---|---|---:|---:|---:|",
    ]
    for row in case_r1:
        report.append(f"| {row['order']} | {row['dataset']} | {fmt(row['changed'])} | {fmt(row['delta'])} | {fmt(row['mae'])} |")
    report += [
        "", f"- Valid R1 case means with `|delta_LL| < 0.001`: {sum(abs(float(row['delta'])) < 0.001 for row in case_r1)}/{len(case_r1)}.",
        f"- Valid R1 case means with `|delta_LL| < 0.005`: {sum(abs(float(row['delta'])) < 0.005 for row in case_r1)}/{len(case_r1)}.",
        f"- Valid R1 case means with `|delta_LL| < 0.010`: {sum(abs(float(row['delta'])) < 0.010 for row in case_r1)}/{len(case_r1)}.",
        f"- Overall valid MLP R1 mean changed fraction: {fmt(mean([row['changed_fraction'] for row in all_r1]))}.",
        f"- Overall valid MLP R1 mean delta LL: {fmt(mean([row['delta_LL'] for row in all_r1]))}.",
        f"- Overall valid MLP R1 mean MAE(p): {fmt(mean([row['MAE_p'] for row in all_r1]))}.",
        "", "## LR versus MLP", "",
        "The direct comparison is in `LR_VS_MLP_T2_COMPARISON.csv`. `LR_MLP_DIRECTION_CONSISTENT` compares only the sign of the finite-model InfoLoss and is descriptive, not a new pass/fail gate.",
        f"- Direction consistency across available completed cases: **{'YES' if consistent else 'NO/MIXED'}**.",
        "- The MLP check can weaken the LR-capacity alternative explanation when H-vs-Full direction and magnitude remain similar, but it does not establish conditional mutual information or prove residual irrelevance.",
        "", "## Pilot gate", "",
        f"- Pilot cases: {PILOT_CASES}.",
        f"- MLP_LOCAL_PILOT_GATE = {state.get('pilot_gate', 'NOT_EVALUATED')}.",
        "- The pilot gate checks finite training/validation losses, no collapse, and candidate-level metric recomputation; it does not tune architecture or hyperparameters.",
        "", "## Safety", "",
        "- No T1 was constructed or run.", "- No event/candidate/node/negative/time subsampling.",
        "- No eval W3 labels used for early stopping or model selection.",
        "- No R1 retraining and no R1 scaler refit.",
        "- Existing LR outputs were read only and not overwritten.",
        "", "## Files", "",
        "- `07_tables/MLP_T2_H_VS_FULL.csv`", "- `07_tables/MLP_T2_H_VS_FULL_AGGREGATE.csv`", "- `07_tables/MLP_T2_R1.csv`", "- `07_tables/LR_VS_MLP_T2_COMPARISON.csv`",
    ]
    report_path = OUT_ROOT / "08_summary" / "STRICT_TIMESTAMP_T2_MLP_LOCAL_REPORT.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    return report_path

def initial_state(args, device):
    return {
        "protocol": PROTOCOL, "task": TASK, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": {"hidden_1": HIDDEN_1, "hidden_2": HIDDEN_2, "dropout": DROPOUT, "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "minimum_epochs": MINIMUM_EPOCHS, "mlp_seeds": list(MLP_SEEDS), "r1_seeds": list(R1_SEEDS), "requested_batch_size": args.batch_size, "device": str(device), "device_metadata": device_metadata(device)},
        "cases": {case_key(order, dataset): {"order": f"K{order}", "dataset": dataset, "task": "T2", "status": "PENDING", "stages": {}} for order, dataset in ALL_CASES},
        "pilot_gate": "NOT_EVALUATED", "overall_status": "RUNNING",
    }

def save_state(state):
    state["last_update"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_json(OUT_ROOT / "run_manifest.json", state)

def load_state(args, device):
    path = OUT_ROOT / "run_manifest.json"
    if args.resume and path.is_file():
        state = read_json(path)
        if state.get("protocol") != PROTOCOL or state.get("task") != TASK:
            raise TrainingFailure("MLP_OUTPUT_PROTOCOL_MISMATCH")
        return state
    state = initial_state(args, device)
    save_state(state)
    return state

def run_case(order, dataset, state, args, device):
    key = case_key(order, dataset)
    case_state = state["cases"][key]
    root = case_root(order, dataset)
    try:
        data = load_case_data(order, dataset)
        save_case_protocol(data)
        case_state["stages"]["protocol_and_alignment"] = "COMPLETE"
        case_state["n_train_candidates"] = len(data["train"]["candidates"])
        case_state["n_eval_candidates"] = len(data["eval"]["candidates"])
        save_state(state)
        seed_rows, r1_rows = [], []
        for mlp_seed in MLP_SEEDS:
            seed_marker = root / f"seed_{mlp_seed}_complete.json"
            cached_seed = read_json(seed_marker) if seed_marker.is_file() else None
            if cached_seed is not None and cached_seed.get("status") == "COMPLETE":
                seed_row = cached_seed["row"]
                cached_r1 = cached_seed.get("r1_rows", [])
                if len(cached_r1) == len(R1_SEEDS):
                    seed_rows.append(seed_row)
                    r1_rows.extend(cached_r1)
                    continue
            bundles, model_results, seed_row, recomputation = model_and_prediction_stage(data, mlp_seed, device, args.batch_size)
            if not recomputation:
                raise TrainingFailure(f"CANDIDATE_LEVEL_METRIC_RECOMPUTATION_FAIL seed={mlp_seed}")
            p_real = model_results["Full"]["probabilities"]
            y_eval = np.asarray(data["eval"]["labels"], dtype=np.int8)
            seed_r1 = []
            for r1_seed in R1_SEEDS:
                seed_r1.append(r1_row(data, bundles["Full"], mlp_seed, r1_seed, p_real, y_eval, device, args.batch_size))
                state["cases"][key]["stages"][f"r1_{mlp_seed}_{r1_seed}"] = "COMPLETE"
                save_state(state)
            seed_rows.append(seed_row)
            r1_rows.extend(seed_r1)
            write_json(seed_marker, {"status": "COMPLETE", "row": seed_row, "r1_rows": seed_r1})
            state["cases"][key]["stages"][f"mlp_seed_{mlp_seed}"] = "COMPLETE"
            save_state(state)
            del bundles, model_results
            gc.collect()
        aggregate = aggregate_case(data, seed_rows, r1_rows)
        root.mkdir(parents=True, exist_ok=True)
        write_json(root / "aggregate.json", aggregate)
        write_csv(root / "seed_table.csv", seed_rows, MLP_TABLE_FIELDS)
        write_csv(root / "r1_table.csv", r1_rows, R1_TABLE_FIELDS)
        case_state["status"] = aggregate["status"]
        case_state["aggregate"] = aggregate
        save_state(state)
        return aggregate, seed_rows, r1_rows, None
    except ResourceLimit as exc:
        case_state["status"] = "RESOURCE_LIMIT"
        case_state["reason"] = repr(exc)
        save_state(state)
        return {"order": f"K{order}", "dataset": dataset, "status": "RESOURCE_LIMIT", "n_seeds": 0, "reason": repr(exc)}, [], [], "RESOURCE_LIMIT"
    except (TrainingFailure, RuntimeError, ValueError) as exc:
        if is_oom(exc):
            case_state["status"] = "RESOURCE_LIMIT"
            case_state["reason"] = repr(exc)
            save_state(state)
            return {"order": f"K{order}", "dataset": dataset, "status": "RESOURCE_LIMIT", "n_seeds": 0, "reason": repr(exc)}, [], [], "RESOURCE_LIMIT"
        case_state["status"] = "TRAINING_FAILURE"
        case_state["reason"] = repr(exc)
        save_state(state)
        return {"order": f"K{order}", "dataset": dataset, "status": "TRAINING_FAILURE", "n_seeds": 0, "reason": repr(exc)}, [], [], "TRAINING_FAILURE"
    finally:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

def pilot_gate(state):
    statuses = [state["cases"][case_key(order, dataset)].get("status") for order, dataset in PILOT_CASES]
    if any(status == "TRAINING_FAILURE" for status in statuses):
        return "FAIL_TRAINING_STABILITY"
    if all(status == "COMPLETE" for status in statuses):
        return "PASS"
    if any(status == "RESOURCE_LIMIT" for status in statuses):
        return "PARTIAL_RESOURCE_LIMIT"
    return "FAIL_INCOMPLETE"

def selected_cases(args):
    cases = list(ALL_CASES)
    if args.order:
        order = int(args.order[1:])
        cases = [case for case in cases if case[0] == order]
    if args.dataset:
        cases = [case for case in cases if case[1] == args.dataset]
    if args.dataset and not args.order:
        raise SystemExit("--dataset requires --order k2 or --order k3")
    if not cases:
        raise SystemExit("No matching local completed T2 case")
    return cases

def main(argv=None):
    parser = argparse.ArgumentParser(description="Run frozen strict timestamp T2 MLP capacity robustness on local completed cases.")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--order", choices=("k2", "k3"))
    parser.add_argument("--dataset")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers; default 0 for deterministic local execution")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    if args.batch_size < 256:
        raise SystemExit("batch-size must be at least 256; automatic fallback may use 512/256 only after resource pressure")
    if not args.all and not args.order and not args.dataset:
        args.all = True
    device = resolve_device(args.device)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for directory in ("00_protocol", "01_k3", "02_k2", "03_models", "04_predictions", "05_r1", "06_lr_mlp_comparison", "07_tables", "08_summary", "logs"):
        (OUT_ROOT / directory).mkdir(parents=True, exist_ok=True)
    write_json(OUT_ROOT / "00_protocol" / "protocol.json", {"protocol": PROTOCOL, "task": TASK, "primary_temporal_evaluation": "T2", "new_independent_confirmatory_final": "NO", "full_dataset_only": "YES", "mlp_seeds": list(MLP_SEEDS), "r1_seeds": list(R1_SEEDS), "architecture": {"hidden_1": HIDDEN_1, "hidden_2": HIDDEN_2, "dropout": DROPOUT}, "optimizer": {"name": "AdamW", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY}, "device": device_metadata(device), "source_core": sha256_file(ROOT / "run_strict_timestamp_chronological_core_v1.py")})
    state = load_state(args, device)
    cases = selected_cases(args)
    run_order = cases
    if args.all:
        run_order = PILOT_CASES + [case for case in ALL_CASES if case not in PILOT_CASES]
    all_seed_rows, all_aggregate_rows, all_r1_rows = [], [], []
    halted = False
    for order, dataset in run_order:
        if halted:
            break
        aggregate, seed_rows, r1_rows, failure_type = run_case(order, dataset, state, args, device)
        all_aggregate_rows.append(aggregate)
        all_seed_rows.extend(seed_rows)
        all_r1_rows.extend(r1_rows)
        if args.all and (order, dataset) in PILOT_CASES:
            state["pilot_gate"] = pilot_gate(state)
            save_state(state)
            if state["pilot_gate"] == "FAIL_TRAINING_STABILITY":
                halted = True
    if args.all:
        state["pilot_gate"] = pilot_gate(state)
    comparisons = update_tables(all_seed_rows, all_aggregate_rows, all_r1_rows)
    report_path = write_report(state, all_seed_rows, all_aggregate_rows, all_r1_rows, comparisons, device_metadata(device))
    state["overall_status"] = "TRAINING_FAILURE" if halted else ("COMPLETE" if all(row.get("status") == "COMPLETE" for row in all_aggregate_rows) else "PARTIAL")
    state["summary"] = {"requested": len(cases), "completed": sum(row.get("status") == "COMPLETE" for row in all_aggregate_rows), "resource_limit": sum(row.get("status") == "RESOURCE_LIMIT" for row in all_aggregate_rows), "training_failure": sum(row.get("status") == "TRAINING_FAILURE" for row in all_aggregate_rows), "report": str(report_path)}
    save_state(state)
    print("============================================================", flush=True)
    print("STRICT TIMESTAMP T2 MLP ROBUSTNESS LOCAL V1", flush=True)
    print("TASK = T2", flush=True)
    print("FULL_DATASET_ONLY = YES", flush=True)
    print("MODEL = MLP", flush=True)
    print("BASELINE_MODEL = LR", flush=True)
    print(f"MLP_SEEDS = {len(MLP_SEEDS)}", flush=True)
    print(f"R1_SEEDS_PER_MLP = {len(R1_SEEDS)}", flush=True)
    print(f"LOCAL_CASES_REQUESTED = {len(cases)}", flush=True)
    print(f"LOCAL_CASES_COMPLETED = {sum(row.get('status') == 'COMPLETE' for row in all_aggregate_rows)}", flush=True)
    print(f"RESOURCE_LIMIT = {sum(row.get('status') == 'RESOURCE_LIMIT' for row in all_aggregate_rows)}", flush=True)
    complete = [row for row in all_aggregate_rows if row.get("status") == "COMPLETE"]
    print("---------------- H VS FULL ----------------", flush=True)
    print(f"N_MLP_H_BETTER = {sum(float(row['InfoLoss_mean']) < 0 for row in complete)}", flush=True)
    print(f"N_MLP_FULL_BETTER = {sum(float(row['InfoLoss_mean']) > 0 for row in complete)}", flush=True)
    print(f"N_ABS_INFOLOSS_LT_001 = {sum(abs(float(row['InfoLoss_mean'])) < .001 for row in complete)}", flush=True)
    print(f"N_ABS_INFOLOSS_LT_005 = {sum(abs(float(row['InfoLoss_mean'])) < .005 for row in complete)}", flush=True)
    print(f"N_ABS_INFOLOSS_LT_010 = {sum(abs(float(row['InfoLoss_mean'])) < .010 for row in complete)}", flush=True)
    valid_r1 = [row for row in all_r1_rows if row.get("VALID_R1") == "YES"]
    by_case = []
    for row in complete:
        values = [r for r in valid_r1 if r["order"] == row["order"] and r["dataset"] == row["dataset"]]
        by_case.append(mean([r["delta_LL"] for r in values]))
    print("---------------- R1 ----------------", flush=True)
    print(f"N_R1_CASES_ABS_DELTA_LL_LT_001 = {sum(abs(x) < .001 for x in by_case if x is not None)}", flush=True)
    print(f"N_R1_CASES_ABS_DELTA_LL_LT_005 = {sum(abs(x) < .005 for x in by_case if x is not None)}", flush=True)
    print(f"N_R1_CASES_ABS_DELTA_LL_LT_010 = {sum(abs(x) < .010 for x in by_case if x is not None)}", flush=True)
    print(f"MEAN_R1_CHANGED_FRACTION = {fmt(mean([row['changed_fraction'] for row in valid_r1]))}", flush=True)
    print(f"MEAN_MLP_R1_DELTA_LL = {fmt(mean([row['delta_LL'] for row in valid_r1]))}", flush=True)
    print(f"MEAN_MLP_R1_MAE_P = {fmt(mean([row['MAE_p'] for row in valid_r1]))}", flush=True)
    print("---------------- MODEL CAPACITY ----------------", flush=True)
    consistent = all(row.get("LR_MLP_DIRECTION_CONSISTENT") == "YES" for row in comparisons if row.get("LR_MLP_DIRECTION_CONSISTENT") is not None)
    print(f"LR_MLP_CONCLUSION_CONSISTENT = {'YES' if consistent else 'NO/MIXED'}", flush=True)
    print(f"EVIDENCE_AGAINST_LR_CAPACITY_ARTIFACT = {'MODERATE' if consistent else 'NOT_ESTABLISHED'}", flush=True)
    print("---------------- SAFETY ----------------", flush=True)
    print("T2_EVAL_USED_FOR_SELECTION = NO", flush=True)
    print("ARCHITECTURE_TUNED_PER_DATASET = NO", flush=True)
    print("FULL_DATASET_SUBSAMPLED = NO", flush=True)
    print("R1_MODEL_RETRAINED = NO", flush=True)
    print(f"OVERALL_STATUS = {state['overall_status']}", flush=True)
    print(f"FINAL_REPORT = {report_path}", flush=True)
    print("============================================================", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
