import argparse
import csv
import json
import os
import random
import time
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn import metrics as met
from tqdm import tqdm

from mp.models import SparseSIN
from root_dir import ROOT_DIR
from simplicial_complex.data_loading import DataLoader
from simplicial_complex.simplex_utils import convert_graph_dataset_with_gudhi


DEFAULT_DATASETS = [
    "contact-high-school",
    "contact-primary-school",
    "email-Enron",
    "email-Eu",
    "NDC-substances",
    "tags-ask-ubuntu",
    "tags-math-sx",
    "threads-ask-ubuntu",
]


def combinations_to_labels(distance_sum):
    offset = 0
    triples = []
    labels = []

    def generate_partitions(total, result, count, output):
        if total < 0:
            return
        if total == 0:
            part = result[:count]
            if len(part) == 3:
                output.append(part)
            return
        value = 1 if count == 0 else result[count - 1]
        while value <= total:
            result[count] = value
            generate_partitions(total - value, result, count + 1, output)
            value += 1

    for radius in range(3, distance_sum + 1):
        partitions = []
        generate_partitions(radius, [None] * radius, 0, partitions)
        for index, part in enumerate(partitions):
            triples.append(part)
            labels.append(1 + offset + index + 1)
        offset += len(partitions)
    return triples, labels


def remove_target_link(k_hop_subgraph, tri):
    graph = nx.Graph(k_hop_subgraph)
    graph.remove_edges_from(list(combinations(tri, 2)))
    return graph


def trnl_node_labeling(k_hop_subgraph, tri):
    node_1, node_2, node_3 = tuple(tri)
    graph = remove_target_link(k_hop_subgraph, tri)
    node_list = list(graph.nodes)
    index_1 = node_list.index(node_1)
    index_2 = node_list.index(node_2)
    index_3 = node_list.index(node_3)

    dist2node_1 = [nx.shortest_path_length(graph, source=node_1, target=node) for node in node_list]
    dist2node_2 = [nx.shortest_path_length(graph, source=node_2, target=node) for node in node_list]
    dist2node_3 = [nx.shortest_path_length(graph, source=node_3, target=node) for node in node_list]
    dist2node_1[index_2] = 0
    dist2node_1[index_3] = 0
    dist2node_2[index_1] = 0
    dist2node_2[index_3] = 0
    dist2node_3[index_1] = 0
    dist2node_3[index_2] = 0

    labels = [None] * len(node_list)
    for index in range(len(node_list)):
        distances = sorted([dist2node_1[index], dist2node_2[index], dist2node_3[index]])
        triples, trnl_labels = combinations_to_labels(sum(distances))
        labels[index] = [trnl_labels[triples.index(distances)]] if distances in triples else [0]
    labels[index_1] = [1]
    labels[index_2] = [1]
    labels[index_3] = [1]
    return labels


def extract_k_hop_subgraph(
    graph,
    num_hops,
    tri,
    sample_ratio=1.0,
    max_nodes_per_step=None,
    enforce_node_cap=False,
):
    visited = set(tri)
    current_level = set(tri)
    for _ in range(1, num_hops + 1):
        next_level = set()
        for node in current_level:
            for neighbor in graph.neighbors(node):
                if neighbor not in visited and neighbor not in next_level:
                    next_level.add(neighbor)
        if sample_ratio < 1.0:
            next_level = set(random.sample(list(next_level), int(sample_ratio * len(next_level))))
        if max_nodes_per_step is not None and len(next_level) > max_nodes_per_step:
            next_level = set(random.sample(list(next_level), max_nodes_per_step))
        if not next_level:
            break
        if enforce_node_cap:
            visited.update(next_level)
        else:
            for node in current_level:
                for neighbor in graph.neighbors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
        current_level = next_level
    return nx.subgraph(graph, visited)


def construct_labeled_networkx_graph(k_hop_subgraph, old_simplices, tri, positives, labelling="TRNL"):
    if labelling != "TRNL":
        raise ValueError(f"Only TRNL is supported by this runner, got {labelling}")
    node_list = list(k_hop_subgraph.nodes)
    edge_list = list(k_hop_subgraph.edges)
    embeddings = trnl_node_labeling(k_hop_subgraph, tri)
    nx.set_node_attributes(
        k_hop_subgraph,
        {node_list[index]: {"feature": embeddings[index]} for index in range(len(embeddings))},
    )

    simplices = []
    node_set = set(node_list)
    for old_simplex in old_simplices:
        if len(old_simplex) >= 3:
            simplex = sorted(set(old_simplex) & node_set)
            if len(simplex) >= 3:
                simplices.append(tuple(simplex))
        simplices = list(set(simplices))
    simplices = [list(simplex) for simplex in simplices]
    for edge in edge_list:
        simplices.insert(0, sorted(list(edge)))

    original = simplices.copy()
    for i in range(len(original)):
        for simplex in original[i + 1 :]:
            if set(original[i]) < set(simplex):
                simplices[i] = [-1]
            if set(original[i]) > set(simplex):
                simplices[original.index(simplex)] = [-1]
    simplices = [simplex for simplex in simplices if simplex != [-1]]
    simplices.sort(key=len)
    return nx.Graph(k_hop_subgraph, y=1 if tri in positives else 0, simplices=simplices)


def to_pyg_data(k_hop_subgraph):
    node_list = list(k_hop_subgraph.nodes)
    node_feature = [k_hop_subgraph.nodes[node]["feature"] for node in node_list]
    directed = k_hop_subgraph.to_directed() if not nx.is_directed(k_hop_subgraph) else k_hop_subgraph
    graph = nx.convert_node_labels_to_integers(directed)
    nodes_new = list(graph.nodes)
    edges_new = list(graph.edges)
    edge_index = torch.tensor(edges_new, dtype=torch.long).t().contiguous()
    x = torch.tensor(node_feature, dtype=torch.float)
    simplices = [list(simplex) for simplex in k_hop_subgraph.graph["simplices"]]
    for simplex in simplices:
        for i in range(len(simplex)):
            simplex[i] = nodes_new[node_list.index(simplex[i])]
        simplex.sort()
    return Data(
        x=x,
        edge_index=edge_index,
        y=torch.tensor([k_hop_subgraph.graph["y"]], dtype=torch.long),
        node_ids=torch.tensor(nodes_new),
        num_nodes=len(node_list),
        simplices=simplices,
    )


def read_int_lines(path):
    values = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                values.append(int(text))
    return values


def get_timestamped_simplices(root_nverts, root_timestamp, root_simplices):
    nverts = read_int_lines(root_nverts)
    timestamps = read_int_lines(root_timestamp)
    flat_simplices = read_int_lines(root_simplices)
    split_simplices = []
    offset = 0
    for count in nverts:
        simplex = flat_simplices[offset : offset + count]
        simplex.sort()
        split_simplices.append(simplex)
        offset += count
    return timestamps, split_simplices


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_triangles(path):
    triangles = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.replace(",", " ").split()
            if len(parts) < 3:
                continue
            tri = tuple(sorted((int(parts[0]), int(parts[1]), int(parts[2]))))
            triangles.append(tri)
    return triangles


def node_label_transform(graph):
    mapping = {node: int(node) for node in graph.nodes()}
    return nx.relabel_nodes(graph, mapping, copy=False)


def sample_uniform(items, cap, seed):
    items = list(items)
    if cap is None or len(items) <= cap:
        return items
    rng = np.random.default_rng(seed)
    indexes = rng.choice(len(items), size=cap, replace=False)
    return [items[int(i)] for i in indexes]


def sample_stratified(candidates, positives, cap, positive_fraction, seed):
    candidates = list(candidates)
    if cap is None or len(candidates) <= cap:
        return candidates
    rng = np.random.default_rng(seed)
    positive_items = [item for item in candidates if item in positives]
    negative_items = [item for item in candidates if item not in positives]
    desired_pos = min(len(positive_items), int(round(cap * positive_fraction)))
    desired_neg = min(len(negative_items), cap - desired_pos)
    if desired_pos + desired_neg < cap:
        remaining_pos = len(positive_items) - desired_pos
        extra_pos = min(remaining_pos, cap - desired_pos - desired_neg)
        desired_pos += extra_pos
    chosen = []
    if desired_pos:
        chosen.extend([positive_items[int(i)] for i in rng.choice(len(positive_items), size=desired_pos, replace=False)])
    if desired_neg:
        chosen.extend([negative_items[int(i)] for i in rng.choice(len(negative_items), size=desired_neg, replace=False)])
    rng.shuffle(chosen)
    return chosen


def raw_paths(dataset):
    raw_dir = Path(ROOT_DIR) / "dataset" / "raw_data" / dataset
    return (
        raw_dir / f"{dataset}-nverts.txt",
        raw_dir / f"{dataset}-times.txt",
        raw_dir / f"{dataset}-simplices.txt",
    )


def get_old_simplices(dataset, split):
    nverts_path, times_path, simplices_path = raw_paths(dataset)
    _, data_simplices = get_timestamped_simplices(
        str(nverts_path), str(times_path), str(simplices_path)
    )
    end = int((0.6 if split == "train" else 0.8) * len(data_simplices))
    return data_simplices[:end]


def split_paths(dataset, split):
    split_dir = Path(ROOT_DIR) / "dataset" / split
    return {
        "graph": split_dir / f"{dataset}_G_{split}.graphml",
        "open": split_dir / f"{dataset}_openTriangles_{split}.txt",
        "positive": split_dir / f"{dataset}_new_closures_{split}.txt",
    }


def build_pyg_samples(dataset, split, args):
    paths = split_paths(dataset, split)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing protocol files for {dataset}/{split}: {missing}")

    graph = node_label_transform(nx.read_graphml(paths["graph"]))
    old_simplices = get_old_simplices(dataset, split)
    candidates = read_triangles(paths["open"])
    positives = set(read_triangles(paths["positive"]))

    raw_cap = args.train_raw_cap if split == "train" else args.test_raw_cap
    complex_cap = args.max_train_complexes if split == "train" else args.max_test_complexes
    if args.sampling_mode == "stratified":
        fraction = args.train_positive_fraction if split == "train" else args.test_positive_fraction
        sampled_candidates = sample_stratified(
            candidates,
            positives,
            min(raw_cap, complex_cap) if raw_cap is not None and complex_cap is not None else complex_cap or raw_cap,
            fraction,
            args.seed + (100 if split == "train" else 1100),
        )
    else:
        sampled_candidates = sample_uniform(candidates, raw_cap, args.seed + (0 if split == "train" else 1000))
        sampled_candidates = sample_uniform(
            sampled_candidates,
            complex_cap,
            args.seed + (100 if split == "train" else 1100),
        )

    pyg_samples = []
    skipped = 0
    positive_count = 0
    iterator = tqdm(sampled_candidates, desc=f"{dataset} {split} graphs", leave=False)
    for tri in iterator:
        try:
            subgraph = extract_k_hop_subgraph(
                graph,
                args.num_hops,
                tri,
                sample_ratio=args.sample_ratio,
                max_nodes_per_step=args.max_nodes_per_step,
                enforce_node_cap=args.enforce_node_cap,
            )
            subgraph_without_target = remove_target_link(subgraph, tri)
            if not nx.is_connected(subgraph_without_target):
                skipped += 1
                continue

            label_graph = subgraph if split == "train" else subgraph_without_target
            labeled = construct_labeled_networkx_graph(
                label_graph,
                old_simplices,
                tri,
                list(positives),
                labelling=args.labelling_method,
            )
            data = to_pyg_data(labeled)
            y = 1 if tri in positives else 0
            data.y = torch.tensor([y], dtype=torch.long)
            positive_count += y
            pyg_samples.append(data)
        except Exception:
            skipped += 1
            if args.strict:
                raise

    stats = {
        "candidate_count": len(candidates),
        "positive_candidate_count": len(positives),
        "sampled_candidate_count": len(sampled_candidates),
        "valid_graph_count": len(pyg_samples),
        "skipped_graph_count": skipped,
        "valid_positive_count": positive_count,
        "valid_negative_count": len(pyg_samples) - positive_count,
    }
    return pyg_samples, stats


def cache_path(dataset, split, args):
    cache_dir = Path(args.output_dir) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    name = (
        f"{dataset}_{split}_seed{args.seed}_hop{args.num_hops}"
        f"_maxstep{args.max_nodes_per_step}_cap{int(args.enforce_node_cap)}"
        f"_{args.sampling_mode}_pf{args.train_positive_fraction if split == 'train' else args.test_positive_fraction:g}"
        f"_n{args.max_train_complexes if split == 'train' else args.max_test_complexes}.pt"
    )
    return cache_dir / name


def load_or_build_complexes(dataset, split, args):
    path = cache_path(dataset, split, args)
    if path.exists() and not args.rebuild_cache:
        payload = torch.load(path, map_location="cpu")
        return payload["complexes"], payload["stats"]

    pyg_samples, stats = build_pyg_samples(dataset, split, args)
    complexes = convert_graph_dataset_with_gudhi(
        pyg_samples,
        expansion_dim=args.max_dim,
        include_down_adj=False,
        init_method=args.init_method,
    )
    torch.save({"complexes": complexes, "stats": stats}, path)
    return complexes, stats


def one_hot(labels, device):
    labels = labels.view(-1).long().to(device)
    return F.one_hot(labels, num_classes=2).float()


def make_class_weights(complexes, device):
    labels = [int(item.y.view(-1)[0].item()) for item in complexes]
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    total = counts.sum()
    weights = total / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float, device=device)


def compute_loss(out, targets, criterion, args, class_weights=None):
    if args.loss_mode == "bce":
        return criterion(out, targets)
    bce = F.binary_cross_entropy(out, targets, reduction="none")
    if args.loss_mode == "weighted_bce":
        if class_weights is not None:
            bce = bce * class_weights.view(1, -1)
        return bce.mean()
    if args.loss_mode == "focal":
        pt = torch.where(targets == 1, out, 1 - out).clamp(1e-6, 1 - 1e-6)
        focal = (1 - pt).pow(args.focal_gamma) * bce
        if class_weights is not None:
            focal = focal * class_weights.view(1, -1)
        return focal.mean()
    raise ValueError(f"Unsupported loss mode: {args.loss_mode}")


def evaluate_model(model, loader, criterion, device, args, class_weights=None):
    model.eval()
    total_loss = 0.0
    total_items = 0
    y_true = []
    y_prob = []
    y_pred = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            labels = data.y.view(-1).long()
            targets = one_hot(labels, device)
            loss = compute_loss(out, targets, criterion, args, class_weights)
            total_loss += loss.item() * labels.numel()
            total_items += labels.numel()
            probs = out[:, 1].detach().cpu().numpy()
            preds = out.argmax(dim=1).detach().cpu().numpy()
            y_prob.extend(probs.tolist())
            y_pred.extend(preds.tolist())
            y_true.extend(labels.detach().cpu().numpy().tolist())

    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_prob_np = np.asarray(y_prob, dtype=np.float64)
    y_pred_np = np.asarray(y_pred, dtype=np.int64)
    baseline = float(y_true_np.mean()) if len(y_true_np) else 0.0

    if len(np.unique(y_true_np)) == 2:
        precision, recall, _ = met.precision_recall_curve(y_true_np, y_prob_np)
        auc_pr = float(met.auc(recall, precision))
        ap = float(met.average_precision_score(y_true_np, y_prob_np))
        f1 = float(met.f1_score(y_true_np, y_pred_np, zero_division=0))
        precision_score = float(met.precision_score(y_true_np, y_pred_np, zero_division=0))
        recall_score = float(met.recall_score(y_true_np, y_pred_np, zero_division=0))
    else:
        auc_pr = float("nan")
        ap = float("nan")
        f1 = float("nan")
        precision_score = float("nan")
        recall_score = float("nan")

    acc = float((y_true_np == y_pred_np).mean()) if len(y_true_np) else float("nan")
    class0_mask = y_true_np == 0
    class1_mask = y_true_np == 1
    acc0 = float((y_pred_np[class0_mask] == 0).mean()) if class0_mask.any() else float("nan")
    acc1 = float((y_pred_np[class1_mask] == 1).mean()) if class1_mask.any() else float("nan")
    return {
        "loss": total_loss / max(total_items, 1),
        "auc_pr": auc_pr,
        "ap": ap,
        "baseline": baseline,
        "lift": auc_pr / baseline if baseline > 0 and not np.isnan(auc_pr) else float("nan"),
        "accuracy": acc,
        "class0_accuracy": acc0,
        "class1_accuracy": acc1,
        "f1": f1,
        "precision": precision_score,
        "recall": recall_score,
        "count": int(len(y_true_np)),
        "positives": int(y_true_np.sum()) if len(y_true_np) else 0,
        "negatives": int(len(y_true_np) - y_true_np.sum()) if len(y_true_np) else 0,
    }


def train_dataset(dataset, args, device):
    train_complexes, train_stats = load_or_build_complexes(dataset, "train", args)
    test_complexes, test_stats = load_or_build_complexes(dataset, "test", args)
    if not train_complexes or not test_complexes:
        raise RuntimeError(f"{dataset}: no valid train/test complexes")

    train_loader = DataLoader(train_complexes, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_complexes, batch_size=args.batch_size, shuffle=False)

    model = SparseSIN(
        num_input_features=args.num_input_features,
        num_layers=args.num_layers,
        num_classes=2,
        hidden=args.hidden,
        max_dim=args.max_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.5)
    criterion = torch.nn.BCELoss().to(device)
    class_weights = make_class_weights(train_complexes, device) if args.loss_mode in {"weighted_bce", "focal"} else None

    best = None
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_items = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            labels = data.y.view(-1).long()
            loss = compute_loss(out, one_hot(labels, device), criterion, args, class_weights)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * labels.numel()
            train_items += labels.numel()
        scheduler.step()

        if epoch == args.epochs or epoch % args.eval_every == 0:
            metrics = evaluate_model(model, test_loader, criterion, device, args, class_weights)
            metrics["epoch"] = epoch
            metrics["train_loss"] = train_loss / max(train_items, 1)
            best = metrics if best is None or metrics["auc_pr"] >= best["auc_pr"] else best
            print(
                f"{dataset} epoch={epoch} train_loss={metrics['train_loss']:.6f} "
                f"test_auc_pr={metrics['auc_pr']:.6f} lift={metrics['lift']:.6f} "
                f"acc={metrics['accuracy']:.6f}",
                flush=True,
            )

    final_metrics = evaluate_model(model, test_loader, criterion, device, args, class_weights)
    final_metrics["train_seconds"] = time.time() - start
    final_metrics["train_valid_graph_count"] = train_stats["valid_graph_count"]
    final_metrics["test_valid_graph_count"] = test_stats["valid_graph_count"]
    final_metrics["train_valid_positive_count"] = train_stats["valid_positive_count"]
    final_metrics["test_valid_positive_count"] = test_stats["valid_positive_count"]
    final_metrics["dataset"] = dataset
    final_metrics["seed"] = args.seed
    final_metrics["lr"] = args.lr
    final_metrics["hidden"] = args.hidden
    final_metrics["num_layers"] = args.num_layers
    final_metrics["epochs"] = args.epochs
    final_metrics["sampling_mode"] = args.sampling_mode
    final_metrics["train_positive_fraction"] = args.train_positive_fraction
    final_metrics["test_positive_fraction"] = args.test_positive_fraction
    final_metrics["loss_mode"] = args.loss_mode
    final_metrics["focal_gamma"] = args.focal_gamma
    final_metrics["best_eval_epoch"] = best["epoch"] if best else args.epochs
    final_metrics["best_eval_auc_pr"] = best["auc_pr"] if best else final_metrics["auc_pr"]
    final_metrics["best_eval_lift"] = best["lift"] if best else final_metrics["lift"]
    final_metrics["sampling"] = json.dumps({"train": train_stats, "test": test_stats}, sort_keys=True)
    return final_metrics


def write_results(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "seed",
        "lr",
        "hidden",
        "num_layers",
        "epochs",
        "sampling_mode",
        "train_positive_fraction",
        "test_positive_fraction",
        "loss_mode",
        "focal_gamma",
        "auc_pr",
        "ap",
        "baseline",
        "lift",
        "accuracy",
        "class0_accuracy",
        "class1_accuracy",
        "f1",
        "precision",
        "recall",
        "loss",
        "count",
        "positives",
        "negatives",
        "best_eval_epoch",
        "best_eval_auc_pr",
        "best_eval_lift",
        "train_valid_graph_count",
        "test_valid_graph_count",
        "train_valid_positive_count",
        "test_valid_positive_count",
        "train_seconds",
        "sampling",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=["email-Enron"])
    parser.add_argument("--output-dir", default=str(Path(ROOT_DIR) / "exp" / "results" / "email_enron_seed44_default"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden", type=int, default=6)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--step-size", type=int, default=50)
    parser.add_argument("--num-input-features", type=int, default=1)
    parser.add_argument("--max-dim", type=int, default=2)
    parser.add_argument("--num-hops", type=int, default=1)
    parser.add_argument("--sample-ratio", type=float, default=1.0)
    parser.add_argument("--max-nodes-per-step", type=int, default=75)
    parser.add_argument("--enforce-node-cap", dest="enforce_node_cap", action="store_true", default=True)
    parser.add_argument("--no-enforce-node-cap", dest="enforce_node_cap", action="store_false")
    parser.add_argument("--train-raw-cap", type=int, default=10000)
    parser.add_argument("--test-raw-cap", type=int, default=2500)
    parser.add_argument("--max-train-complexes", type=int, default=200)
    parser.add_argument("--max-test-complexes", type=int, default=500)
    parser.add_argument("--sampling-mode", choices=["uniform", "stratified"], default="stratified")
    parser.add_argument("--train-positive-fraction", type=float, default=0.2)
    parser.add_argument("--test-positive-fraction", type=float, default=0.034)
    parser.add_argument("--loss-mode", choices=["bce", "weighted_bce", "focal"], default="bce")
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--labelling-method", default="TRNL")
    parser.add_argument("--init-method", default="sum")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    rows = []
    for dataset in args.datasets:
        print(f"=== {dataset} ===", flush=True)
        try:
            row = train_dataset(dataset, args, device)
            rows.append(row)
            write_results(rows, Path(args.output_dir) / "results.csv")
        except Exception as exc:
            rows.append({"dataset": dataset, "sampling": f"ERROR: {exc}"})
            write_results(rows, Path(args.output_dir) / "results.csv")
            print(f"{dataset} failed: {exc}", flush=True)
            if args.strict:
                raise
    print(f"wrote {Path(args.output_dir) / 'results.csv'}", flush=True)


if __name__ == "__main__":
    main()
