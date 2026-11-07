from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path

BUNDLE = Path(__file__).resolve().parent
PROJECT = BUNDLE / "src" / "project"
K2_DATASETS = ("contact-high-school", "contact-primary-school", "email-Enron", "email-Eu", "NDC-substances")
K3_DATASETS = ("email-Eu", "NDC-substances", "tags-math-sx", "tags-ask-ubuntu", "threads-math-sx")
K4_DATASETS = ("email-Eu", "tags-ask-ubuntu", "tags-math-sx")

def check() -> None:
    errors = []
    required = ("numpy", "scipy", "sklearn", "torch", "pandas", "pyarrow")
    for name in required:
        try:
            importlib.import_module(name)
        except ImportError as exc:
            errors.append(f"dependency unavailable: {name}: {exc}")
    manifest = json.loads((BUNDLE / "manifests" / "HASHES.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["files"].items():
        path = BUNDLE / relative
        if not path.is_file():
            errors.append(f"missing file: {relative}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append(f"hash mismatch: {relative}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"CHECK=PASS FILES={len(manifest['files'])} K2_DATASETS=5 K3_DATASETS=5 K4_DATASETS=3 TRAINING_STARTED=NO")

def configure_modules():
    sys.path.insert(0, str(PROJECT))
    import run_strict_timestamp_t2_mlp_local_v1 as mlp
    import run_strict_timestamp_chronological_core_v1 as strict
    strict.ROOT = BUNDLE
    mlp.ROOT = BUNDLE
    mlp.OLD_ROOT = BUNDLE / "inputs" / "strict_t2"
    return mlp, strict

def main(argv=None):
    parser = argparse.ArgumentParser(description="K2-K4 task-core experiments")
    parser.add_argument("--order", type=int, choices=(2, 3, 4), default=2)
    parser.add_argument("--dataset")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--experiment", choices=("saturation", "retention", "perturbation"), default="saturation")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    panel = {2: K2_DATASETS, 3: K3_DATASETS, 4: K4_DATASETS}[args.order]
    if args.dataset and args.dataset not in panel:
        parser.error(f"--dataset must be one of: {', '.join(panel)}")
    if args.order != 4 and args.experiment != "saturation":
        parser.error("--experiment retention/perturbation require --order 4")
    if args.check:
        check()
        return 0
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    mlp, strict = configure_modules()
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    if args.order == 2:
        import run_k2_b1_b2_task_core_local_v1 as runner
        runner.ROOT = BUNDLE
        runner.OUT = BUNDLE / "output" / "k2"
        runner.OLD_MLP = BUNDLE / "inputs" / "mlp"
        command = ["--resume", "--device", args.device]
        command += ["--dataset", args.dataset] if args.dataset else ["--all"]
        status = runner.main(command)
        if args.dataset:
            return 0 if (runner.OUT / "cases" / args.dataset / "COMPLETE.json").is_file() else 1
        return status
    if args.order == 3:
        from scripts import run_k3_spatial_order_task_core_v1 as runner
        runner.ROOT = BUNDLE
        runner.OUT = BUNDLE / "output" / "k3"
        runner.LOCAL = runner.OUT / "local"
        runner.OLD = BUNDLE / "inputs" / "k3_old"
        runner.PANEL_SOURCE = BUNDLE / "inputs" / "k3_panel_source.json"
        runner.AUDIT_CACHE = BUNDLE / "inputs" / "timestamp_audit_cache.json"
        runner.TAG_PROTOCOL = BUNDLE / "inputs" / "tags_ask_protocol.json"
        selected = args.dataset or ",".join(panel)
        runner.run(argparse.Namespace(datasets=selected, server_mode=True, negative_control_only=False))
        return 0
    from scripts import run_k4_task_core_perturbation_v1 as k4
    k4.ROOT = BUNDLE
    k4.OUT = BUNDLE / "output" / "k4_perturbation"
    k4.LOCAL = k4.OUT / "local"
    k4.AUDIT_CACHE = BUNDLE / "inputs" / "timestamp_audit_cache.json"
    if args.experiment == "perturbation":
        k4.run(argparse.Namespace(datasets=args.dataset or ",".join(panel), audit_only=False, skip_r1=False))
        return 0
    if args.experiment == "retention":
        from scripts import run_task_core_confirmatory_v2 as runner
        runner.ROOT = BUNDLE
        runner.OUT = BUNDLE / "output" / "k4_retention"
        runner.K4_OUT = runner.OUT
        runner.LOCAL = runner.OUT / "local"
        runner.K4_PANEL = (args.dataset,) if args.dataset else panel
        runner.retention_audit(force=True)
        return 0
    from scripts import run_k4_strict_feature_nested_saturation_v2 as runner
    runner.ROOT = BUNDLE
    runner.OUT = BUNDLE / "output" / "k4_nested"
    runner.LOCAL = runner.OUT / "local"
    runner.OLD_OUT = BUNDLE / "inputs" / "k4_standard"
    runner.OLD_FEATURE = runner.OLD_OUT / "feature_cache"
    runner.SCOPE_FEATURE = BUNDLE / "inputs" / "k4_scopes"
    runner.STRUCT_AUDIT = BUNDLE / "inputs" / "k4_structural_nesting.csv"
    runner.DATASETS = (args.dataset,) if args.dataset else panel
    runner.run_experiment()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
