from __future__ import annotations
import argparse, sys
from pathlib import Path

BUNDLE = Path(__file__).resolve().parents[1]
PROJECT = BUNDLE / "src" / "project"
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))
from scripts import run_k3_spatial_order_task_core_v1 as runner

def main() -> None:
    runner.ROOT = BUNDLE
    runner.OUT = BUNDLE / "server_results" / "k3_remaining"
    runner.LOCAL = runner.OUT / "local"
    frozen = BUNDLE / "local_results" / "frozen"
    runner.OLD = frozen / "k3_old"
    runner.PANEL_SOURCE = frozen / "k3_panel_source.json"
    runner.AUDIT_CACHE = frozen / "timestamp_audit_cache.json"
    runner.TAG_PROTOCOL = frozen / "tags_ask_protocol.json"
    runner.strict.ROOT = BUNDLE
    runner.mlp.ROOT = BUNDLE
    args = argparse.Namespace(datasets="tags-ask-ubuntu,threads-math-sx", server_mode=True, negative_control_only=False)
    runner.run(args)

if __name__ == "__main__":
    main()
