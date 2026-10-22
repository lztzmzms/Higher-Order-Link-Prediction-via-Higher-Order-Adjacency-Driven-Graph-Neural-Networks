from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

BUNDLE = Path(__file__).resolve().parents[1]
LOCAL = BUNDLE / "local_results" / "completed" / "k3_final_5dataset"
SERVER = BUNDLE / "server_results" / "k3_remaining"
FINAL = BUNDLE / "server_results" / "final" / "k3_final_5dataset"
PANEL = ["email-Eu", "NDC-substances", "tags-math-sx", "tags-ask-ubuntu", "threads-math-sx"]

def combine(name: str, server_name: str | None = None) -> pd.DataFrame:
    parts = []
    local_path = LOCAL / name
    if local_path.is_file(): parts.append(pd.read_csv(local_path))
    server_path = SERVER / (server_name or name)
    if server_path.is_file(): parts.append(pd.read_csv(server_path))
    if not parts: raise FileNotFoundError(name)
    frame = pd.concat(parts, ignore_index=True)
    key = [value for value in ("dataset", "model", "seed", "reference") if value in frame.columns]
    return frame[frame.dataset.isin(PANEL)].drop_duplicates(key or None, keep="last")

def main() -> None:
    FINAL.mkdir(parents=True, exist_ok=True)
    by_seed = combine("K3_FINAL_5DATASET_SATURATION_BY_SEED.csv", "K3_TASK_CORE_SATURATION_BY_SEED.csv")
    aggregate = combine("K3_FINAL_5DATASET_SATURATION_AGGREGATED.csv", "K3_TASK_CORE_SATURATION_AGGREGATED.csv")
    bootstrap = combine("K3_FINAL_5DATASET_BOOTSTRAP.csv", "K3_TASK_CORE_SATURATION_BOOTSTRAP.csv")
    by_seed.to_csv(FINAL / "K3_FINAL_5DATASET_SATURATION_BY_SEED.csv", index=False)
    aggregate.to_csv(FINAL / "K3_FINAL_5DATASET_SATURATION_AGGREGATED.csv", index=False)
    bootstrap.to_csv(FINAL / "K3_FINAL_5DATASET_BOOTSTRAP.csv", index=False)
    panel = pd.read_csv(LOCAL / "K3_FINAL_5DATASET_PANEL_MANIFEST.csv").set_index("dataset")
    server_panel_path = SERVER / "K3_8DATASET_PANEL_MANIFEST.csv"
    if server_panel_path.is_file():
        server_panel = pd.read_csv(server_panel_path).set_index("dataset")
        for dataset in ("tags-ask-ubuntu", "threads-math-sx"):
            if dataset in server_panel.index:
                for column, value in server_panel.loc[dataset].items(): panel.loc[dataset, column] = value
    panel.reset_index().to_csv(FINAL / "K3_FINAL_5DATASET_PANEL_MANIFEST.csv", index=False)
    missing = sorted(set(PANEL) - set(aggregate.dataset))
    if missing: raise RuntimeError(f"FINAL_PANEL_INCOMPLETE: {missing}")
    lines = ["# K3 Final Five-Dataset Saturation Report", "", "- K3_FINAL_PANEL_SIZE = 5", "- N_COMPLETED = 5", "- N_PENDING_SERVER = 0", "- Excluded from all aggregates: email-Enron, contact-high-school, contact-primary-school.", "", "| dataset | B0 | B1 | B2 | B3 | B4 | Full | minimum saturation | CI upper | classification |", "|---|---:|---:|---:|---:|---:|---:|---|---:|---|"]
    for row in aggregate.set_index("dataset").loc[PANEL].itertuples():
        lines.append(f"| {row.Index} | {row.LogLoss_B0:.6f} | {row.LogLoss_B1:.6f} | {row.LogLoss_B2:.6f} | {row.LogLoss_B3:.6f} | {row.LogLoss_B4:.6f} | {row.LogLoss_B5:.6f} | {row.minimum_saturation_model} | {row.saturation_ci_upper:.6f} | {row.classification} |")
    (FINAL / "K3_FINAL_5DATASET_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"merged final K3 panel: {len(aggregate)} datasets")

if __name__ == "__main__":
    main()
