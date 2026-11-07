# Supplementary Experiments

This directory contains heuristic baselines, K2-K4 task-core experiments and K4 benchmark expansion tools. Each bundle retains its own entry points and source layout.

```sh
python -m pip install -r supplementary_sources/requirements.txt
python supplementary_sources/heuristics/run.py --dataset email-Enron --raw-dir /path/to/raw/events --method Simplicial_CN
python supplementary_sources/task_core_server_bundle_v2/run.py --check
python supplementary_sources/k4_server_benchmark_expansion_bundle_v1/scripts/k4_server_pipeline.py s0 --dry-run
```

The task-core and K4 expansion bundles are source-code releases with configurations and entry points. They are not self-contained experiment reproductions: candidate-aligned features, split/baseline archives, models/scalers or raw event inputs required by individual branches are not all supplied. A complete acquisition or regeneration chain for these missing frozen resources is not included. Higher-order tasks require these additional candidate sets, features, splits and checkpoints. Prepare these according to the bundle configuration and data manifests before running the corresponding experiment. The task-core --check command checks dependencies and bundled code hashes, not the availability of all experimental resources. K4 public raw sources are listed in its manifests/K4_LOCAL_EXPANSION_DATA_MANIFEST.json. Ablation configurations and experimental settings are described in the paper.
