# K2-K4 task-core experiments

This bundle publishes experiment source code and configurations. Frozen candidate/feature, baseline, model and scaler resources required by its procedures are not distributed here. The commands below assume those external resources have already been supplied; this bundle is not a self-contained supplementary experiment reproduction.

Install `requirements.txt` into the environment used for the model package.
Run commands from this directory:

```bash
python run.py --check
python run.py --order 2 --dataset email-Eu
python run.py --order 3 --dataset tags-ask-ubuntu
python run.py --order 4 --dataset email-Eu
python run.py --order 4 --experiment retention
python run.py --order 4 --experiment perturbation --dataset email-Eu
```

Omit `--dataset` to run the order's full panel. `--device cpu` or `--device cuda` selects the device. Results are written under `output/`; completed cases can resume there.

K2: contact-high-school, contact-primary-school, email-Enron, email-Eu, NDC-substances. K3: email-Eu, NDC-substances, tags-math-sx, tags-ask-ubuntu, threads-math-sx. K4: email-Eu, tags-ask-ubuntu, tags-math-sx. K4's NDC-classes, congress-bills and DAWN stages are in the sibling `k4_server_benchmark_expansion_bundle_v1` component.

These are strict timestamp-block T2 structural task-core experiments. The primary task-core comparison uses LogLoss with the frozen 0.02 noninferiority margin, five seeds 21001-21005 and 2,000 paired bootstrap replicates. AP uses `sklearn.metrics.average_precision_score`; Brier and ROC AUC are additional diagnostics. The separate heuristic benchmark uses AP.

Prepare canonical event streams in `dataset/raw_data/`. Prepare `inputs/` with candidate-aligned features, split identities, baseline predictions and model/scaler inputs read by the frozen task-core procedure. The K2 MLP B1/Full and structurally equivalent K3 baseline branches reuse these inputs; the other required branches train during a run. K4 strict nested saturation loads the H4 and standard-residual Full checkpoints and trains its cumulative C1*-C3* branches. These experiment input files are not included in this code distribution and must be prepared separately.


## External resource layout

Paths below are relative to this bundle and correspond to the assignments in run.py.

| Resource | Required content / consumer |
|---|---|
| dataset/raw_data/ | Canonical timestamp, event-size and flattened-node streams for the selected dataset. |
| inputs/strict_t2/ | Frozen timestamp-block candidates, aligned features, labels and split identities used by the task-core feature loader. |
| inputs/mlp/ | K2 B1/Full baseline prediction, model and scaler archives reused by the K2 runner. |
| inputs/k3_old/ and inputs/k3_panel_source.json | Existing K3 baseline inputs and panel source identities. |
| inputs/timestamp_audit_cache.json and inputs/tags_ask_protocol.json | Timestamp audit and tags-ask-ubuntu protocol metadata read by the K3/K4 runners. |
| inputs/k4_standard/, inputs/k4_scopes/ and inputs/k4_structural_nesting.csv | K4 standard features/checkpoints, scope-specific features and structural nesting identities. |

These frozen resources have no bundled download location or complete regeneration recipe in this source release. Public raw events alone do not reconstruct their candidate, feature, baseline and checkpoint identities. Their absence limits the scope of this distribution to source inspection and integration with separately supplied experiment resources. The --check command validates dependencies and bundled source hashes only.

Server task manifests distinguish execution_plan=SERVER from the actual execution_status. A scheduled optional case does not imply an observed resource failure; resource_limit_observed is true only when its recorded status is RESOURCE_LIMIT. Completed cases are excluded from the server-required list.
