# Higher-Order Link Prediction

Project: [Higher-Order Link Prediction via Higher-Order Adjacency-Driven Graph Neural Networks](https://github.com/lztzmzms/Higher-Order-Link-Prediction-via-Higher-Order-Adjacency-Driven-Graph-Neural-Networks).

Graph and simplicial models for K=2 higher-order link prediction (three-node candidate links). **Dual-Mapping** encodes the candidate-derived graph with GAT; **MLSHoL** encodes local simplicial complexes with SparseSIN/MPSN. Both use higher-order structure, random-walk (RW) representations, and auxiliary structural features.

All eight datasets default to `component_joint`, combining structural embeddings with positively encoded historical predictor and rule outputs. NDC-substances and threads-math-sx use an explicit transfer of the tags-math-sx two-member HistoryNet recipe, implemented by the `historynet_joint` pipeline. This transfer defines their current method; it does not reconstruct a pre-existing historical H predictor.

## Requirements

Use Python 3.11 or 3.12. Dependency versions are specified in `requirements.txt`; software checks have used Python 3.12, PyTorch 2.4.1, PyG 2.6.1, NumPy 1.26.4, and GUDHI 3.13.0. Install a PyTorch build compatible with your device, then install the remaining dependencies:

```sh
python -m pip install -r requirements.txt
```

The bundled `torch_scatter` module implements the operations required by the included dense-index simplicial pipeline. It is not a complete replacement for all torch-scatter APIs. Supplementary experiments have additional requirements in `supplementary_sources/requirements.txt`.

## Datasets

| Dataset | Default mode | Component dimension |
|---|---|---|
| email-Enron | component_joint | 7 |
| email-Eu | component_joint | 5 |
| contact-high-school | component_joint | 4 |
| contact-primary-school | component_joint | 3 |
| tags-ask-ubuntu | component_joint | 5 |
| tags-math-sx | component_joint | 2 |
| NDC-substances | component_joint | 2 |
| threads-math-sx | component_joint | 2 |

Each configuration is in `configs/<dataset>.json`. Input events, fixed candidate cohorts, packed local complexes, structural features, and native protocol archives are organized under `data/<dataset>/`. Public raw data can be obtained from [ScHoLP Data](https://www.cs.cornell.edu/~arb/data/) or its [data repository](https://github.com/arbenson/ScHoLP-Data). See [data preparation and protocol notes](docs/DATA.md) and [dataset settings](docs/DATASET_CONFIGURATIONS.md).

New event data may be chronological JSONL records (`{"timestamp": 1, "nodes": ["a", "b", "c"]}`) or NPZ arrays `times`, `counts`, and flattened `vertices`. Node IDs cannot repeat within one event.

## Usage

Run commands from the repository root. Each output directory must be new.

### Prepare raw events

```sh
python scripts/convert_scholp.py --times /path/to/times.txt --nverts /path/to/nverts.txt --simplices /path/to/simplices.txt --output /path/to/events.npz
```

This converts the public file format. The supplied eight-dataset experiments use their bundled fixed cohorts; the conversion alone does not reconstruct those cohort selections. For new datasets, the unified pipeline generates open-triangle candidates, structural features, RW inputs, and local complexes from event prefixes.

### Generate historical components

```sh
python scripts/generate_components.py --dataset email-Enron --output outputs/enron_components
python scripts/generate_components.py --dataset NDC-substances --output outputs/ndc_components
python scripts/generate_components.py --dataset threads-math-sx --output outputs/threads_components
```

The same generation command supports all eight configured datasets. It rebuilds real supervised predictor members and rule outputs from historical events. For NDC-substances and threads-math-sx, `prepared/` contains aligned cohorts and structural inputs; `components/` contains raw member logits, historical windows, the scaled arrays, recipe metadata and preprocessing parameters. It does not require a precomputed component-score file. `run.py` performs this step automatically for `component_joint`. Component schemas and time constraints are described in [COMPONENT_SOURCES.md](docs/COMPONENT_SOURCES.md).

For serial server batches across all eight datasets, ten independent seeds or a ten-epoch budget, see [server commands and resume instructions](docs/SERVER_RUN.md). The batch entry is scripts/run_server.py.

### Train, predict, and evaluate

```sh
python run.py --dataset email-Enron --model dual --seed 9119 --output outputs/enron_dual
python run.py --dataset tags-ask-ubuntu --model mlshol --seed 9115 --output outputs/ubuntu_mlshol
python run.py --dataset email-Eu --mode structure_only --model both --output outputs/eu_structure
python run.py --dataset NDC-substances --model both --output outputs/ndc
python run.py --dataset threads-math-sx --model both --output outputs/threads
```

These commands train the selected model, generate target predictions, and compute AP and Lift for the bundled evaluation protocol. `--model` accepts `dual`, `mlshol`, or `both`; `--device` accepts `cpu`, `cuda:0`, or `auto`. Omitting `--seed` runs the three downstream seeds configured for each requested dataset and model. The default is `component_joint` for all eight datasets. `--epochs N` overrides only the downstream training budget for a bundled dataset; it leaves historical predictor training unchanged and does not edit its JSON configuration. `--mode structure_only` uses the current structural encoder without historical components. For NDC-substances and threads-math-sx this remains the common GAT/SparseSIN pipeline, rather than restoring the archived native training procedure.

```sh
python run.py --config examples/new_structure_only.json --input /path/to/events.jsonl --model both --output outputs/new_structure
python run.py --config examples/new_component_joint.json --input /path/to/events.jsonl --model dual --output outputs/new_joint
```

A new `component_joint` dataset requires an explicit `module:ComponentGenerator` implementation; see `examples/component_generator_template.py`. No frequency-based replacement is selected automatically. For new data, evaluation is enabled only when the configuration declares a complete observed future horizon.

## Reproducibility

All eight datasets use the exact settings in their JSON files: 25 complete epochs, microbatch size 32, representation width 32, three message-passing layers, and three downstream seeds per model. See [dataset settings](docs/DATASET_CONFIGURATIONS.md) for the seed lists and their selection protocol. Each epoch visits all N training candidates and performs one optimizer update. Dual performs one full-graph forward and one backward over the cohort loss. MLSHoL accumulates gradients over separate microbatch forwards; BatchNorm statistics and Dropout follow those microbatches, so these gradients are not equivalent to a single full-cohort forward. The batch_size setting controls MLSHoL microbatches; Dual uses the complete candidate graph. Training histories distinguish configured batch_size from effective_batch_size and record forward_batches/backward_calls. Both models therefore use 25N candidate visits and 25 updates. NDC-substances and threads-math-sx retain the archived fit/validation IDs and final evaluation sample IDs. Their current joint models use the shared weighted BCE, two-step candidate diffusion RW and the common training settings above; the archived native engines are outside the default route. The transferred upstream HistoryNet uses member seeds 41/42, width 128, 20 epochs and historical weighted BCE, separately from the downstream seed lists.

Outputs include candidate IDs, predictions, model weights, configuration, component preprocessing parameters, `metrics.json`, and per-seed training histories. AP is `sklearn.metrics.average_precision_score`; Lift is AP divided by prevalence in the same candidate set. Ablation configurations and experimental settings are described in the paper.

Software contract tests and a small end-to-end pipeline check are available separately:

```sh
python -m unittest discover -s tests -p 'test_*.py'
python scripts/verify_runtime.py --output outputs/software_checks.json
python scripts/check_repository.py
```

Software checks establish execution behavior, not independent evidence of benchmark accuracy. Data/cohort and recipe-selection conditions that affect interpretation are documented in [DATA.md](docs/DATA.md).

## Implementation Details

- `models/dual.py` constructs the candidate-derived graph and GAT layers. `models/network.py` reads structural embeddings and combines branches.
- `mp/` and `simplicial_complex/` provide SparseSIN/MPSN operators, local simplex labels, adjacency, and readout.
- `historical_components/generators/` exposes all eight dataset routes through `create_generator`; `upstream/` contains the original feature, supervised-model and rule definitions. `pipelines/historynet_joint.py` supplies the NDC/threads HistoryNet transfer and its cohort alignment.
- Component scores use fixed-origin average percentile ranks, training-only StandardScaler parameters, and clipping to [-8,8]. Activity indicators are separate contextual inputs.
- PositiveLinear uses softplus-constrained, independently initialized weights; Softplus activations preserve positive score directions. A nonlinear joint head combines the component and structural representations. Training uses true-link weighted BCE, without an aggregate historical score input or output residual.

Supplementary higher-order and heuristic experiments are under `supplementary_sources/`. These supplementary bundles publish source code and configurations; they require external experiment resources listed in their own documentation and are not a self-contained release of those supplementary experiments. Licensing information is provided in `LICENSE` and `THIRD_PARTY_NOTICES.txt`.
