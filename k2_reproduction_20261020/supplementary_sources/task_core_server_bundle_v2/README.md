# K2-K4 task-core experiments

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

`dataset/raw_data/` contains canonical event streams. `inputs/` holds candidate-aligned features, split identities, baseline predictions and model/scaler inputs read by the frozen task-core procedure. The K2 MLP B1/Full and structurally equivalent K3 baseline branches reuse these inputs; the other required branches train during a run. K4 strict nested saturation loads the H4 and standard-residual Full checkpoints and trains its cumulative C1*-C3* branches. These files are required experiment inputs.
