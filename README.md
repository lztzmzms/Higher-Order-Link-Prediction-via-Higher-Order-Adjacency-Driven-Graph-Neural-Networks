# Higher-Order Link Prediction

## Environment

Use Python 3.11 or 3.12 and install the dependencies from the repository root.

```sh
python -m pip install -r requirements.txt
```

## Run

The bundled datasets are email-Enron, email-Eu, contact-high-school,
contact-primary-school, tags-ask-ubuntu, tags-math-sx, NDC-substances,
and threads-math-sx.

```sh
python run.py --dataset email-Enron --model both --device auto --output outputs/enron
python run.py --dataset email-Enron --mode structure_only --model both --device auto --output outputs/enron_structure
python run.py --help
```

Run from the repository root. Each output directory must be new. Dataset
parameters are stored in configs/<dataset>.json.

## Batch execution

```sh
python scripts/run_server.py --runs 10 --epochs 25 --seed-start 9101 --device cuda:0 --output server_results/main --dry-run
python scripts/run_server.py --runs 10 --epochs 25 --seed-start 9101 --device cuda:0 --output server_results/main
```

## Input preparation

```sh
python scripts/convert_scholp.py --help
python scripts/generate_components.py --help
```

See LICENSE, THIRD_PARTY_NOTICES.txt and NOTICE.md for source notices.
