# Eight-dataset server runs

Run from the extracted repository root on a Linux server. All eight datasets use their current component-joint method. Dual-Mapping uses GAT; MLSHoL uses SparseSIN. NDC/threads use the two-member HistoryNet transfer recipe. No supplementary experiment resources are needed for these main-model runs.

## Environment

Use Python 3.11 or 3.12. For an NVIDIA server, the following uses the official PyTorch 2.4.1 CUDA 11.8 wheels. CUDA 12.1 and 12.4 wheels are also listed in the [official version-specific installation instructions](https://pytorch.org/get-started/previous-versions/#v241).

```bash
conda create -n holp python=3.11 -y
conda activate holp
python -m pip install --upgrade pip
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
python -m pip check
python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
python scripts/check_repository.py
```

The repository supplies the torch_scatter operations used by these models. No separate torch-scatter compilation is required. CPU execution is available with --device cpu. The batch runner defaults to serial jobs so local-complex memory and first-use input extraction are not shared between simultaneous workers.

## Run the configured seed lists

The following commands use the three seeds listed for each model in [the dataset settings](DATASET_CONFIGURATIONS.md), for 48 model fits in total. Omit the seed flag to apply the JSON lists.

```bash
for dataset in email-Enron email-Eu contact-high-school contact-primary-school tags-ask-ubuntu tags-math-sx NDC-substances threads-math-sx; do
  python -u run.py --dataset "$dataset" --model both --device cuda:0 --output "server_results/configured_seeds/$dataset" || exit 1
done
```

The batch commands below explicitly request contiguous seed ranges for the original server protocol.

## Ten independent runs per model and dataset

This means eight datasets, two models, ten downstream seeds 9101-9110: 160 model fits. Each fit retains 25 complete training epochs. A worker handles one dataset and seed, generating historical components once for both models. The launcher therefore starts 80 serial workers.

Preview the plan without creating outputs or training:

```bash
python scripts/run_server.py --runs 10 --epochs 25 --seed-start 9101 --device cuda:0 --output server_results/ten_runs --dry-run
```

Start the full job and retain its launcher log after disconnecting:

```bash
nohup python -u scripts/run_server.py --runs 10 --epochs 25 --seed-start 9101 --device cuda:0 --output server_results/ten_runs > server_ten_runs.log 2>&1 &
echo $! > server_ten_runs.pid
tail -f server_ten_runs.log
```

The output directory must be new on the first invocation. Omitting --epochs retains the per-model JSON training budgets, currently 25 for all eight datasets. --runs changes the number of downstream seeds. It does not change upstream predictor seeds, historical windows or predictor training budgets: the HistoryNet members still use seeds 41/42 and 20 epochs.

## If ten rounds means ten training epochs

Use this separate command for one seed per model and dataset, with ten complete downstream epochs: 16 model fits in eight workers.

```bash
nohup python -u scripts/run_server.py --runs 1 --epochs 10 --seed-start 9101 --device cuda:0 --output server_results/ten_epochs > server_ten_epochs.log 2>&1 &
echo $! > server_ten_epochs.pid
```

To use both ten independent runs and ten training epochs, combine --runs 10 --epochs 10. The upstream component-generation budgets remain unchanged in every case.

The ordinary single-dataset entry also supports an epoch override:

```bash
python run.py --dataset NDC-substances --model both --seed 9101 --epochs 10 --device cuda:0 --output server_results/ndc_one_seed
```

## Results and resume

The batch output contains:

- plan.json: datasets, model names, seeds, effective epochs and source/configuration/input hashes.
- environment.json: Python and installed dependency versions.
- run_metrics.csv: candidate-level checked AP and Lift for every completed model/seed.
- summary.csv and summary.json: means, sample standard deviations (ddof=1), completed counts and requested counts for all dataset/model pairs. Standard deviation is blank for a single completed seed.
- <dataset>/seed_<seed>/attempt_0001/: the ordinary run.py outputs, including predictions, weights, configuration, historical components, preprocessing and training histories.
- <dataset>/seed_<seed>/attempt_0001.log, status.json and complete.json: worker logs and completion status.

A result is counted only after both requested models finish and their outputs pass identity, prediction-level AP/Lift, checkpoint-presence and full-epoch-history checks. The candidate IDs and labels must match the fixed dataset evaluation cohort. Failures are recorded and the next worker continues; the launcher exits with a nonzero status if any requested model run remains incomplete. Partial means are always accompanied by actual counts.

Resume with exactly the same settings and source/input files:

```bash
nohup python -u scripts/run_server.py --runs 10 --epochs 25 --seed-start 9101 --device cuda:0 --output server_results/ten_runs --resume > server_ten_runs_resume.log 2>&1 &
```

Resume revalidates completed dataset/seed pairs and skips them. An interrupted or failed pair restarts in a new attempt directory; earlier logs and artifacts are preserved. This is job-level resume, not restoration of optimizer state halfway through an epoch. If the second model failed, the entire pair is rerun so it shares a freshly generated component cohort. Source/configuration/input changes cause resume to refuse the old plan; use a new output directory for a different experiment.

Run only one launcher per output directory. run.lock records its host and PID. After an ungraceful system shutdown, remove a stale lock only after confirming that its launcher and worker have stopped.

Optional --datasets email-Enron email-Eu selects a subset; --model dual or --model mlshol selects one model. Defaults select all eight datasets and both models. With multiple physical GPUs, setting CUDA_VISIBLE_DEVICES=1 makes --device cuda:0 refer to physical GPU 1.
