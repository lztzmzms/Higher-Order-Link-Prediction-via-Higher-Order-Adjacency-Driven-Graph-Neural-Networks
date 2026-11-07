#!/usr/bin/env bash
set -euo pipefail
BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BUNDLE_ROOT"
K4_PY="${K4_PYTHON:-python3}"
mkdir -p logs server_results/k4_server_benchmark_expansion_v1
echo "K4 bundle root: $BUNDLE_ROOT"
"$K4_PY" -c 'import sys; print("Python=" + sys.version.replace("\n", " "))'
"$K4_PY" -c 'import torch; print("PyTorch=" + torch.__version__ + " CUDA=" + str(torch.cuda.is_available())); print("GPU=" + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE"))'
"$K4_PY" scripts/k4_server_pipeline.py s0 2>&1 | tee -a logs/S0_validate.log
bash RUN_STAGE_S1_DATA.sh
bash RUN_STAGE_S2_CENSUS.sh
bash RUN_STAGE_S3_GATE.sh
bash RUN_STAGE_S4_MLP.sh
bash RUN_STAGE_S5_FINALIZE.sh
echo "K4_SERVER_BENCHMARK_EXPANSION_COMPLETE=YES"
