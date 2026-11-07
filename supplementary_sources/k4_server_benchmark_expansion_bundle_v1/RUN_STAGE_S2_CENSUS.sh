#!/usr/bin/env bash
set -euo pipefail
BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BUNDLE_ROOT"
K4_PY="${K4_PYTHON:-python3}"
mkdir -p logs
for dataset in NDC-classes congress-bills DAWN; do
  "$K4_PY" scripts/k4_server_pipeline.py s2 --dataset "$dataset" 2>&1 | tee -a "logs/S2_${dataset}.log"
done
