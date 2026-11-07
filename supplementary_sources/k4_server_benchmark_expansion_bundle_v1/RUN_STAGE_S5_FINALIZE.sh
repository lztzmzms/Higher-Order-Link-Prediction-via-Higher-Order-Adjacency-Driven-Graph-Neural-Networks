#!/usr/bin/env bash
set -euo pipefail
BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BUNDLE_ROOT"
K4_PY="${K4_PYTHON:-python3}"
mkdir -p logs
"$K4_PY" scripts/k4_server_pipeline.py s5 2>&1 | tee -a logs/S5_finalize.log
