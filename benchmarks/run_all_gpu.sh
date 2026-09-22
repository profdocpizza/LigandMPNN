#!/usr/bin/env bash
# Run every constrained-decoding benchmark, GPU if one is visible.
#
#   cd ~/code/LigandMPNN && conda activate ligandmpnn_env
#   bash benchmarks/run_all_gpu.sh
#
# run.py picks the device itself (run.py:162, torch.cuda.is_available), so
# nothing here forces it -- the check below just reports what you got.
#
# Sizes are tuned for a GPU. Override any of them, e.g.
#   POOL=8192 N_DESIGNS=64 bash benchmarks/run_all_gpu.sh
#
# Everything lands in benchmarks/results/*.csv. Figures are NOT made here --
# they need pandas/matplotlib, which ligandmpnn_env does not have.
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
unset PYTHONSAFEPATH 2>/dev/null || true
export PYTHONPATH="$PWD"
PY="${PYTHON:-python}"

N_DESIGNS="${N_DESIGNS:-32}"     # sequences per backbone per condition
N_BASELINE="${N_BASELINE:-64}"   # unconstrained reference per backbone
POOL="${POOL:-4096}"             # rejection-sampling pool per backbone
MIN_BIN="${MIN_BIN:-40}"         # min pool hits before a charge bin is used

echo "=== device check ==="
"$PY" - <<'PYEOF'
import torch
ok = torch.cuda.is_available()
print(f"torch {torch.__version__}  cuda_available={ok}")
if ok:
    print(f"device: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: no CUDA visible -- this will run on CPU and take hours.")
PYEOF

echo
echo "=== 0/5  backbone panel (fetch, idempotent) ==="
"$PY" benchmarks/fetch_panel.py

echo
echo "=== 1/5  charge + A280 across 16 backbones ==="
"$PY" benchmarks/bench_panel.py --n-designs "$N_DESIGNS" --n-baseline "$N_BASELINE"

echo
echo "=== 2/5  rejection-sampling reference (the slow one) ==="
"$PY" benchmarks/bench_rejection.py --pool "$POOL" --n-designs "$N_DESIGNS" \
      --min-bin "$MIN_BIN"

echo
echo "=== 3/5  charge sweep and tolerance, three model types ==="
"$PY" benchmarks/bench_charge.py all

echo
echo "=== 4/5  extinction, multi-backbone multi-model ==="
"$PY" benchmarks/bench_extinction.py

echo
echo "=== 5/5  SPPS ==="
"$PY" benchmarks/bench_spps.py

echo
echo "=== done -- CSVs written ==="
ls -la benchmarks/results/*.csv
