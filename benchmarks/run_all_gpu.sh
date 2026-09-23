#!/usr/bin/env bash
# Regenerate every benchmark result in benchmarks/results/ from scratch.
#
#   cd ~/code/LigandMPNN && conda activate ligandmpnn_env
#   bash benchmarks/run_all_gpu.sh
#
# run.py selects its own device (run.py:162, torch.cuda.is_available), so no
# flag is needed; the check below only reports what you got. GPU and CPU runs
# do NOT produce identical sequences at the same seed -- different kernel
# reduction orders change the draws -- so regenerate everything on one device
# rather than mixing.
#
# Override sizes if needed:
#   POOL=1024 N_DESIGNS=32 bash benchmarks/run_all_gpu.sh
#
# Figures are NOT built here: they need pandas/matplotlib, which
# ligandmpnn_env does not have. Run benchmarks/make_figures.py separately.
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
unset PYTHONSAFEPATH 2>/dev/null || true
export PYTHONPATH="$PWD"
PY="${PYTHON:-python}"

POOL="${POOL:-384}"            # rejection-sampling pool per backbone
MIN_BIN="${MIN_BIN:-25}"       # min pool hits before a charge bin is compared
N_DESIGNS="${N_DESIGNS:-24}"   # sequences per cell, depth arms
N_SMALL="${N_SMALL:-16}"       # sequences per cell, breadth arms
PANEL_DESIGNS="${PANEL_DESIGNS:-32}"

echo "=== device ==="
"$PY" - <<'PYEOF'
import torch
ok = torch.cuda.is_available()
print(f"torch {torch.__version__}  cuda_available={ok}")
print(f"device: {torch.cuda.get_device_name(0)}" if ok else
      "WARNING: no CUDA visible -- this will run on CPU and take far longer.")
PYEOF

echo
echo "=== 1/3  structures (idempotent) ==="
"$PY" benchmarks/fetch_panel.py

echo
echo "=== 2/3  exactness + A280 across the 16-backbone panel ==="
"$PY" benchmarks/bench_panel.py --n-designs "$PANEL_DESIGNS" --n-baseline 32

echo
echo "=== 3/3  rejection reference, model types, tolerance, A280, SPPS ==="
"$PY" benchmarks/bench_minimal.py --pool "$POOL" --min-bin "$MIN_BIN" \
      --n-designs "$N_DESIGNS" --n-small "$N_SMALL"

echo
echo "Done. Results in benchmarks/results/. Now build figures in an"
echo "environment with pandas + matplotlib:"
echo "    python benchmarks/make_figures.py"
